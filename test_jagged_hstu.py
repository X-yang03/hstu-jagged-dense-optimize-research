import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import fbgemm_gpu
from fused_jagged_hstu.fused_hstu_op import FusedHSTUOp
from fused_hstu_v2.fused_hstu_op_v2 import FusedHSTUOpv2
from fused_hstu_v3.fused_hstu_op_v3 import FusedHSTUOpv3

def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    rab = torch.randn(B, 1, n, n, requires_grad=True, device="cuda")

    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    rab1 = rab.clone().detach().requires_grad_(True)
    
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, rab,  q1, k1, v1, rab1, attn_mask

def origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets):
    padded_q = torch.ops.fbgemm.jagged_to_padded_dense(  #根据x_offsets的位置信息，将q和k转换为padded形式，统一为长为n的序列， [B, n, num_heads*dqk]
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
    padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",  #在attention_dim维度上计算q和k的点积  ,attn形状(B,num_heads,n,n)
            padded_q.view(B, n, head, d),
            padded_k.view(B, n, head, d),
        )           
    qk_attn = qk_attn + rab
    qk_attn = F.silu(qk_attn) / n #SiLU之后局部归一化
    qk_attn = qk_attn * attn_mask
    attn_output = torch.ops.fbgemm.dense_to_jagged( #Φ(qk)v  , dense_to_jagged将输出转换为(sum_N, head*d)形状
            torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                    B, n, head, d  #将v转换为padded形式
                ),
            ).reshape(B, n, head * d), 
            [x_offsets],
        )[0]
    return attn_output

seq_len = [120,128, 256, 512, 1024]
max_seq = 200
min_seq = 100
n = 0
B  = 128
x_offsets = [0]
for i in range(1, B+1):
    # rand_seq_len = random.choice(seq_len)
    rand_seq_len = random.randint(min_seq, max_seq)
    n = max(n, rand_seq_len)
    x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

n += 11  #符合原本hstu的流程
head, d = 2 , 25
sum_N = int(x_offsets[-1])

print('benchmark config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))
print('input q k v & output shape: ({}, {})'.format(sum_N, head*d))
print('input rab shape: ({}, {}, {}, {})'.format(B, 1, n, n))
print('input attn_mask shape: ({}, {}, {}, {})'.format(1, 1, n, n))

print('===========================================================')

print('warm up')
for _ in tqdm(range(3)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)
    warmup_einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
    warmup_fused_attn = FusedHSTUOpv3.apply(q1, k1, v1, rab1, attn_mask, head, d, n, x_offsets)
print('warm up done')

print('===========================================================')

print('start test')

avg_forward_diff = []
max_forward_diff = []
avg_forward = []

avg_backward_diff = []
max_backward_diff = []
avg_backward = []

test_num = 10
for _ in tqdm(range(test_num)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)

    einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)

    fused_attn = FusedHSTUOpv3.apply(q1, k1, v1, rab1, attn_mask, head, d, n, x_offsets)


    #print("forward pass check: ", torch.allclose(einsum_attn, fused_attn, atol=1e-4))
    assert(torch.allclose(einsum_attn, fused_attn, atol=1e-4))
    avg_forward_diff.append(torch.mean(torch.abs(einsum_attn - fused_attn)))
    max_forward_diff.append(torch.max(torch.abs(einsum_attn - fused_attn)))
    avg_forward.append(torch.abs(torch.mean(einsum_attn)))

    y_true = torch.randn_like(einsum_attn)
    
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    loss = criterion(einsum_attn, y_true)

    #loss = einsum_attn.sum()

    
    loss.backward()
    
    loss1 = criterion1(fused_attn, y_true)
    
    loss1.backward()

    avg_backward_diff.append(torch.mean(torch.abs(q.grad - q1.grad)))
    avg_backward_diff.append(torch.mean(torch.abs(k.grad - k1.grad)))
    avg_backward_diff.append(torch.mean(torch.abs(v.grad - v1.grad)))
    avg_backward_diff.append(torch.mean(torch.abs(rab.grad - rab1.grad)))

    max_backward_diff.append(torch.max(torch.abs(q.grad - q1.grad)))
    max_backward_diff.append(torch.max(torch.abs(k.grad - k1.grad)))
    max_backward_diff.append(torch.max(torch.abs(v.grad - v1.grad)))
    max_backward_diff.append(torch.max(torch.abs(rab.grad - rab1.grad)))

    avg_backward.append(torch.abs(torch.mean(q.grad)))
    avg_backward.append(torch.abs(torch.mean(k.grad)))
    avg_backward.append(torch.abs(torch.mean(v.grad)))
    avg_backward.append(torch.abs(torch.mean(rab.grad)))

forward_diff_ratio = [avg_forward_diff[i]/avg_forward[i] for i in range(len(avg_forward_diff))]
print("avg_forward_diff: ", sum(avg_forward_diff)/len(avg_forward_diff))
print("max_forward_diff: ", max(max_forward_diff))
print("diff ratio avg：{}, max: {}".format(sum(forward_diff_ratio)/len(forward_diff_ratio), max(forward_diff_ratio)))

backward_diff_ratio = [avg_backward_diff[i]/avg_backward[i] for i in range(len(avg_backward_diff))]
print("avg_backward_diff: ", sum(avg_backward_diff)/len(avg_backward_diff))
print("max_backward_diff: ", max(max_backward_diff))
print("diff ratio avg：{}, max: {}".format(sum(backward_diff_ratio)/len(backward_diff_ratio), max(backward_diff_ratio)))