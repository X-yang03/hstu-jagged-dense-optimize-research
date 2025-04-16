import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import fbgemm_gpu
import torch.profiler
from fused_jagged_hstu.fused_hstu_op import FusedHSTUOp
from fused_jagged_hstu.fused_simpler_op import FusedHSTUOp_
from fused_hstu_v2.fused_hstu_op_v2 import FusedHSTUOpv2
from fused_hstu_v3.fused_hstu_op_v3 import FusedHSTUOpv3
from fused_jagged_hstu.torch_backward import CustomAttentionFunction

from contextlib import contextmanager
import torch.fx.traceback

@contextmanager
def no_fx_traceback():
    orig = torch.fx.traceback.format_stack
    torch.fx.traceback.format_stack = lambda: []
    try:
        yield
    finally:
        torch.fx.traceback.format_stack = orig

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

seq_len = [128, 120, 256, 260, 512, 510,100, 200, 300, 400]
max_seq = 200
min_seq = 100
n = 0
B = 64
x_offsets = [0]
for i in range(1, B+1):
    rand_seq_len = random.choice(seq_len)
    # rand_seq_len = random.randint(min_seq, max_seq)
    n = max(n, rand_seq_len)
    x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

n += 11
head, d = 8 , 32
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
    #warmup_einsum_attn = FusedHSTUOp_.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    loss = warmup_einsum_attn.sum()
    loss.backward()
    with no_fx_traceback():
        warmup_fused_attn = FusedHSTUOpv3.apply(q1, k1, v1, rab1, attn_mask, head, d, n, x_offsets)
    loss1 = warmup_fused_attn.sum()
    loss1.backward()
print('warm up done')

print('===========================================================')

print('start benchmark')

einsum_forward_time = []
fused_forward_time = []
einsum_backward_time = []
fused_backward_time = []

test_num = 50
for _ in tqdm(range(test_num)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    start_event.record()
    einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
    # einsum_attn = FusedHSTUOp_.apply(q, k, v, rab, attn_mask, head, d, n, x_offsets)
    end_event.record()
    torch.cuda.synchronize()
    einsum_forward_time.append(start_event.elapsed_time(end_event))

    start_event.record()
    with no_fx_traceback():
        fused_attn = FusedHSTUOpv3.apply(q1, k1, v1, rab, attn_mask, head, d, n, x_offsets)
    end_event.record()
    torch.cuda.synchronize()
    fused_forward_time.append(start_event.elapsed_time(end_event))

    attn_true = torch.randn_like(einsum_attn)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(einsum_attn, attn_true)
    start_event.record()
    loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    einsum_backward_time.append(start_event.elapsed_time(end_event))

    criterion1 = nn.CrossEntropyLoss()
    loss1 = criterion1(fused_attn, attn_true)
    start_event.record()
    loss1.backward()
    end_event.record()
    torch.cuda.synchronize()
    fused_backward_time.append(start_event.elapsed_time(end_event))

print("avg einsum forward time: ", sum(einsum_forward_time) / len(einsum_forward_time))
print("avg fused forward time: ", sum(fused_forward_time) / len(fused_forward_time))
print("avg einsum backward time: ", sum(einsum_backward_time) / len(einsum_backward_time))
print("avg fused backward time: ", sum(fused_backward_time) / len(fused_backward_time))

print('===========================================================')

speedup_forward = [einsum_forward_time[i] / fused_forward_time[i] for i in range(len(einsum_forward_time))]
speedup_backward = [einsum_backward_time[i] / fused_backward_time[i] for i in range(len(einsum_backward_time))]

print("avg forward speedup: ", sum(speedup_forward) / len(speedup_forward))
print("avg backward speedup: ", sum(speedup_backward) / len(speedup_backward))
print('===========================================================')
print('benchmark done')

# print(einsum_backward_time)
# print(fused_backward_time)