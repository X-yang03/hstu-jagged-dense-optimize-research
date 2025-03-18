import torch
import torch.nn.functional as F
from tqdm import tqdm
from fused_jagged_hstu import fused_jagged_hstu
import random
import fbgemm_gpu

def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, device="cuda")
    k = torch.randn(sum_N, head*d, device="cuda")
    v = torch.randn(sum_N, head*d, device="cuda")
    rab = torch.randn(B, 1, n, n, device="cuda")
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, rab, attn_mask

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

seq_len = [128, 120, 256, 260, 512, 510, 1024, 1020, 100, 200, 300, 400]
n = max(seq_len)
B = 20
x_offsets = [0]
for i in range(1, B+1):
    x_offsets.append(x_offsets[-1] + random.choice(seq_len)) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

head, d = 2 , 32
sum_N = x_offsets[-1]

print('benchmark config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))
print('input q k v & output shape: ({}, {})'.format(sum_N, head*d))
print('input rab shape: ({}, {}, {}, {})'.format(B, 1, n, n))
print('input attn_mask shape: ({}, {}, {}, {})'.format(1, 1, n, n))

print('===========================================================')

print('warm up')
for _ in tqdm(range(3)):
    q, k, v, rab, attn_mask = get_input(sum_N, head, d, B, n)
    warmup_einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
    warmup_fused_attn = fused_jagged_hstu(q, k, v, rab, attn_mask, head, d, n, x_offsets)
print('warm up done')

print('===========================================================')

print('start benchmark')

einsum_time = []
fused_time = []

test_num = 100

for _ in tqdm(range(test_num)):
    q, k, v, rab, attn_mask = get_input(sum_N, head, d, B, n)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
    end_event.record()
    torch.cuda.synchronize()
    einsum_time.append(start_event.elapsed_time(end_event))
    #print("einsum Time: {}ms ".format(start_event.elapsed_time(end_event)))

    start_event.record()
    fused_attn = fused_jagged_hstu(q, k, v, rab, attn_mask, head, d, n, x_offsets).permute(1, 0, 2).contiguous().view(sum_N, head*d)
    end_event.record()
    torch.cuda.synchronize()
    fused_time.append(start_event.elapsed_time(end_event))
#print("Warp Triton Time: {}ms ".format(start_event.elapsed_time(end_event)))
#print("diff: ", torch.mean(torch.abs(einsum_attn - fused_attn)))

acc_ratio = [einsum_time[i] / fused_time[i] for i in range(test_num)]
print('average einsum time: {}ms'.format(sum(einsum_time) / test_num))
print('average fused time: {}ms'.format(sum(fused_time) / test_num))
print('average acceleration ratio: {}'.format(sum(acc_ratio) / test_num))
print('max acceleration ratio: {}'.format(max(acc_ratio)))
print('min acceleration ratio: {}'.format(min(acc_ratio)))