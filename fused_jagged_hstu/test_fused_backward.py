import torch
import torch.nn.functional as F
from tqdm import tqdm
from fused_jagged_hstu import fused_jagged_hstu
import random
import fbgemm_gpu
import torch.profiler
from fused_hstu_op import FusedHSTUOp
from fused_jagged_hstu_backward import fused_jagged_hstu_backward
from fused_backward_simpler import fused_backward_simpler

def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")

    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    rab = torch.randn(B, 1, n, n, device="cuda")
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, q1, k1, v1, rab, attn_mask

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
n = 0
B = 128
x_offsets = [0]
max_seq = 200
min_seq = 100
for i in range(1, B+1):
    rand_seq_len = random.randint(min_seq, max_seq)
    n = max(n, rand_seq_len)
    x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

head, d = 8 , 32
sum_N = int(x_offsets[-1])

print('benchmark config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))
print('input q k v & output shape: ({}, {})'.format(sum_N, head*d))
print('input rab shape: ({}, {}, {}, {})'.format(B, 1, n, n))
print('input attn_mask shape: ({}, {}, {}, {})'.format(1, 1, n, n))


q, k, v, q1, k1, v1, rab, attn_mask = get_input(sum_N, head, d, B, n)
einsum_attn = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
# fused_attn = FusedHSTUOp.apply(q1, k1, v1, rab, attn_mask, head, d, n, x_offsets)

# print('forward diff:', (torch.abs(einsum_attn - fused_attn)).mean(), torch.max(torch.abs(einsum_attn - fused_attn)))

# loss = einsum_attn.sum()
# loss.backward()

# loss1 = fused_attn.sum()
# loss1.backward()

q, k, v, q1, k1, v1, rab, attn_mask = get_input(sum_N, head, d, B, n)

grad_output = torch.ones_like(einsum_attn)
dq, dk, dv = fused_jagged_hstu_backward(
    grad_output, q, k, v, rab, attn_mask, head, d, n, x_offsets
)

dq1, dk1, dv1 = fused_backward_simpler(
    grad_output, q1, k1, v1, rab, attn_mask, head, d, n, x_offsets
)

print('backward diff q1:', (dq - dq1).abs().mean(), (dq - dq1).abs().max())
print('backward diff k1:', (dk - dk1).abs().mean(), (dk - dk1).abs().max())
print('backward diff v1:', (dv - dv1).abs().mean(), (dv - dv1).abs().max())

for _ in range(10):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    grad_output = torch.randn_like(einsum_attn)
    q, k, v, q1, k1, v1, rab, attn_mask = get_input(sum_N, head, d, B, n)

    start_event.record()
    dq, dk, dv = fused_jagged_hstu_backward(
        grad_output, q, k, v, rab, attn_mask, head, d, n, x_offsets
    )
    end_event.record()
    torch.cuda.synchronize()
    print('fused_jagged_hstu_backward time:', start_event.elapsed_time(end_event))

    start_event.record()
    dq1, dk1, dv1 = fused_backward_simpler(
        grad_output, q1, k1, v1, rab, attn_mask, head, d, n, x_offsets
    )
    end_event.record()
    torch.cuda.synchronize()
    print('fused_backward_simpler time:', start_event.elapsed_time(end_event))

    print('backward diff q:', (dq - dq1).abs().mean(), (dq - dq1).abs().max())
    print('backward diff k:', (dk - dk1).abs().mean(), (dk - dk1).abs().max())
    print('backward diff v:', (dv - dv1).abs().mean(), (dv - dv1).abs().max())