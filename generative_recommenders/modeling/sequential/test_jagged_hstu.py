import torch
import torch.nn.functional as F
from tqdm import tqdm
from fused_jagged_hstu import fused_jagged_hstu
import random
import fbgemm_gpu

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

interval = [256, 510, 1020]
n = max(interval)
B = 20
x_offsets = [0]
for i in range(1, B+1):
    x_offsets.append(x_offsets[-1] + random.choice(interval))
x_offsets = torch.tensor(x_offsets, device="cuda")

head, d = 8 , 32
sum_N = x_offsets[-1]

q, k, v, rab, attn_mask = get_input(sum_N, head, d, B, n)

print('benchmark config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
# 记录开始时间
start_event.record()

attn_output = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)

# 记录结束时间
end_event.record()
torch.cuda.synchronize()
print("einsum Time round 1: {}ms ".format(start_event.elapsed_time(end_event)))

start_event.record()
attn_output = origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
end_event.record()
torch.cuda.synchronize()
print("einsum Time round 2: {}ms ".format(start_event.elapsed_time(end_event)))


print("attn_output shape: ", attn_output.shape)

start_event.record()
output = fused_jagged_hstu(q, k, v, rab, attn_mask, head, d, n, x_offsets).permute(1, 0, 2).contiguous().view(sum_N, head*d)
end_event.record()
torch.cuda.synchronize()
print("Triton Time round1: {}ms ".format(start_event.elapsed_time(end_event)))

start_event.record()
output = fused_jagged_hstu(q, k, v, rab, attn_mask, head, d, n, x_offsets).permute(1, 0, 2).contiguous().view(sum_N, head*d)
end_event.record()
torch.cuda.synchronize()
print("Triton Time round 2: {}ms ".format(start_event.elapsed_time(end_event)))
print("triton output shape: ", output.shape)

print("avg diff: ", torch.mean(torch.abs(attn_output - output)))
print("max diff: ", torch.max(torch.abs(attn_output - output)))

