import torch
import os
import torch.nn.functional as F
from fused_hstu_attn import hstu_fused_attention_v1
from fused_gemm import triton_batched_matmul
from fused_hstu import hstu_fused_attention
from tqdm import tqdm


# 生成四维输入
B, n, num_heads, head_dim = 16, 1024, 8, 32
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
#In practice, we share rabp,t across different attention heads within a layer, hence rabp,t ∈ R1×N×N .
rab = torch.randn(B,1, n, n, device="cuda")

# origin_time = []
# triton_time = []

# for _ in tqdm(range(100)):
#     q = torch.randn(B, n, num_heads, head_dim, device="cuda")
#     k = torch.randn(B, n, num_heads, head_dim, device="cuda")
#     v = torch.randn(B, n, num_heads, head_dim, device="cuda")
#     rab = torch.randn(B, n, n, device="cuda")

#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     start_event.record()
#     # 原始计算
#     qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
#     qk = F.silu(qk)/n
#     attn = torch.einsum("bhnm,bmhd->bnhd", qk, v)

#     end_event.record()
#     torch.cuda.synchronize()
#     origin_time.append(start_event.elapsed_time(end_event))

#     start_event.record()
#     attn1 = hstu_fused_attention_v1(q, k, v, rab,False).permute(0, 2, 1, 3).contiguous()
#     end_event.record()
#     torch.cuda.synchronize()
#     triton_time.append(start_event.elapsed_time(end_event))

# print('avg origin time:', sum(origin_time)/100)
# print('avg triton time:', sum(triton_time)/100)

# one_time_speedup = [origin_time[i]/triton_time[i] for i in range(100)]

# print('avg speedup:', sum(origin_time)/sum(triton_time))
# print('max speedup:', max(one_time_speedup))
# print('min speedup:', min(one_time_speedup))

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)



# 记录开始时间
start_event.record()
# 原始计算
qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
qk = F.silu(qk)/n
qk += rab
attn = torch.einsum("bhnm,bmhd->bnhd", qk, v)

end_event.record()
torch.cuda.synchronize()
# print("config: B={}, n={}, num_heads={}, head_dim={}".format(B, n, num_heads, head_dim))
print("origin einsum Time: {}ms ".format(start_event.elapsed_time(end_event)))
    

attn1 = hstu_fused_attention(q, k, v, rab,True).permute(0, 2, 1, 3).contiguous()
attn2 = hstu_fused_attention_v1(q, k, v, rab,True).permute(0, 2, 1, 3).contiguous()

print(attn[1,23,7,1])
print(attn1[1,23,7,1])
print(attn2[1,23,7,1])

print("diff 1:",torch.sum(torch.abs(attn - attn1))/(B*n*num_heads*head_dim))
print("diff 2:",torch.sum(torch.abs(attn - attn2))/(B*n*num_heads*head_dim))
# attn = triton_batched_matmul(q, k)

# assert torch.allclose(qk, attn, atol=1e-3)
