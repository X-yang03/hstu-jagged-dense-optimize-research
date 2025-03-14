import torch
import os
import torch.nn.functional as F
from fused_gemm import triton_batched_matmul
from fused_gemm_v2 import triton_batched_matmul_v2
from fused_gemm_v3 import triton_batched_matmul_v3
from fused_gemm_v4 import triton_batched_matmul_v4
from fused_gemm_v5 import triton_batched_matmul_v5

# 生成四维输入
B, n, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")

# 原始计算
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 记录开始时间
start_event.record()
qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
qk = F.silu(qk)/n
# 记录结束时间
end_event.record()
torch.cuda.synchronize()
print("Time: ", start_event.elapsed_time(end_event))

attn = triton_batched_matmul(q, k).view(B,num_heads,n,n).contiguous()
attn1 = triton_batched_matmul_v2(q, k).view(B,num_heads,n,n).contiguous()
attn2 = triton_batched_matmul_v3(q, k).view(B,num_heads,n,n).contiguous()
attn3 = triton_batched_matmul_v4(q, k)
attn4 = triton_batched_matmul_v5(q, k)
# print(qk.size())
# print(attn.size())
print(qk[1,7,123,1])
#print(attn[1,7,123,1])
print(attn1[1,7,123,1])
print(attn2[1,7,123,1])
print(attn3[1,7,123,1])
print(attn4[1,7,123,1])
# print(attn1[1,7,123,1])
print(torch.sum(torch.abs(qk - attn1))/(B*n*num_heads*n))

# assert torch.allclose(qk, attn, atol=1e-3)

