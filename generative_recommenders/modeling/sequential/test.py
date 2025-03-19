import torch
import torch.nn.functional as F
from fused_hstu_attn import hstu_fused_attention_v1
from fused_gemm import triton_batched_matmul
from fused_hstu import hstu_fused_attention
from tqdm import tqdm

B, n, num_heads, head_dim = 16, 1024, 8, 32
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
rab = torch.randn(B, 1, n, n, device="cuda")

# 预热阶段：执行多次以消除冷启动开销
print("Warmup Triton")
for _ in tqdm(range(10)):
    warmup = hstu_fused_attention_v1(q, k, v, rab, True)
    temp = torch.einsum("bnhd,bmhd->bhnm", q, k)
    temp += rab
    temp = F.silu(temp)/n
    temp = torch.einsum("bhnm,bmhd->bnhd", temp, v)
print("Warmup Triton Done")

print('Start Benchmark')


q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
rab = torch.randn(B, 1, n, n, device="cuda")

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# 原始计算
qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
qk += rab
qk = F.silu(qk)/n
attn = torch.einsum("bhnm,bmhd->bnhd", qk, v)

end_event.record()
torch.cuda.synchronize()
print("einsum Time: {}ms ".format(start_event.elapsed_time(end_event)))


fused_attn = hstu_fused_attention_v1(q, k, v, rab, True)

print("diff: ", torch.mean(torch.abs(attn - fused_attn.permute(0,2,1,3))))
#print("Triton Time: {}ms ".format(start_event.elapsed_time(end_event)))
