import torch
import os
import torch.nn.functional as F
from ..generative_recommenders.modeling.sequential.fused_hstu_attn import hstu_fused_attention_4d_wrapper

# 生成四维输入
B, n, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
rab = torch.randn(B, n, n, device="cuda")

# 原始计算
qk = torch.einsum("bnhd,bmhd->bhnm", q, k) + rab.unsqueeze(1)
qk = F.silu(qk) / n
attn_original = torch.einsum("bhnm,bmhd->bnhd", qk, v)

# Triton 融合计算
attn_triton = hstu_fused_attention_4d_wrapper(q, k, v, rab)

# 验证误差
assert torch.allclose(attn_original, attn_triton, atol=1e-3)