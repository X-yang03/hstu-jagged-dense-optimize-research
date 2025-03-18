import torch
import torch.nn.functional as F
from tqdm import tqdm
from fused_jagged_hstu import fused_jagged_hstu
import random
import fbgemm_gpu

interval = [250,510,1020]
n = max(interval)
B = 20
x_offsets = [0]
for i in range(1, B+1):
    x_offsets.append(x_offsets[-1] + random.choice(interval))
x_offsets = torch.tensor(x_offsets, device="cuda")

head, d = 2, 32
sum_N = x_offsets[-1]
q = torch.randn(sum_N, head*d, device="cuda")
k = torch.randn(sum_N, head*d, device="cuda")
v = torch.randn(sum_N, head*d, device="cuda")
rab = torch.randn(B, 1, n, n, device="cuda")
attn_mask = torch.randn(1, n, n, device="cuda")

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
# 记录开始时间
start_event.record()

# 原本hstu计算
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
qk_attn = F.silu(qk_attn) / n #SiLU之后局部归一化
attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, head, d  #将v转换为padded形式
            ),
        )
# 记录结束时间
end_event.record()
torch.cuda.synchronize()
print("einsum Time: {}ms ".format(start_event.elapsed_time(end_event)))



print("attn_output shape: ", attn_output.shape)
print(attn_output[0, 0, 0, 10])
output = fused_jagged_hstu(q, k, v, head, d, n, x_offsets).permute(0, 2, 1, 3)
output1 = fused_jagged_hstu(q, k, v, head, d, n, x_offsets).permute(0, 2, 1, 3)
print(output[0, 0, 0, 10])
print("output shape: ", output.shape)
print("avg diff: ", torch.mean(torch.abs(attn_output - output)))
print("max diff: ", torch.max(torch.abs(attn_output - output)))