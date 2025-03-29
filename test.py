from fused_jagged_hstu.fused_jagged_hstu_backward import fused_jagged_hstu_backward
import torch
import torch.nn.functional as F
import triton

q = torch.load("q.pt") 
k = torch.load("k.pt")
v = torch.load("v.pt")
print("q",q[0,0,:])

rab = torch.load("rab.pt")
attn_mask = torch.load("attn_mask.pt")
grad_output = torch.load("grad_output.pt")


head = 8
dim = 32
n = 61
x_offsets = torch.load("x_offsets.pt")

for _ in range(100):
    dq, dk, dv = fused_jagged_hstu_backward(grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets)
    # print("dk340", dk[0,340,:])
    # print("dk",dk[0,341,:])
    print(torch.isnan(dq).any())
    print(torch.isnan(dk).any())
    print(torch.isnan(dv).any())

# dq, dk, dv = fused_jagged_hstu_backward(grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets)
# print("dk340", dk[0,340,:])
# print("dk",dk[0,341,:])
# print(torch.isnan(dq).any())
# print(torch.isnan(dk).any())
# print(torch.isnan(dv).any())