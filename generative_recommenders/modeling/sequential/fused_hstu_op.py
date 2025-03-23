import torch
import torch.nn.functional as F
import triton
from fused_jagged_hstu import fused_jagged_hstu
from fused_jagged_hstu_backward import fused_jagged_hstu_backward

class FusedHSTUOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rab, attn_mask, head, dim, n, x_offsets):
        sum_N, _ = q.shape
        q = q.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
        k = k.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
        v = v.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d
        # pad_dim = triton.next_power_of_2(dim)
        # if  pad_dim != dim:
        #     #raise ValueError("dim must be a power of 2")
        #     q = F.pad(q, (0, pad_dim - dim), "constant", 0)
        #     k = F.pad(k, (0, pad_dim - dim), "constant", 0)
        #     v = F.pad(v, (0, pad_dim - dim), "constant", 0)
        #     print("pad q,", q.shape)

        ctx.save_for_backward(q, k, v, rab, attn_mask)
        ctx.head = head
        ctx.dim = dim
        ctx.n = n
        ctx.x_offsets = x_offsets

        return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2).contiguous().view(sum_N, head*dim)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, rab, attn_mask = ctx.saved_tensors
        head = ctx.head
        dim = ctx.dim
        n = ctx.n
        x_offsets = ctx.x_offsets

        print("grad_output shape: ", grad_output.shape)
        sum_N, _ = grad_output.shape
        
        grad_output = grad_output.view(sum_N, head, dim).permute(1, 0, 2).contiguous()

        grad_q, grad_k, grad_v = fused_jagged_hstu_backward(
           grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets
        )
        grad_q = grad_q.permute(1, 0, 2).contiguous().view(sum_N, head*dim)
        grad_k = grad_k.permute(1, 0, 2).contiguous().view(sum_N, head*dim)
        grad_v = grad_v.permute(1, 0, 2).contiguous().view(sum_N, head*dim)

        return grad_q, grad_k, grad_v, None, None, None, None, None, None