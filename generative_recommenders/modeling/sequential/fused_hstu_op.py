import torch
import torch.nn.functional as F
from fused_jagged_hstu import fused_jagged_hstu
from fused_jagged_hstu_backward import fused_jagged_hstu_backward

class FusedHSTUOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rab, attn_mask, head, dim, n, x_offsets):
        ctx.save_for_backward(q, k, v, rab, attn_mask)
        ctx.head = head
        ctx.dim = dim
        ctx.n = n
        ctx.x_offsets = x_offsets

        return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2).contiguous().view(q.shape)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, rab, attn_mask = ctx.saved_tensors
        head = ctx.head
        dim = ctx.dim
        n = ctx.n
        x_offsets = ctx.x_offsets

        grad_q, grad_k, grad_v = fused_jagged_hstu_backward(
           grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets
        )
        grad_q = grad_q.permute(1, 0, 2).contiguous().view(q.shape)
        grad_k = grad_k.permute(1, 0, 2).contiguous().view(k.shape)
        grad_v = grad_v.permute(1, 0, 2).contiguous().view(v.shape)

        return grad_q, grad_k, grad_v, None, None, None, None, None, None