import torch
import torch.nn.functional as F
import triton

from fused_hstu_v3.fused_backward_v3 import fused_backward_simpler
from fused_hstu_v3.fused_hstu_v3 import fused_jagged_hstu
# 在改善nan的bug后发现，simpler版本的backward更快，因此将backward的代码改为simpler版本
class FusedHSTUOpv3(torch.autograd.Function):



    @staticmethod
    def forward(ctx, q, k, v, rab, attn_mask, head, dim, n, x_offsets):
        sum_N, _ = q.shape
        q = q.view(sum_N, head, dim)  # (head, sum_N, d)
        k = k.view(sum_N, head, dim)  # (head, sum_N, d)
        v = v.view(sum_N, head, dim)  # (head, sum_N, d
        pad_len = triton.next_power_of_2(dim) - dim

        if  pad_len != 0:
            #raise ValueError("dim must be a power of 2")
            q = F.pad(q, (0, pad_len), "constant", 0)
            k = F.pad(k, (0, pad_len), "constant", 0)
            v = F.pad(v, (0, pad_len), "constant", 0)
            dim += pad_len
            #print("pad q,", q.shape)

        ctx.save_for_backward(q, k, v, rab, attn_mask)
        ctx.head = head
        ctx.dim = dim
        ctx.n = n
        ctx.x_offsets = x_offsets
        ctx.pad_len = pad_len

        if pad_len != 0:
            return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets)[:,:,:(-pad_len)].contiguous().view(sum_N, head*(dim-pad_len))

        else:
            return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).view(sum_N, head*dim)

        #return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2)[:,:,:(-pad_len)] .view(sum_N, head*(dim-pad_len))

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, rab, attn_mask = ctx.saved_tensors
        head = ctx.head
        dim = ctx.dim
        n = ctx.n
        x_offsets = ctx.x_offsets
        pad_len = ctx.pad_len

        #("grad_output shape: ", grad_output.shape)
        sum_N, _ = grad_output.shape
        
        grad_output = grad_output.view(sum_N, head, (dim - pad_len)) 
        #print("grad",grad_output[0,0,:])

        if pad_len != 0:
            grad_output = F.pad(grad_output, (0, pad_len), "constant", 0)
        assert(torch.isnan(grad_output).any() == False)
        grad_q, grad_k, grad_v, grad_rab = fused_backward_simpler(
           grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets
        )

        
        if pad_len != 0:
            grad_q = grad_q[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
            grad_k = grad_k[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
            grad_v = grad_v[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
        else:
            grad_q = grad_q.view(sum_N, head*dim)
            grad_k = grad_k.view(sum_N, head*dim)
            grad_v = grad_v.view(sum_N, head*dim)


        return grad_q, grad_k, grad_v, grad_rab, None, None, None, None, None