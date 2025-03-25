import torch
import torch.nn.functional as F
import triton
from fused_jagged_hstu.fused_jagged_hstu import fused_jagged_hstu
from fused_jagged_hstu.fused_jagged_hstu_backward import fused_jagged_hstu_backward

class FusedHSTUOp(torch.autograd.Function):



    @staticmethod
    def forward(ctx, q, k, v, rab, attn_mask, head, dim, n, x_offsets):
        sum_N, _ = q.shape
        q = q.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
        k = k.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d)
        v = v.view(sum_N, head, dim).permute(1, 0, 2).contiguous() # (head, sum_N, d
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
            return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2)[:,:,:(-pad_len)].contiguous().view(sum_N, head*(dim-pad_len))

        else:
            return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2).contiguous().view(sum_N, head*dim)

        #return fused_jagged_hstu(q, k, v, rab, attn_mask, head, dim, n, x_offsets).permute(1, 0, 2)[:,:,:(-pad_len)].contiguous().view(sum_N, head*(dim-pad_len))

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
        
        grad_output = grad_output.view(sum_N, head, (dim - pad_len)).permute(1, 0, 2).contiguous()
        #print("grad",grad_output[0,0,:])

        if pad_len != 0:
            grad_output = F.pad(grad_output, (0, pad_len), "constant", 0)

        grad_q, grad_k, grad_v = fused_jagged_hstu_backward(
           grad_output, q, k, v, rab, attn_mask, head, dim, n, x_offsets
        )

        # print("test q nan:", torch.isnan(grad_q).any())
        # print("test k nan:", torch.isnan(grad_k).any())
        # print("test v nan:", torch.isnan(grad_v).any())
        # if(torch.isnan(grad_q).any()):
        #     print(grad_q)
        # if(torch.isnan(grad_k).any()):
        #     #print(grad_k)
        #     nan_indices = torch.isnan(grad_k).nonzero()
        #     print(nan_indices)
        #     print(x_offsets)
        #     print("dk340:", grad_k[0,340,:])
        #     print("dk341:",grad_k[0,341,:])
        #     torch.save(q, "q.pt")
        #     torch.save(k, "k.pt")
        #     torch.save(v, "v.pt")
        #     torch.save(rab, "rab.pt")
        #     torch.save(attn_mask, "attn_mask.pt")
        #     torch.save(x_offsets, "x_offsets.pt")
        #     torch.save(grad_output, "grad_output.pt")
        #     print("config: head: {}, dim: {}, n: {}".format(head, dim, n))
        if pad_len != 0:
            grad_q = grad_q.permute(1, 0, 2)[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
            grad_k = grad_k.permute(1, 0, 2)[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
            grad_v = grad_v.permute(1, 0, 2)[:, :, :(-pad_len)].contiguous().view(sum_N, head*(dim - pad_len))
        else:
            grad_q = grad_q.permute(1, 0, 2).contiguous().view(sum_N, head*dim)
            grad_k = grad_k.permute(1, 0, 2).contiguous().view(sum_N, head*dim)
            grad_v = grad_v.permute(1, 0, 2).contiguous().view(sum_N, head*dim)

        
        return grad_q, grad_k, grad_v, None, None, None, None, None, None