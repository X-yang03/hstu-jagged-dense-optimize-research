# simpler code, but also slower
import torch
import triton
import triton.language as tl
@triton.jit
def silu(x):
    return x*tl.sigmoid(x) 

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=4),
    ],
    key=["N"],
)


@triton.jit
def fused_backward_kernel(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dRab_ptr, 
    dOut_ptr,
    attn_mask_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,
    stride_kn, stride_kh, stride_kd,
    stride_dkn, stride_dkh, stride_dkd,
    stride_qn, stride_qh, stride_qd,
    stride_dqn, stride_dqh, stride_dqd,
    stride_vn, stride_vh, stride_vd,
    stride_dvn, stride_dvh, stride_dvd,
    stride_rab_b, stride_rab_h, stride_rab_n, stride_rab_m,
    stride_drab_b, stride_drab_h, stride_drab_n, stride_drab_m,
    stride_mask_n, stride_mask_m,
    stride_out_n, stride_out_h, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    ):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = (end - start).to(tl.int32)

    for block_kv in range(0, len_sample, BLOCK_SIZE_N):  #load  K_i V_i
        k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        d_k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)
        d_v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)

        mask_kv = (block_kv + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

        k_ptrs = K_ptr + start * stride_kn + block_kv * stride_kn + pid_h * stride_kh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_kn + \
                    tl.arange(0, D)[None, :] * stride_kd

        v_ptrs = V_ptr + start * stride_vn + block_kv * stride_vn + pid_h * stride_vh +\
                tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_vn + \
                tl.arange(0, D)[None, :] * stride_vd

        dk_ptrs = dK_ptr + start * stride_dkn + block_kv * stride_dkn + pid_h * stride_dkh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_dkn + \
                    tl.arange(0, D)[None, :] * stride_dkd

        dv_ptrs = dV_ptr + start * stride_dvn + block_kv * stride_dvn + pid_h * stride_dvh +\
                tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_dvn + \
                tl.arange(0, D)[None, :] * stride_dvd
        
        k = tl.load(k_ptrs, mask=mask_kv, other=0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0)
            
        for block_q in range(block_kv, len_sample, BLOCK_SIZE_N):  # load Q_i, dQ_i, O_i, dO_i, d_attn_i
            
            rab_ptrs = rab_ptr + pid_b * stride_rab_b +\
                    block_q * stride_rab_n + block_kv * stride_rab_m +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_rab_n +\
                    tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_rab_m

            drab_ptrs = dRab_ptr + pid_b * stride_drab_b + pid_h * stride_drab_h +\
                    block_q * stride_drab_n + block_kv * stride_drab_m +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_drab_n +\
                    tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_drab_m

            mask = (block_q + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample
            rab = tl.load(rab_ptrs, mask = mask & mask_kv.T, other=0)

            q_ptrs = Q_ptr + start * stride_qn + block_q * stride_qn + pid_h * stride_qh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                    tl.arange(0, D)[None, :] * stride_qd
            
            q = tl.load(q_ptrs, mask=mask, other=0)
            #q = tl.load(q_ptrs)
            dq_ptrs = dQ_ptr + start * stride_dqn + block_q * stride_dqn + pid_h * stride_dqh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_dqn + \
                    tl.arange(0, D)[None, :] * stride_dqd
            
            do_ptrs = dOut_ptr + start * stride_out_n + block_q * stride_out_n + pid_h * stride_out_h +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                    tl.arange(0, D)[None, :] * stride_out_d
            
            d_q = tl.load(dq_ptrs, mask=mask, other=0)
            d_o = tl.load(do_ptrs, mask=mask, other=0)
            

            #计算qk
            qk = tl.dot(q, k.T, input_precision = "ieee") + rab

            sigmoid_qk = tl.sigmoid(qk)
            qk_normalized = (qk * sigmoid_qk) / N

            d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))

            d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
            # (BLOCK_SIZE_N, D) * (D, BLOCK_SIZE_N) -> (BLOCK_SIZE_N, BLOCK_SIZE_N)

                
            if block_kv == block_q:  #mask的处理方式
                mask_ptrs = attn_mask_ptr + block_q * stride_mask_n + block_kv * stride_mask_m +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_mask_n +\
                            tl.arange(0, BLOCK_SIZE_N)[None,:] * stride_mask_m
                attn_mask = tl.load(mask_ptrs, mask = mask & mask_kv.T, other=0)
                # mask 也要控制读取内存边界！！！！ 否则会illegal memory access 或读出 nan
                                
                # attn_mask = tl.load(mask_ptrs)
                qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
                d_qk = d_qk * attn_mask #掩码梯度

            d_v += tl.dot(qk_normalized.T, d_o, input_precision = "ieee")
            #mask_qk @ d_o  (m, n)@(n, d) -> (m, d)
            d_qk = d_qk * d_silu_qk

            tl.store(drab_ptrs, d_qk, mask= mask & mask_kv.T)

            d_q += tl.dot(d_qk, k, input_precision = "ieee") 
            # (BLOCK_SIZE_N, BLOCK_SIZE_N) * (BLOCK_SIZE_N, D) -> (BLOCK_SIZE_N, D)

            d_k += tl.dot(d_qk.T, q, input_precision = "ieee")

            tl.store(dq_ptrs, d_q, mask=mask)
  
        tl.store(dk_ptrs, d_k, mask=mask_kv)
        tl.store(dv_ptrs, d_v, mask=mask_kv)
        


def fused_backward_simpler(d_attn, q, k, v, rab, attn_mask, head, dim, n, x_offsets):

    B = x_offsets.shape[0] - 1
    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)

    d_rab = torch.zeros((B, head, n, n), dtype=d_attn.dtype, device=d_attn.device)

    grid = (head, B)

    fused_backward_kernel[grid](
        q, k, v, rab,
        d_q, d_k, d_v, d_rab,
        d_attn ,
        attn_mask,
        x_offsets,
        B, head, n, dim,
        k.stride(0), k.stride(1), k.stride(2),
        d_k.stride(0), d_k.stride(1), d_k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        d_q.stride(0), d_q.stride(1), d_q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d_v.stride(0), d_v.stride(1), d_v.stride(2),
        rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
        d_rab.stride(0), d_rab.stride(1), d_rab.stride(2), d_rab.stride(3),
        attn_mask.stride(2), attn_mask.stride(3),
        d_attn.stride(0), d_attn.stride(1), d_attn.stride(2),
    )
    return d_q, d_k, d_v, d_rab



