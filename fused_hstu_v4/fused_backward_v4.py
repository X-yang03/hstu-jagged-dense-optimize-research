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
def fused_backward_q_kernel(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dRab_ptr, 
    dOut_ptr,
    attn_mask_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,
    stride_kn, stride_kh, stride_kd,
    stride_qn, stride_qh, stride_qd,
    stride_dqn, stride_dqh, stride_dqd,
    stride_vn, stride_vh, stride_vd,
    stride_rab_b, stride_rab_h, stride_rab_n, stride_rab_m,
    stride_drab_b, stride_drab_h, stride_drab_n, stride_drab_m,
    stride_mask_n, stride_mask_m,
    stride_out_n, stride_out_h, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    ):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_block = tl.program_id(2)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = (end - start).to(tl.int32)

    block_q = (pid_block * BLOCK_SIZE_N).to(tl.int32)

    if block_q + BLOCK_SIZE_N <= len_sample:  # q块在范围内
        # q 是 （N, H, D）形状
        q_blk_ptrs = tl.make_block_ptr(
            base = Q_ptr + pid_h * stride_qh + start * stride_qn,
            shape = ((len_sample * H), D),
            strides = (stride_qn, stride_qd),  #实际上 stride_qn = stride_qh * H
            offsets = (block_q , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        q = tl.load(q_blk_ptrs)

        do_blk_ptrs = tl.make_block_ptr(
            base = dOut_ptr + pid_h * stride_out_h + start * stride_out_n,
            shape = ((len_sample * H), D),
            strides = (stride_out_n, stride_out_d),
            offsets = (block_q , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        d_o = tl.load(do_blk_ptrs)
        
        d_q = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)  #累加d_q
        
        for block_kv in range(0, block_q + BLOCK_SIZE_N, BLOCK_SIZE_N):  #load  K_j V_j
            #当q在范围内时，由于kv<=q，所以kv也在范围内
            k_blk_ptrs = tl.make_block_ptr(
            base = K_ptr + pid_h * stride_kh + start * stride_kn,
            shape = ((len_sample * H), D),
            strides = (stride_kn, stride_kd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
            v_blk_ptrs = tl.make_block_ptr(
            base = V_ptr + pid_h * stride_vh + start * stride_vn,
            shape = ((len_sample * H), D),
            strides = (stride_vn, stride_vd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )

            k = tl.load(k_blk_ptrs)
            v = tl.load(v_blk_ptrs)

            rab_blk_ptrs = tl.make_block_ptr(  # rab shape : (B,1,N,N)
                base = rab_ptr + pid_b * stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = (block_q , block_kv ),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )
            rab = tl.load(rab_blk_ptrs)

            drab_blk_ptrs = tl.make_block_ptr(  
                base = dRab_ptr + pid_b * stride_drab_b + pid_h * stride_drab_h,
                shape = (N, N),
                strides = (stride_drab_n, stride_drab_m),
                offsets = (block_q , block_kv ),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
            )

            qk = tl.dot(q, k.T, input_precision = "ieee") + rab

            sigmoid_qk = tl.sigmoid(qk)
            qk_normalized = (qk * sigmoid_qk) / N

            d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))

            d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N

            if block_kv == block_q:  #mask的处理方式,由于kv和q都在范围内，mask可以用blk_ptr读取
                # 因为mask是下三角的1矩阵，当block_kv < block_q时，不用做任何处理
                # 当block_kv == block_q时，需要将qk与mask相乘
                    mask_ptrs = tl.make_block_ptr(
                        base = attn_mask_ptr,
                        shape = (N, N),
                        strides = (stride_mask_n, stride_mask_m),
                        offsets = (block_q , block_kv),
                        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                        order = (0, 1)
                    )
                    attn_mask = tl.load(mask_ptrs)
                    qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
                    d_qk = d_qk * attn_mask #掩码梯度
            
            d_qk = d_qk * d_silu_qk
            tl.store(drab_blk_ptrs, d_qk)
            d_q += tl.dot(d_qk, k, input_precision = "ieee") 
            
        dq_blk_ptrs = tl.make_block_ptr(
            base = dQ_ptr + pid_h * stride_dqh + start * stride_dqn,
            shape = ((len_sample * H), D),
            strides = (stride_dqn, stride_dqd),
            offsets = (block_q , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        tl.store(dq_blk_ptrs, d_q)

    ############## 处理末尾不规则长度的情况 ################
    else: 
        if block_q < len_sample: # 末尾是不规则长度
            mask = (block_q + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

            q_ptrs = Q_ptr + start * stride_qn + block_q * stride_qn + pid_h * stride_qh +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                            tl.arange(0, D)[None, :] * stride_qd
                    
            q = tl.load(q_ptrs, mask=mask, other=0)

            dq_ptrs = dQ_ptr + start * stride_dqn + block_q * stride_dqn + pid_h * stride_dqh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_dqn + \
                    tl.arange(0, D)[None, :] * stride_dqd
            
            do_ptrs = dOut_ptr + start * stride_out_n + block_q * stride_out_n + pid_h * stride_out_h +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                    tl.arange(0, D)[None, :] * stride_out_d
            
            d_q = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)  #累加d_q
            d_o = tl.load(do_ptrs, mask=mask, other=0)

            for block_kv in range(0, block_q + BLOCK_SIZE_N, BLOCK_SIZE_N):  #load  K_i V_i
                rab_ptrs = rab_ptr + pid_b * stride_rab_b +\
                            block_q * stride_rab_n + block_kv * stride_rab_m +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_rab_n +\
                            tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_rab_m
                
                drab_ptrs = dRab_ptr + pid_b * stride_drab_b + pid_h * stride_drab_h +\
                        block_q * stride_drab_n + block_kv * stride_drab_m +\
                        tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_drab_n +\
                        tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_drab_m
                
                if block_kv + BLOCK_SIZE_N <= len_sample: #kv块在范围内，此时kv小于q，不用乘以mask
                    k_blk_ptrs = tl.make_block_ptr(
                    base = K_ptr + pid_h * stride_kh + start * stride_kn,
                    shape = ((len_sample * H), D),
                    strides = (stride_kn, stride_kd),
                    offsets = (block_kv , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                    v_blk_ptrs = tl.make_block_ptr(
                    base = V_ptr + pid_h * stride_vh + start * stride_vn,
                    shape = ((len_sample * H), D),
                    strides = (stride_vn, stride_vd),
                    offsets = (block_kv , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )

                    k = tl.load(k_blk_ptrs)
                    v = tl.load(v_blk_ptrs)

                    rab = tl.load(rab_ptrs, mask = mask, other=0)

                    qk = tl.dot(q, k.T, input_precision = "ieee") + rab

                    sigmoid_qk = tl.sigmoid(qk)
                    qk_normalized = (qk * sigmoid_qk) / N

                    d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))
                    d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
                    d_qk = d_qk * d_silu_qk
                    tl.store(drab_ptrs, d_qk, mask= mask)
                    d_q += tl.dot(d_qk, k, input_precision = "ieee") 

                else: #此时q和kv都在末尾，q == kv

                    mask_kv = (block_kv + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

                    k_ptrs = K_ptr + start * stride_kn + block_kv * stride_kn + pid_h * stride_kh +\
                                tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_kn + \
                                tl.arange(0, D)[None, :] * stride_kd

                    v_ptrs = V_ptr + start * stride_vn + block_kv * stride_vn + pid_h * stride_vh +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_vn + \
                            tl.arange(0, D)[None, :] * stride_vd
                    
                    k = tl.load(k_ptrs, mask=mask_kv, other=0)
                    v = tl.load(v_ptrs, mask=mask_kv, other=0)

                    rab = tl.load(rab_ptrs, mask = mask & mask_kv.T, other=0)
                    
                    qk = tl.dot(q, k.T, input_precision = "ieee") + rab

                    sigmoid_qk = tl.sigmoid(qk)
                    qk_normalized = (qk * sigmoid_qk) / N

                    d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))

                    d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
                    # (BLOCK_SIZE_N, D) * (D, BLOCK_SIZE_N) -> (BLOCK_SIZE_N, BLOCK_SIZE_N)
                        
                    mask_ptrs = attn_mask_ptr + block_q * stride_mask_n + block_kv * stride_mask_m +\
                                tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_mask_n +\
                                tl.arange(0, BLOCK_SIZE_N)[None,:] * stride_mask_m
                    attn_mask = tl.load(mask_ptrs, mask = mask & mask_kv.T, other=0)
                    # mask 也要控制读取内存边界！！！！ 否则会illegal memory access 或读出 nan
                                    
                    # attn_mask = tl.load(mask_ptrs)
                    qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
                    d_qk = d_qk * attn_mask #掩码梯度

                    d_qk = d_qk * d_silu_qk

                    tl.store(drab_ptrs, d_qk, mask= mask & mask_kv.T)

                    d_q += tl.dot(d_qk, k, input_precision = "ieee") 

            tl.store(dq_ptrs, d_q, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4,num_stages=4),
    ],
    key=["N"],
)

@triton.jit
def fused_backward_kv_kernel(
    Q_ptr, K_ptr, V_ptr, rab_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dRab_ptr, 
    dOut_ptr,
    attn_mask_ptr,
    x_offsets_ptr,
    B, H, N, D :tl.constexpr,
    stride_kn, stride_kh, stride_kd,
    stride_dkn, stride_dkh, stride_dkd,
    stride_qn, stride_qh, stride_qd,
    stride_vn, stride_vh, stride_vd,
    stride_dvn, stride_dvh, stride_dvd,
    stride_rab_b, stride_rab_h, stride_rab_n, stride_rab_m,
    stride_mask_n, stride_mask_m,
    stride_out_n, stride_out_h, stride_out_d,
    BLOCK_SIZE_N: tl.constexpr
    ):

    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_block = tl.program_id(2)

    start = tl.load(x_offsets_ptr + pid_b)
    end = tl.load(x_offsets_ptr + pid_b + 1)
    len_sample = (end - start).to(tl.int32)

    block_kv = pid_block * BLOCK_SIZE_N

    if block_kv + BLOCK_SIZE_N <= len_sample:  # kv块在范围内
        k_blk_ptrs = tl.make_block_ptr(
            base = K_ptr + pid_h * stride_kh + start * stride_kn,
            shape = ((len_sample * H), D),
            strides = (stride_kn, stride_kd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        v_blk_ptrs = tl.make_block_ptr(
        base = V_ptr + pid_h * stride_vh + start * stride_vn,
        shape = ((len_sample * H), D),
        strides = (stride_vn, stride_vd),
        offsets = (block_kv , 0),
        block_shape = (BLOCK_SIZE_N, D),
        order = (0, 1)
    )
        k = tl.load(k_blk_ptrs)
        v = tl.load(v_blk_ptrs)

        d_k = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)  #累加dk dv
        d_v = tl.zeros((BLOCK_SIZE_N, D), dtype=tl.float32)

        for block_q in range(block_kv, len_sample, BLOCK_SIZE_N):
            if block_q + BLOCK_SIZE_N <= len_sample: # q块在范围内
                q_blk_ptrs = tl.make_block_ptr(
                    base = Q_ptr + pid_h * stride_qh + start * stride_qn,
                    shape = ((len_sample * H), D),
                    strides = (stride_qn, stride_qd),
                    offsets = (block_q , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                q = tl.load(q_blk_ptrs)

                do_blk_ptrs = tl.make_block_ptr(
                    base = dOut_ptr + pid_h * stride_out_h + start * stride_out_n,
                    shape = ((len_sample * H), D),
                    strides = (stride_out_n, stride_out_d),
                    offsets = (block_q , 0),
                    block_shape = (BLOCK_SIZE_N, D),
                    order = (0, 1)
                )
                d_o = tl.load(do_blk_ptrs)

                rab_blk_ptrs = tl.make_block_ptr(  # rab shape : (B,1,N,N)
                base = rab_ptr + pid_b * stride_rab_b,
                shape = (N, N),
                strides = (stride_rab_n, stride_rab_m),
                offsets = (block_q , block_kv ),
                block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                order = (0, 1)
                )
                rab = tl.load(rab_blk_ptrs)

                qk = tl.dot(q, k.T, input_precision = "ieee") + rab
                sigmoid_qk = tl.sigmoid(qk)
                qk_normalized = (qk * sigmoid_qk) / N
                d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))
                d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
                if block_kv == block_q:  #由于kv和q都在范围内，mask可以用blk_ptr读取
                    mask_ptrs = tl.make_block_ptr(
                        base = attn_mask_ptr,
                        shape = (N, N),
                        strides = (stride_mask_n, stride_mask_m),
                        offsets = (block_q , block_kv),
                        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_N),
                        order = (0, 1)
                    )
                    attn_mask = tl.load(mask_ptrs)
                    qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
                    d_qk = d_qk * attn_mask #掩码梯度
                
                d_v += tl.dot(qk_normalized.T, d_o, input_precision = "ieee")
                #mask_qk @ d_o  (m, n)@(n, d) -> (m, d)
                d_qk = d_qk * d_silu_qk

                d_k += tl.dot(d_qk.T, q, input_precision = "ieee")

            else: #此时q在末尾，q > kv，无需mask
                mask = (block_q + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample

                q_ptrs = Q_ptr + start * stride_qn + block_q * stride_qn + pid_h * stride_qh +\
                            tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                            tl.arange(0, D)[None, :] * stride_qd
                    
                q = tl.load(q_ptrs, mask=mask, other=0)

                do_ptrs = dOut_ptr + start * stride_out_n + block_q * stride_out_n + pid_h * stride_out_h +\
                        tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                        tl.arange(0, D)[None, :] * stride_out_d
                
                d_o = tl.load(do_ptrs, mask=mask, other=0)

                rab_ptrs = rab_ptr + pid_b * stride_rab_b +\
                    block_q * stride_rab_n + block_kv * stride_rab_m +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_rab_n +\
                    tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_rab_m
                
                rab = tl.load(rab_ptrs, mask = mask, other=0)

                qk = tl.dot(q, k.T, input_precision = "ieee") + rab
                sigmoid_qk = tl.sigmoid(qk)
                qk_normalized = (qk * sigmoid_qk) / N
                d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))
                d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N

                d_v += tl.dot(qk_normalized.T, d_o, input_precision = "ieee")
                #mask_qk @ d_o  (m, n)@(n, d) -> (m, d)
                d_qk = d_qk * d_silu_qk

                d_k += tl.dot(d_qk.T, q, input_precision = "ieee")

        dk_blk_ptrs = tl.make_block_ptr(
            base = dK_ptr + pid_h * stride_dkh + start * stride_dkn,
            shape = ((len_sample * H), D),
            strides = (stride_dkn, stride_dkd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        dv_blk_ptrs = tl.make_block_ptr(
            base = dV_ptr + pid_h * stride_dvh + start * stride_dvn,
            shape = ((len_sample * H), D),
            strides = (stride_dvn, stride_dvd),
            offsets = (block_kv , 0),
            block_shape = (BLOCK_SIZE_N, D),
            order = (0, 1)
        )
        tl.store(dk_blk_ptrs, d_k)
        tl.store(dv_blk_ptrs, d_v)

    else: # kv块在范围外
        if block_kv < len_sample: # 末尾是不规则长度
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
                

            #由于block_q >= block_kv, 所以block_q此时也是最末尾块
            block_q = block_kv
            
            rab_ptrs = rab_ptr + pid_b * stride_rab_b +\
                    block_q * stride_rab_n + block_kv * stride_rab_m +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_rab_n +\
                    tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_rab_m

            mask = (block_q + tl.arange(0, BLOCK_SIZE_N))[:,None] < len_sample
            rab = tl.load(rab_ptrs, mask = mask & mask_kv.T, other=0)

            q_ptrs = Q_ptr + start * stride_qn + block_q * stride_qn + pid_h * stride_qh +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_qn + \
                    tl.arange(0, D)[None, :] * stride_qd
            
            q = tl.load(q_ptrs, mask=mask, other=0)
            #q = tl.load(q_ptrs)
            
            do_ptrs = dOut_ptr + start * stride_out_n + block_q * stride_out_n + pid_h * stride_out_h +\
                    tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_out_n + \
                    tl.arange(0, D)[None, :] * stride_out_d
            
            d_o = tl.load(do_ptrs, mask=mask, other=0)
            
            #计算qk
            qk = tl.dot(q, k.T, input_precision = "ieee") + rab

            sigmoid_qk = tl.sigmoid(qk)
            qk_normalized = (qk * sigmoid_qk) / N

            d_silu_qk = sigmoid_qk * (1 + qk * (1 - sigmoid_qk))

            d_qk = tl.dot(d_o, v.T, input_precision = "ieee") / N
            # (BLOCK_SIZE_N, D) * (D, BLOCK_SIZE_N) -> (BLOCK_SIZE_N, BLOCK_SIZE_N)

            mask_ptrs = attn_mask_ptr + block_q * stride_mask_n + block_kv * stride_mask_m +\
                        tl.arange(0, BLOCK_SIZE_N)[:,None] * stride_mask_n +\
                        tl.arange(0, BLOCK_SIZE_N)[None,:] * stride_mask_m
            attn_mask = tl.load(mask_ptrs, mask = mask & mask_kv.T, other=0)
            # mask 也要控制读取内存边界！！！！ 否则会illegal memory access 或读出 nan
                            
            # attn_mask = tl.load(mask_ptrs)
            qk_normalized = qk_normalized * attn_mask  #(BLOCK_SIZE_N, BLOCK_SIZE_N)
            d_qk = d_qk * attn_mask #掩码梯度

            d_v = tl.dot(qk_normalized.T, d_o, input_precision = "ieee")
            #mask_qk @ d_o  (m, n)@(n, d) -> (m, d)
            d_qk = d_qk * d_silu_qk

            d_k = tl.dot(d_qk.T, q, input_precision = "ieee")

            tl.store(dk_ptrs, d_k, mask=mask_kv)
            tl.store(dv_ptrs, d_v, mask=mask_kv)
    
def fused_backward_simpler(d_attn, q, k, v, rab, attn_mask, head, dim, n, x_offsets):

    B = x_offsets.shape[0] - 1
    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)

    d_rab = torch.zeros((B, head, n, n), dtype=d_attn.dtype, device=d_attn.device)

    grid = lambda meta:(head, B, triton.cdiv(n, meta['BLOCK_SIZE_N']))

    stream1 = torch.cuda.Stream()       # 默认优先级的流1
    stream2 = torch.cuda.Stream()       # 默认优先级的流2

    with torch.cuda.stream(stream1):
        fused_backward_kv_kernel[grid](
            q, k, v, rab,
            d_q, d_k, d_v, d_rab,
            d_attn ,
            attn_mask,
            x_offsets,
            B, head, n, dim,
            k.stride(0), k.stride(1), k.stride(2),
            d_k.stride(0), d_k.stride(1), d_k.stride(2),
            q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            d_v.stride(0), d_v.stride(1), d_v.stride(2),
            rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
            attn_mask.stride(2), attn_mask.stride(3),
            d_attn.stride(0), d_attn.stride(1), d_attn.stride(2),
        )
    # print("fused_backward_kv_kernel finished")
    # assert(d_q.sum() != 0).any()

    with torch.cuda.stream(stream2):
        fused_backward_q_kernel[grid](
            q, k, v, rab,
            d_q, d_k, d_v, d_rab,
            d_attn ,
            attn_mask,
            x_offsets,
            B, head, n, dim,
            k.stride(0), k.stride(1), k.stride(2),
            q.stride(0), q.stride(1), q.stride(2),
            d_q.stride(0), d_q.stride(1), d_q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            rab.stride(0), rab.stride(1), rab.stride(2), rab.stride(3),
            d_rab.stride(0), d_rab.stride(1), d_rab.stride(2), d_rab.stride(3),
            attn_mask.stride(2), attn_mask.stride(3),
            d_attn.stride(0), d_attn.stride(1), d_attn.stride(2),
        )
    stream1.synchronize()
    stream2.synchronize()
    return d_q, d_k, d_v, d_rab



