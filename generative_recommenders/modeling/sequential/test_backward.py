import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import fbgemm_gpu


def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    rab = torch.randn(B, 1, n, n, device="cuda")
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, rab, attn_mask

def origin_einsum_attn(q, k, v, rab, attn_mask, B, n, head, d, x_offsets):
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
    qk_attn = qk_attn + rab
    qk_attn = F.silu(qk_attn) / n #SiLU之后局部归一化
    qk_attn = qk_attn * attn_mask
    attn_output = torch.ops.fbgemm.dense_to_jagged( #Φ(qk)v  , dense_to_jagged将输出转换为(sum_N, head*d)形状
            torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                    B, n, head, d  #将v转换为padded形式
                ),
            ).reshape(B, n, head * d), 
            [x_offsets],
        )[0]
    return attn_output


def attn_forward(q, k, v, rab, attn_mask, B, n, head, d, x_offsets):
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
    qk_attn = qk_attn + rab
    qk_attn = F.silu(qk_attn) / n #SiLU之后局部归一化
    qk_attn = qk_attn * attn_mask
    attn_output = torch.ops.fbgemm.dense_to_jagged( #Φ(qk)v  , dense_to_jagged将输出转换为(sum_N, head*d)形状
            torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                    B, n, head, d  #将v转换为padded形式
                ),
            ).reshape(B, n, head * d), 
            [x_offsets],
        )[0]
    return attn_output


class CustomAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rab, attn_mask, B, n, head, d, x_offsets):
        """
        前向传播：
        q/k/v: 稀疏张量，形状为 (sum_N, num_heads*d)
        x_offsets: 用于描述稀疏结构的偏移量
        rab: 位置偏置项，形状 (B, num_heads, n, n)
        attn_mask: 注意力掩码，形状 (B, n, n)
        n: 填充后的序列长度
        """
        # B = len(x_offsets) - 1  # batch size
        # head = q.shape[1] // (k.shape[-1] // n)  # 推导头数（假设 d 相同）
        # d = k.shape[-1] // head

        # Step 1: 稀疏转稠密（q, k）
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            q, [x_offsets], [n], padding_value=0.0
        ).view(B, n, head, d)  # (B, n, h, d)

        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            k, [x_offsets], [n], padding_value=0.0
        ).view(B, n, head, d)

        # Step 2: 计算注意力分数
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)  # (B, h, n, m)
        qk_attn = qk_attn + rab  # 添加位置偏置
        
        # Step 3: 激活函数 + 归一化 + 掩码
        silu_qk = F.silu(qk_attn)  # SiLU激活
        normalized_qk = silu_qk / n  # 局部归一化
        masked_qk = normalized_qk * attn_mask  # (B, h, n, n)

        # Step 4: 计算注意力输出（需要将v填充）
        padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
            v, [x_offsets], [n], padding_value=0.0
        ).view(B, n, head, d)

        attn_output = torch.einsum("bhnm,bmhd->bnhd", masked_qk, padded_v)  # (B, n, h, d)
        attn_output = attn_output.reshape(B, n, -1)  # (B, n, h*d)
        
        # Step 5: 转换回稀疏格式
        attn_output_sparse = torch.ops.fbgemm.dense_to_jagged(
            attn_output, [x_offsets]
        )[0]  # (sum_N, h*d)

        # 保存反向传播所需的中间变量
        ctx.save_for_backward(
            q, k, v,
            padded_q, padded_k, padded_v,
            qk_attn, silu_qk, masked_qk,
            rab, attn_mask
        )
        ctx.x_offsets = x_offsets
        ctx.n = n
        ctx.B, ctx.head, ctx.d = B, head, d

        return attn_output_sparse

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：
        grad_output: 输出的梯度，形状 (sum_N, num_heads*d)
        需要计算 q, k, v, rab, attn_mask 的梯度（后两者如果 requires_grad=True）
        """
        # 获取保存的中间变量
        (q, k, v,
         padded_q, padded_k, padded_v,
         qk_attn, silu_qk, masked_qk,
         rab, attn_mask) = ctx.saved_tensors
        B, head, d = ctx.B, ctx.head, ctx.d
        n = ctx.n
        x_offsets = ctx.x_offsets

        # ---------------------------------------------------------------
        # 反向步骤 1: dense_to_jagged 的梯度
        # 将稀疏梯度 grad_output 转换为稠密形式
        grad_attn_output = torch.ops.fbgemm.jagged_to_padded_dense(
            grad_output, [x_offsets], [n], padding_value=0.0
        ).view(B, n, head, d)  # (B, n, h, d)
        #print(grad_attn_output.shape)
        # ---------------------------------------------------------------
        # 反向步骤 2: einsum("bhnm,bmhd->bnhd") 的梯度
        # 计算对 masked_qk 和 padded_v 的梯度
        # 公式: d(masked_qk) = grad_attn_output @ padded_v^T
        #        d(padded_v) = masked_qk^T @ grad_attn_output
        grad_masked_qk = torch.einsum("bnhd,bmhd->bhnm", grad_attn_output, padded_v)
        grad_padded_v = torch.einsum("bhnm,bnhd->bmhd", masked_qk, grad_attn_output)
        #到这一步都没问题


        # print(grad_masked_qk.shape)
        # print(grad_padded_v.shape)
        # ---------------------------------------------------------------
        # 反向步骤 3: 掩码 + 归一化 + SiLU 的梯度
        # 公式: grad_normalized = grad_masked_qk * attn_mask.unsqueeze(1)
        #       grad_silu = grad_normalized / n
        #       grad_qk_attn = grad_silu * silu_derivative(qk_attn + rab)
        
        # 计算 SiLU 的导数: silu_grad = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        x = qk_attn  # qk_attn + rab? or qk_attn * n ?  
        sigmoid_x = torch.sigmoid(x)
        silu_derivative = sigmoid_x * (1 + x * (1 - sigmoid_x))
        
        # 反向传播链式法则
        grad_normalized = grad_masked_qk * attn_mask  # 掩码梯度
        grad_silu = grad_normalized / n  # 归一化梯度
        grad_qk_attn = grad_silu * silu_derivative  # SiLU 梯度

        # ---------------------------------------------------------------
        # 反向步骤 4: 位置偏置 rab 的梯度（若需要）
        grad_rab = grad_qk_attn if rab.requires_grad else None

        # ---------------------------------------------------------------
        # 反向步骤 5: einsum("bnhd,bmhd->bhnm") 的梯度
        # 计算对 padded_q 和 padded_k 的梯度
        # 公式: d(padded_q) = grad_qk_attn @ padded_k
        #        d(padded_k) = grad_qk_attn^T @ padded_q
        grad_padded_q = torch.einsum("bhnm,bmhd->bnhd", grad_qk_attn, padded_k)
        grad_padded_k = torch.einsum("bhnm,bnhd->bmhd", grad_qk_attn, padded_q)

        #print(grad_padded_q.shape)
        grad_q = torch.ops.fbgemm.dense_to_jagged(
            grad_padded_q.reshape(B, n, head * d), [x_offsets]
        )[0]
        grad_k = torch.ops.fbgemm.dense_to_jagged(
            grad_padded_k.reshape(B, n, head * d), [x_offsets]
        )[0]
        grad_v = torch.ops.fbgemm.dense_to_jagged(
            grad_padded_v.reshape(B, n, head * d), [x_offsets]
        )[0]
        print(grad_q.shape)

        # ---------------------------------------------------------------
        # 反向步骤 6: jagged_to_padded_dense 的梯度（q, k, v）
        # 将稠密梯度转换回稀疏格式
        def dense_to_sparse_grad(grad_padded, original_tensor):
            # 将梯度从 (B, n, ...) 转换回原始稀疏形状
            grad_flat = grad_padded.view(B * n, -1)
            lengths = x_offsets[1:] - x_offsets[:-1]  # 每个样本的实际长度
            grad_sparse = torch.ops.fbgemm.dense_to_jagged(
                grad_flat, [x_offsets]
            )[0]
            return grad_sparse.contiguous()

        # grad_q = dense_to_sparse_grad(grad_padded_q, q)
        # grad_k = dense_to_sparse_grad(grad_padded_k, k)
        # grad_v = dense_to_sparse_grad(grad_padded_v, v)

        # ---------------------------------------------------------------
        # 注意：attn_mask 的梯度通常不需要（如果是固定掩码）
        grad_attn_mask = None

        # 返回梯度顺序需与 forward 的输入参数一致
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None

# 使用示例
# B = 2
# seq_len = [12, 13, 10]
# n = 0
# head = 4
# d = 16
# #sum_N = 25  # 稀疏序列总长度

# x_offsets = [0]
# for i in range(1, B+1):
#     rand_seq_len = random.choice(seq_len)
#     n = max(n, rand_seq_len)
#     x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
# x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor
# sum_N = int(x_offsets[-1])
# #x_offsets = torch.tensor([0, 12, 25], dtype=torch.int32, device="cuda")  # B=2
# rab = torch.randn(B, head, n, n, requires_grad=True, device="cuda")
# #attn_mask = torch.randn(B, n, n, device="cuda") > 0  # 布尔掩码
# attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
#     # 调整形状为 (1, 1, n, n)
# attn_mask = attn_mask.view(1, 1, n, n) 

# # 模拟输入
# q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
# k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
# v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
interval = [256, 510, 1020]
n = max(interval)
B = 20
x_offsets = [0]
for i in range(1, B+1):
    x_offsets.append(x_offsets[-1] + random.choice(interval))
x_offsets = torch.tensor(x_offsets, device="cuda")

head, d = 8 , 32
sum_N = x_offsets[-1]

q, k, v, rab, attn_mask = get_input(sum_N, head, d, B, n)

q1 = q.clone().detach().requires_grad_(True)
k1 = k.clone().detach().requires_grad_(True)
v1 = v.clone().detach().requires_grad_(True)

# 前向计算
output = CustomAttentionFunction.apply(q, k, v, rab, attn_mask, B, n, head, d, x_offsets)
loss = output.sum()
loss.backward()
print(q.grad[0, :])

output1 = origin_einsum_attn(q1, k1, v1, rab, attn_mask, B, n, head, d, x_offsets)

loss1 = output1.sum()
print('diff between two forward: ', (output - output1).abs().mean(), (output - output1).abs().max())
loss1.backward()

print('diff between two q backward: ', (q.grad - q1.grad).abs().mean(), (q.grad - q1.grad).abs().max())
print('diff between two k backward: ', (k.grad - k1.grad).abs().mean(), (k.grad - k1.grad).abs().max())
print('diff between two v backward: ', (v.grad - v1.grad).abs().mean(), (v.grad - v1.grad).abs().max())
print(q1.grad[0, :])
# 验证梯度存在
print(q.grad.shape)  # torch.Size([25, 64])
print(k.grad.shape)  # torch.Size([25, 64])
print(v.grad.shape)  # torch.Size([25, 64])
#print(rab.grad.shape) # torch.Size([2, 4, 10, 10])