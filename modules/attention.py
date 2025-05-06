import math

import torch
import torch.nn.functional as F


try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_varlen
    )
    has_sageattn = True
except ImportError:
    has_sageattn = False
    sageattn = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_varlen = None


MEMORY_LAYOUT = {
    # flash模式:
    # 预处理: 输入 [batch_size, seq_len, num_heads, head_dim]
    # 后处理: 保持形状不变
    "flash": (
        lambda x: x,  # 保持形状
        lambda x: x,  # 保持形状
    ),
    # torch/vanilla模式:
    # 预处理: 交换序列和注意力头的维度 [B,S,A,D] -> [B,A,S,D]
    # 后处理: 交换回原始维度 [B,A,S,D] -> [B,S,A,D]
    "torch": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "sage": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    ),
    "sage-nhd": (
        lambda x: x,
        lambda x: x,
    ),
}


def get_optimal_sage_kernel(device_type, dtype):
    """
    Select the optimal SageAttention kernel based on GPU type and data type
    
    Args:
        device_type (str): GPU architecture (e.g., 'cuda:0')
        dtype (torch.dtype): Data type (e.g., torch.float16, torch.bfloat16)
    
    Returns:
        function: Optimal SageAttention function for the hardware
    """
    # Default to auto-selection
    if not has_sageattn:
        raise ImportError("SageAttention is not installed. Please install it first.")
    
    device_prop = torch.cuda.get_device_properties(device_type)
    compute_capability = device_prop.major * 10 + device_prop.minor
    
    if compute_capability >= 90:
        return sageattn_qk_int8_pv_fp8_cuda_sm90
    
    elif compute_capability >= 89:
        if torch.cuda.get_device_capability(device_type)[0] >= 12 and torch.cuda.get_device_capability(device_type)[1] >= 4:
            return sageattn_qk_int8_pv_fp8_cuda
        else:
            return sageattn_qk_int8_pv_fp16_cuda
    
    elif compute_capability >= 80:
        return sageattn_qk_int8_pv_fp16_cuda
    
    else:
        return sageattn_qk_int8_pv_fp16_triton


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    sage_tensor_layout="HND",
):
    """
    执行QKV自注意力计算

    Args:
        q (torch.Tensor): 查询张量，形状 [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): 键张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        v (torch.Tensor): 值张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        mode (str): 注意力模式，可选 'flash', 'torch', 'vanilla', 'sage', 'sage-nhd'
        drop_rate (float): 注意力矩阵的dropout概率
        attn_mask (torch.Tensor): 注意力掩码，形状根据模式不同而变化
        causal (bool): 是否使用因果注意力（仅关注前面位置）
        sage_tensor_layout (str): SageAttention的张量布局，可选 'HND' 或 'NHD'

    Returns:
        torch.Tensor: 注意力输出，形状 [batch_size, seq_len, num_heads * head_dim]
    """
    if mode in ['sage', 'sage-nhd'] and not has_sageattn:
        raise ImportError("SageAttention is not installed. Falling back to 'flash' mode.")
        mode = 'flash'
    
    if mode == 'sage':
        actual_tensor_layout = "HND"  # [B,A,S,D] format after pre_attn_layout
    elif mode == 'sage-nhd':
        actual_tensor_layout = "NHD"  # [B,S,A,D] format after pre_attn_layout
    else:
        actual_tensor_layout = None  # Not used for other modes
    
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode in ['sage', 'sage-nhd']:
        if q.dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError(f"SageAttention requires torch.float16 or torch.bfloat16 inputs, got {q.dtype}")
        
        if attn_mask is not None:
            import warnings
            warnings.warn("SageAttention does not support custom attention masks. Using causal flag only.")
        
        if q.shape[-3] != k.shape[-3]:
            x = sageattn(
                q, k, v, 
                tensor_layout=actual_tensor_layout,
                is_causal=causal
            )
        else:
            optimal_kernel = get_optimal_sage_kernel(q.device, q.dtype)
            
            x = optimal_kernel(
                q, k, v,
                tensor_layout=actual_tensor_layout,
                is_causal=causal
            )

    elif mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        assert flash_attn_func is not None, "flash_attn_func未定义"
        assert attn_mask is None, "不支持的注意力掩码"
        x: torch.Tensor = flash_attn_func(
            q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None
        )  # type: ignore
    elif mode == "vanilla":
        # 手动实现注意力机制
        scale_factor = 1 / math.sqrt(q.size(-1))  # 缩放因子 1/sqrt(d_k)

        b, a, s, _ = q.shape  # 获取形状参数
        s1 = k.size(2)  # 键值序列长度

        # 初始化注意力偏置
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)

        # 处理因果掩码
        if causal:
            assert attn_mask is None, "因果掩码和注意力掩码不能同时使用"
            # 生成下三角因果掩码
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(q.dtype)

        # 处理自定义注意力掩码
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask  # 允许类似ALiBi的位置偏置

        # 计算注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * scale_factor  # [B,A,S,S1]
        attn += attn_bias

        # softmax和dropout
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)

        # 计算输出
        x = attn @ v  # [B,A,S,D]
    else:
        raise NotImplementedError(f"不支持的注意力模式: {mode}")

    # 应用后处理变换
    x = post_attn_layout(x)  # 恢复原始维度顺序

    # 合并注意力头维度
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)  # [B,S,A*D]
    return out
