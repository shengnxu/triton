import pytest
import random
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz

def test_op_fwd_varlen(Z, H, D_HEAD, causal,
        use_bias, bias_type, dtype=torch.float16):
    torch.manual_seed(20)
    # Random sequence lengths
    max_seqlens_q=10
    seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z,))
    seqlens_k=seqlens_q

    print(seqlens_q)
    print(seqlens_k)

    max_seqlen_q=max(seqlens_q)
    max_seqlen_k=max(seqlens_k)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0]), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0]), seqlens_k.cumsum(dim=0)])

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = torch.randn((H, total_q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((H, total_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((H, total_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    print(q.shape)
    print(k.shape)
    print(v.shape)

    # Initialize bias
    if use_bias:
        if bias_type == "vector":
            bias = torch.randn((1, H, 1, max_seqlen_k), dtype=torch.float32, device="cuda")
        elif bias_type == "matrix":
            bias = torch.randn((1, H, max_seqlen_k, max_seqlen_k), dtype=torch.float32, device="cuda")
    else:
        bias = None

    if TORCH_HAS_FP8E5:
        q = q.to(torch_dtype)
        k = k.to(torch_dtype)
    sm_scale = D_HEAD ** -0.5

    p = torch.matmul(q, k.transpose(1, 2)) * sm_scale
    print(p.shape)
    # Reference implementation with masking for variable lengths
    M = torch.zeros((Z, max_seqlen_q, max_seqlen_k), device="cuda")

    # Adjust the mask for each sequence based on their actual lengths
    for i, (len_q, len_k) in enumerate(zip(seqlens_q, seqlens_k)):
        M[i, :len_q, :len_k] = 1

    # Now reshape and expand M to match the shape of p
    # p has shape [total_q, H, max_seqlen_k]
    # We need to align the sequence dimension of M with the total_q dimension
    expanded_M = torch.zeros_like(p, dtype=torch.bool)
    idx = 0
    for i, len_q in enumerate(seqlens_q):
        expanded_M[idx:idx+len_q] = M[i, :len_q].unsqueeze(1)
        idx += len_q


    if causal:
       causal_mask = torch.tril(torch.ones((max_seqlen_q, max_seqlen_k), device="cuda"))
       M *= causal_mask

    # Apply the mask
    p = p.masked_fill(expanded_M == 0, float("-inf"))

#   if use_bias:
#       # Add bias here as per the original implementation
#       pass

    p = torch.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v)

    # Triton implementation (or other custom implementation)
#    tri_out = attention(q, k, v, causal, bias, sm_scale)

    # Compare outputs
#    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)

Z = 3  # Number of sequences in the batch
H = 4  # Number of attention heads
#max_seqlen_q = 10  # Maximum length of any sequence in the query batch
#max_seqlen_k = 10  # Maximum length of any sequence in the key/value batch
D_HEAD = 64  # Dimension of each head
causal = False
use_bias = False
bias_type = "vector"

test_op_fwd_varlen(Z, H, D_HEAD, causal, use_bias, bias_type)
