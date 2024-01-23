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
    print('seqlens_q= ', seqlens_q)
    print('seqlens_k= ', seqlens_k)

    max_seqlen_q=max(seqlens_q)
    max_seqlen_k=max(seqlens_k)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0]), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0]), seqlens_k.cumsum(dim=0)])

    num_seqs=len(cu_seqlens_q) -1
    print("num_seqs= ", num_seqs)

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = torch.randn((total_q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    print('q.shape ', q.shape)
    print('k.shape', k.shape)
    print('v.shape', v.shape)

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

    # Concatenating along the number of heads dimension
    p= sm_scale * torch.einsum('qhd, khd -> qhk',q,k).half()
    print("p.shape=",p.shape)

    start=0
    for seq_len in seqlens_q:

        end=start+seq_len

        mask = torch.tril(torch.ones((seq_len, seq_len), device="cuda"))
        if mask.size(0) != seq_len or mask.size(1) != seq_len:
           raise ValueError(f"Mask shape {mask.size()} does not match sequence length {seq_len}")
        expanded_mask = mask.unsqueeze(1).expand(seq_len, H, seq_len)

        p[start:end, :, start:end] = p[start:end, :, start:end].masked_fill(expanded_mask == 0, float("-inf"))
        start += seq_len

#   if use_bias:
#       # Add bias here as per the original implementation
#       pass

    p = torch.softmax(p, dim=-1)
    ref_out= torch.einsum('qhk, khd -> qhd',p,v).half()

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
