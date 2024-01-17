import torch

def test_op_fwd_varlen(Z, H, N_CTX, D_HEAD, causal, use_bias, bias_type, dtype=torch.float16):
    torch.manual_seed(20)
    # Random sequence lengths
    seqlens_q = torch.randint(1, max_seqlen_q + 1, (Z,))
    seqlens_k = torch.randint(1, max_seqlen_k + 1, (Z,))
    print(seqlens_q)
    print(seqlens_k)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0]), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0]), seqlens_k.cumsum(dim=0)])

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = torch.randn((total_q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()

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

    # Reference implementation with masking for variable lengths
    M = torch.zeros((max_seqlen_q, max_seqlen_k), device="cuda")
    for i, (len_q, len_k) in enumerate(zip(cu_seqlens_q[1:], cu_seqlens_k[1:])):
        M[i, :len_k] = 1

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        causal_mask = torch.tril(torch.ones((max_seqlen_q, max_seqlen_k), device="cuda"))
        M *= causal_mask

    p = p.masked_fill(M.unsqueeze(1).unsqueeze(0) == 0, float("-inf"))

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
max_seqlen_q = 10  # Maximum length of any sequence in the query batch
max_seqlen_k = 10  # Maximum length of any sequence in the key/value batch
D_HEAD = 64  # Dimension of each head
causal = False
use_bias = False
bias_type = "vector"

test_op_fwd_varlen(Z, H, max_seqlen_q, max_seqlen_k, D_HEAD, causal, use_bias, bias_type)
