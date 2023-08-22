"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = qvk_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = qvk_offset + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # write back l
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc.to(tl.float16))

@triton.jit
def _bwd_preprocess(
    Out, DO,
    NewDO, Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    # See fwd pass above for explanation.
    qk_scale = sm_scale * 1.44269504
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[None, :] * stride_qm + offs_k[:, None] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk * qk_scale - l_i[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v)
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    off_q = qvk_offset + start_m * BLOCK_M * stride_qm + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # K and V are used transposed only, so load them transposed.
    off_k = qvk_offset + offs_m[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = qvk_offset + offs_m[None, :] * stride_vk + offs_d[:, None] * stride_vn
    off_do = qvk_offset + start_m * BLOCK_M * stride_qm +  offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    do_ptrs = DO + off_do
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(k_ptrs)
    k = (k * qk_scale).to(tl.float16)
    v = tl.load(v_ptrs)
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = start_m * BLOCK_M
    hi = N_CTX
    # loop over q, do
    for start_n in range(lo, hi, BLOCK_N):
        offs_m_curr = offs_n[:, None] + start_n
        # -- load q, do --
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - Di
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        q_ptrs += BLOCK_N * stride_qm
        do_ptrs += BLOCK_N * stride_qm
    # initialize pointers to output
    off_dk = off_hz * stride_kh + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    off_dv = off_hz * stride_vh + offs_m[:, None] * stride_vk + offs_n[None, :] * stride_vn
    dk_ptrs = DK + off_dk
    dv_ptrs = DV + off_dv
    tl.store(dk_ptrs, (dk * sm_scale).to(tl.float16))
    tl.store(dv_ptrs, dv.to(tl.float16))

@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = qvk_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = qvk_offset + offs_n[None, :] * stride_vk + offs_d[:, None] * stride_vn
    off_do = qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    do_ptrs = DO + off_do
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(tl.float16)
    do = tl.load(do_ptrs)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        dq += tl.dot(ds.to(Q.dtype.element_ty), tl.trans(k))
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # initialize pointers to output
    off_dq = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    dq_ptrs = DQ + off_dq
    tl.store(dq_ptrs, (dq * sm_scale).to(tl.float16))

empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            IS_CAUSAL=causal,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk, num_warps=num_warps,
            num_stages=2,
        )
        # print(h.asm["ttgir"])

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, l = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            do_scaled, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            l,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4,
            num_stages=1,
        )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None


attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    # This is from the attention algorithm.
    sm_scale = q.shape[-1] ** (-0.5)
    causal = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        assert torch.allclose(ref_dv, tri_dv, atol=0.5, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(10, 14)],
    line_arg='provider',
    line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
    line_names=['Triton'] + (['Flash'] if HAS_FLASH else []),
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, 'mode': mode}
) for mode in ['fwd', 'bwd']]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    causal = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "flash":
        lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn((BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=True)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path='.', print_data=True)
