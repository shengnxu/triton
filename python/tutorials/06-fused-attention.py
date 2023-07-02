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
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # tl.device_print("", tl.max(qk, 1))
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess(
    Out, DO, L,
    NewDO, Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L, M,
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
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
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
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
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


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        if torch.version.hip is not None:
            BLOCK = 256
        else:
            BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L, m,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk, num_warps=8,
            num_stages=1,
        )
        # print(h.asm["ttgir"])

        ctx.save_for_backward(q, k, v, o, L, m)
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
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do, l,
            do_scaled, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            l, m,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            num_stages=1,
        )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None


attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(1, 1, 64, 64)])
def test_fa_compiled_ir(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_fwd_kernel_0d1d2d34d5d6d7d8d9d10c11d12d13d14c15d16d17d18c19d20d21d22c2324d25d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.splat %2 : (i32) -> tensor<128xi32, #blocked>
    %7 = tt.splat %2 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %8 = tt.splat %2 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %9 = arith.addi %6, %3 : tensor<128xi32, #blocked>
    %10 = arith.addi %7, %4 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %12 = arith.muli %1, %arg8 : i32
    %13 = tt.expand_dims %10 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x1xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %14 = tt.expand_dims %11 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %15 = tt.splat %arg9 : (i32) -> tensor<128x1xi32, #blocked1>
    %16 = arith.muli %14, %15 : tensor<128x1xi32, #blocked1>
    %17 = tt.splat %12 : (i32) -> tensor<128x1xi32, #blocked1>
    %18 = arith.addi %17, %16 : tensor<128x1xi32, #blocked1>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi32, #blocked1>
    %21 = tt.broadcast %18 : (tensor<128x1xi32, #blocked1>) -> tensor<128x64xi32, #blocked1>
    %22 = tt.broadcast %20 : (tensor<1x64xi32, #blocked1>) -> tensor<128x64xi32, #blocked1>
    %23 = arith.addi %21, %22 : tensor<128x64xi32, #blocked1>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %25 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
    %26 = tt.expand_dims %24 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    %27 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<1x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %28 = tt.splat %arg12 : (i32) -> tensor<1x128xi32, #blocked2>
    %29 = arith.muli %26, %28 : tensor<1x128xi32, #blocked2>
    %30 = tt.splat %12 : (i32) -> tensor<1x128xi32, #blocked2>
    %31 = arith.addi %30, %29 : tensor<1x128xi32, #blocked2>
    %32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
    %34 = tt.broadcast %31 : (tensor<1x128xi32, #blocked2>) -> tensor<64x128xi32, #blocked2>
    %35 = tt.broadcast %33 : (tensor<64x1xi32, #blocked2>) -> tensor<64x128xi32, #blocked2>
    %36 = arith.addi %34, %35 : tensor<64x128xi32, #blocked2>
    %37 = tt.expand_dims %5 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %38 = arith.muli %37, %15 : tensor<128x1xi32, #blocked1>
    %39 = arith.addi %17, %38 : tensor<128x1xi32, #blocked1>
    %40 = tt.broadcast %39 : (tensor<128x1xi32, #blocked1>) -> tensor<128x64xi32, #blocked1>
    %41 = arith.addi %40, %22 : tensor<128x64xi32, #blocked1>
    %42 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %43 = tt.addptr %42, %23 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %44 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<64x128x!tt.ptr<f16>, #blocked2>
    %45 = tt.addptr %44, %36 : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
    %46 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %47 = tt.addptr %46, %41 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %48 = tt.load %43 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %49 = triton_gpu.convert_layout %48 : (tensor<128x64xf16, #blocked1>) -> tensor<128x64xf16, #shared>
    %50 = arith.addi %0, %c1_i32 : i32
    %51 = arith.muli %50, %c128_i32 : i32
    %52 = tt.splat %arg3 : (f32) -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %53 = tt.broadcast %13 : (tensor<128x1xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %54 = arith.muli %arg12, %c128_i32 : i32
    %55 = tt.splat %54 : (i32) -> tensor<64x128xi32, #blocked2>
    %56 = arith.muli %arg15, %c128_i32 : i32
    %57 = tt.splat %56 : (i32) -> tensor<128x64xi32, #blocked1>
    %58:5 = scf.for %arg22 = %c0_i32 to %51 step %c128_i32 iter_args(%arg23 = %cst, %arg24 = %cst_0, %arg25 = %cst_1, %arg26 = %45, %arg27 = %47) -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %79 = tt.load %arg26 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked2>
      %80 = triton_gpu.convert_layout %79 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared1>
      %81 = triton_gpu.convert_layout %49 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %82 = triton_gpu.convert_layout %80 : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %83 = tt.view %81 : (tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>>
      %84 = tt.view %82 : (tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>>
      %85 = tt.dot %84, %83, %cst_3 {allowTF32 = true} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>> -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>
      %86 = tt.view %85 : (tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>) -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %87 = arith.mulf %86, %52 : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %88 = tt.splat %arg22 : (i32) -> tensor<1x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %89 = arith.addi %88, %27 : tensor<1x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %90 = tt.broadcast %89 : (tensor<1x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %91 = "triton_gpu.cmpi"(%53, %90) <{predicate = 5 : i64}> : (tensor<128x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>, tensor<128x128xi32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xi1, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %92 = "triton_gpu.select"(%91, %87, %cst_2) : (tensor<128x128xi1, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>, tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>, tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %93 = "tt.reduce"(%92) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %125 = "triton_gpu.cmpf"(%arg28, %arg29) <{predicate = 2 : i64}> : (f32, f32) -> i1
        %126 = arith.select %125, %arg28, %arg29 : f32
        tt.reduce.return %126 : f32
      }) : (tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %94 = "triton_gpu.cmpf"(%93, %arg25) <{predicate = 2 : i64}> : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %95 = "triton_gpu.select"(%94, %93, %arg25) : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %96 = arith.subf %arg25, %95 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %97 = math.exp %96 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %98 = arith.mulf %arg23, %97 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %99 = tt.expand_dims %95 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %100 = tt.broadcast %99 : (tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %101 = arith.subf %92, %100 : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %102 = math.exp %101 : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %103 = "tt.reduce"(%102) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %125 = arith.addf %arg28, %arg29 : f32
        tt.reduce.return %125 : f32
      }) : (tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %104 = arith.addf %103, %98 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %105 = arith.divf %cst_4, %104 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %106 = tt.expand_dims %105 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %107 = tt.broadcast %106 : (tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %108 = arith.mulf %102, %107 : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %109 = arith.mulf %98, %105 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %110 = tt.expand_dims %109 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %111 = tt.broadcast %110 : (tensor<128x1xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %112 = arith.mulf %arg24, %111 : tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %113 = arith.truncf %108 : tensor<128x128xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>> to tensor<128x128xf16, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %114 = tt.load %arg27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
      %115 = triton_gpu.convert_layout %114 : (tensor<128x64xf16, #blocked1>) -> tensor<128x64xf16, #shared>
      %116 = triton_gpu.convert_layout %113 : (tensor<128x128xf16, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %117 = triton_gpu.convert_layout %115 : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>
      %118 = tt.view %116 : (tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>>
      %119 = tt.view %117 : (tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>>
      %120 = tt.view %112 : (tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<64x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>
      %121 = tt.dot %119, %118, %120 {allowTF32 = true} : tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>}>> -> tensor<64x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>
      %122 = tt.view %121 : (tensor<64x128xf32, #triton_gpu.mfma<{warpsPerCTA = [1, 4], isTransposed = true}>>) -> tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
      %123 = tt.addptr %arg26, %55 : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
      %124 = tt.addptr %arg27, %57 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %104, %122, %95, %123, %124 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>, tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    %59 = triton_gpu.convert_layout %58#2 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128xf32, #blocked>
    %60 = triton_gpu.convert_layout %58#0 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>}>>) -> tensor<128xf32, #blocked>
    %61 = arith.muli %1, %arg21 : i32
    %62 = tt.addptr %arg4, %61 : !tt.ptr<f32>, i32
    %63 = tt.splat %62 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>, #blocked>
    %64 = tt.addptr %63, %9 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %65 = tt.addptr %arg5, %61 : !tt.ptr<f32>, i32
    %66 = tt.splat %65 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>, #blocked>
    %67 = tt.addptr %66, %9 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %64, %60 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked>
    tt.store %67, %59 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked>
    %68 = arith.muli %1, %arg17 : i32
    %69 = tt.splat %arg18 : (i32) -> tensor<128x1xi32, #blocked1>
    %70 = arith.muli %14, %69 : tensor<128x1xi32, #blocked1>
    %71 = tt.splat %68 : (i32) -> tensor<128x1xi32, #blocked1>
    %72 = arith.addi %71, %70 : tensor<128x1xi32, #blocked1>
    %73 = tt.broadcast %72 : (tensor<128x1xi32, #blocked1>) -> tensor<128x64xi32, #blocked1>
    %74 = arith.addi %73, %22 : tensor<128x64xi32, #blocked1>
    %75 = tt.splat %arg6 : (!tt.ptr<f16>) -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %76 = tt.addptr %75, %74 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %77 = arith.truncf %58#1 : tensor<128x64xf32, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>> to tensor<128x64xf16, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>
    %78 = triton_gpu.convert_layout %77 : (tensor<128x64xf16, #triton_gpu.mfma<{warpsPerCTA = [4, 1], isTransposed = true}>>) -> tensor<128x64xf16, #blocked1>
    tt.store %76, %78 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64xf16, #blocked1>
    tt.return
  }
}
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        _fwd_kernel = triton.compile(f.name)
    
    if torch.version.hip is not None:
        BLOCK = 128
    else:
        BLOCK = 128
        # shape constraints
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    # tri_out = attention(q, k, v, sm_scale)

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    # Z, H, N_CTX,
    # BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    # BLOCK_N: tl.constexpr,

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L, m,
        o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(2), k.stride(3),
        v.stride(2),
        o.stride(1), o.stride(2), o.stride(3),
        q.shape[2],
        BLOCK, Lk, BLOCK, 4, 1
    )
    tri_out = o
    # Q, K, V, sm_scale,
    # L, M,
    # Out,
    # stride_qz, stride_qh, stride_qm, stride_qk,
    # stride_kz, stride_kh, stride_kn, stride_kk,
    # stride_vz, stride_vh, stride_vk, stride_vn,
    # stride_oz, stride_oh, stride_om, stride_on,
    # Z, H, N_CTX,
    # BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    # BLOCK_N: tl.constexpr,

    assert torch.testing.assert_allclose(ref_out, tri_out, atol=1e-2, rtol=0)

        # print(h.asm["ttgir"])

    # rs = RandomState(17)
    # x = rs.randint(0, 4, (M, N)).astype('float32')
    # x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')

    # if axis == 0:
    #     z = np.zeros((1, N)).astype('float32')
    # else:
    #     z = np.zeros((M, 1)).astype('float32')

    # x_tri = torch.tensor(x, device=device)
    # z_tri = torch.tensor(z, device=device)
    # pgm = kernel[(1, 1, 4)](x_tri, x_tri.stride(0), z_tri)

    # z_ref = np.max(x, axis=axis, keepdims=True)

    # np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, sm_scale)
    # print(ref_out)
    # print(tri_out)
    if torch.version.hip is None:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        # TODO: Enable backward pass for MFMA dot.
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
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
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
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
# bench_flash_attention.run(save_path='.', print_data=True)
