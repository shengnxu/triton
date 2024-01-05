"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16
TORCH_HAS_FP8 = False
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2')
TORCH_HAS_FP8E5FNUZ = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2
    TORCH_HAS_FP8 = True
if TORCH_HAS_FP8E5FNUZ:
    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz
    TORCH_HAS_FP8 = True

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX,
                    pre_load_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
@triton.autotune(
   configs=[
    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'waves_per_eu': 3, 'slice_k_tile': 64, 'pre_load_v': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 64, 'pre_load_v': False}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 64, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'slice_k_tile': 32, 'pre_load_v': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 32, 'pre_load_v': True}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'slice_k_tile': 32, 'pre_load_v': False}, num_stages=1, num_warps=4),
   ],
   key=['Z', 'H', 'N_CTX', 'STAGE', 'BLOCK_DMODEL'],
)


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H,
              N_CTX,
              BLOCK_DMODEL: tl.constexpr,
              STAGE: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              pre_load_v: tl.constexpr,
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(q.dtype)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        pre_load_v,
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        pre_load_v,
                                        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         NewDO, Delta,  #
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(O + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel_dk_dv(Q, K, V, sm_scale,
                      Out, DO,
                      DK, DV,
                      L, D,
                      stride_qz, stride_qh, stride_qm, stride_qk,
                      stride_kz, stride_kh, stride_kn, stride_kk,
                      stride_vz, stride_vh, stride_vk, stride_vn,
                      Z, H, N_CTX,
                      BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_hz * stride_qh
    qdo_offset = qvk_offset + start_m * BLOCK_M * stride_qm
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qdo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qdo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(k.dtype)
    v = tl.load(V_block_ptr)
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
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # -- compute dv ----
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - Di
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_N, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_N, 0))
    # initialize pointers to output
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(tl.float16))

@triton.jit
def _bwd_kernel_dq(Q, K, V, sm_scale,
                   Out, DO,
                   DQ,
                   L,D,
                   stride_qz, stride_qh, stride_qm, stride_qk,
                   stride_kz, stride_kh, stride_kn, stride_kk,
                   stride_vz, stride_vh, stride_vk, stride_vn,
                   Z, H, N_CTX,
                   BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(q.dtype)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(tl.float16))

empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, split_kernel=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8
            # Tuning for H100
            if torch.cuda.get_device_capability()[0] == 9:
                num_warps = 8
                num_stages = 7 if Lk >= 64 else 3

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[0] * q.shape[1],
            1
        )
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            STAGE=stage,
        )

        ## restore the grid for bwd kernel
        best_config = _attn_fwd.get_best_config()
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.split_kernel = split_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        # configuration is not supported
        assert(not (ctx.split_kernel and not ctx.causal))
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        delta = torch.empty_like(L)
        do_scaled = torch.empty_like(do)
        # Figure out what BLOCK size fwd used and adjust num_blocks accordingly.
        # If the two are the same, we don't need this but the bwd pass block size
        # is smaller than the fwd so we need this scaling to ensure we loop over all
        # values and don't skip some blocks. 
        # Alternatively we could compute a new grid but this keeps it consistent
        # with fwd and easier to reason about.
        block_scale = (q.shape[2] // ctx.grid[0]) // BLOCK
        _attn_bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,  #
            do_scaled, delta,  #
            BLOCK_M=block_scale * BLOCK, D_HEAD=ctx.BLOCK_DMODEL,  #
        )
        if not ctx.split_kernel:
            _bwd_kernel[(ctx.grid[1],)](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dq, dk, dv,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                block_scale * ctx.grid[0],
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4,
                CAUSAL=ctx.causal,
                num_stages=1,
            )
        else :
            dq = torch.zeros_like(q)
            _bwd_kernel_dk_dv[(block_scale * ctx.grid[0], ctx.grid[1])](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dk, dv,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4,
                num_stages=1,
            )
            _bwd_kernel_dq[ctx.grid](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dq,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                BLOCK_M=2*BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=4, waves_per_eu=1,
                num_stages=1,
            )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None

attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (4, 48, 1024, 128),
                          (4, 48, 2048, 128),
                          (4, 48, 4096, 128),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
@pytest.mark.parametrize('causal', [False, True])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    if TORCH_HAS_FP8:
        q = q.to(torch_dtype)
        k = k.to(torch_dtype)
    sm_scale = 0.5
    dout = torch.randn_like(q, dtype=torch.float16)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q.half(), k.transpose(2, 3).half()) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (1, 16, 8192, 64),
                          ])
def test_op_bwd(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    causal = True
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5
    split_kernel = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, split_kernel)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)






ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [8, 1], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mfma>
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = arith.mulf %arg3, %cst_2 : f32
    %1 = tt.splat %0 : (f32) -> tensor<256x128xf32, #blocked>
    %2 = triton_gpu.view_slice %1[0, 96] [256, 32] [1, 1] : tensor<256x128xf32, #blocked> to tensor<256x32xf32, #blocked>
    %3 = triton_gpu.view_slice %1[0, 64] [256, 32] [1, 1] : tensor<256x128xf32, #blocked> to tensor<256x32xf32, #blocked>
    %4 = triton_gpu.view_slice %1[0, 32] [256, 32] [1, 1] : tensor<256x128xf32, #blocked> to tensor<256x32xf32, #blocked>
    %5 = triton_gpu.view_slice %1[0, 0] [256, 32] [1, 1] : tensor<256x128xf32, #blocked> to tensor<256x32xf32, #blocked>
    %6 = tt.get_program_id x : i32
    %7 = arith.muli %6, %c256_i32 : i32
    %8 = tt.splat %7 : (i32) -> tensor<256xi32, #blocked1>
    %9 = arith.extsi %7 : i32 to i64
    %10 = tt.splat %9 : (i64) -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %11 = tt.splat %9 : (i64) -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.get_program_id y : i32
    %13 = arith.muli %12, %arg20 : i32
    %14 = tt.addptr %arg4, %13 : !tt.ptr<f32, 1>, i32
    %15 = tt.splat %14 : (!tt.ptr<f32, 1>) -> tensor<256x!tt.ptr<f32, 1>, #blocked1>
    %16 = arith.muli %12, %arg7 : i32
    %17 = tt.addptr %arg5, %16 : !tt.ptr<f16, 1>, i32
    %18 = tt.splat %17 : (!tt.ptr<f16, 1>) -> tensor<256x1x!tt.ptr<f16, 1>, #blocked2>
    %19 = tt.addptr %arg1, %16 : !tt.ptr<f16, 1>, i32
    %20 = tt.splat %19 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked3>
    %21 = tt.addptr %arg2, %16 : !tt.ptr<f16, 1>, i32
    %22 = tt.splat %21 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
    %23 = tt.addptr %arg0, %16 : !tt.ptr<f16, 1>, i32
    %24 = tt.splat %23 : (!tt.ptr<f16, 1>) -> tensor<256x1x!tt.ptr<f16, 1>, #blocked>
    %25 = arith.extsi %arg8 : i32 to i64
    %26 = tt.splat %25 : (i64) -> tensor<256x1xi64, #blocked>
    %27 = arith.extsi %arg14 : i32 to i64
    %28 = tt.splat %27 : (i64) -> tensor<128x1xi64, #blocked2>
    %29 = arith.extsi %arg11 : i32 to i64
    %30 = tt.splat %29 : (i64) -> tensor<1x128xi64, #blocked3>
    %31 = arith.extsi %arg17 : i32 to i64
    %32 = tt.splat %31 : (i64) -> tensor<256x1xi64, #blocked2>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %34 = arith.extsi %33 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %35 = arith.addi %11, %34 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %36 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %37 = arith.extsi %36 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %38 = arith.addi %10, %37 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %39 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %40 = arith.addi %8, %39 : tensor<256xi32, #blocked1>
    %41 = tt.addptr %15, %40 : tensor<256x!tt.ptr<f32, 1>, #blocked1>, tensor<256xi32, #blocked1>
    %42 = tt.expand_dims %35 {axis = 1 : i32} : (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<256x1xi64, #blocked>
    %43 = arith.muli %42, %26 : tensor<256x1xi64, #blocked>
    %44 = tt.addptr %24, %43 : tensor<256x1x!tt.ptr<f16, 1>, #blocked>, tensor<256x1xi64, #blocked>
    %45 = tt.expand_dims %38 {axis = 1 : i32} : (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<256x1xi64, #blocked2>
    %46 = arith.muli %45, %32 : tensor<256x1xi64, #blocked2>
    %47 = tt.addptr %18, %46 : tensor<256x1x!tt.ptr<f16, 1>, #blocked2>, tensor<256x1xi64, #blocked2>
    %48 = tt.broadcast %44 : (tensor<256x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<256x128x!tt.ptr<f16, 1>, #blocked>
    %49 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %50 = arith.extsi %49 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %51 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %52 = arith.extsi %51 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %53 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %54 = arith.extsi %53 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %55 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %56 = arith.extsi %55 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %57 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %58 = arith.extsi %57 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %59 = tt.expand_dims %50 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %60 = tt.expand_dims %52 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
    %61 = tt.broadcast %59 : (tensor<1x128xi64, #blocked>) -> tensor<256x128xi64, #blocked>
    %62 = tt.addptr %48, %61 : tensor<256x128x!tt.ptr<f16, 1>, #blocked>, tensor<256x128xi64, #blocked>
    %63 = triton_gpu.view_slice %62[0, 96] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16, 1>, #blocked> to tensor<256x32x!tt.ptr<f16, 1>, #blocked>
    %64 = tt.load %63 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x32xf16, #blocked>
    %65 = arith.extf %64 : tensor<256x32xf16, #blocked> to tensor<256x32xf32, #blocked>
    %66 = arith.mulf %65, %2 : tensor<256x32xf32, #blocked>
    %67 = arith.truncf %66 : tensor<256x32xf32, #blocked> to tensor<256x32xf16, #blocked>
    %68 = triton_gpu.convert_layout %67 : (tensor<256x32xf16, #blocked>) -> tensor<256x32xf16, #shared>
    %69 = triton_gpu.convert_layout %68 : (tensor<256x32xf16, #shared>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %70 = triton_gpu.view_slice %62[0, 64] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16, 1>, #blocked> to tensor<256x32x!tt.ptr<f16, 1>, #blocked>
    %71 = tt.load %70 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x32xf16, #blocked>
    %72 = arith.extf %71 : tensor<256x32xf16, #blocked> to tensor<256x32xf32, #blocked>
    %73 = arith.mulf %72, %3 : tensor<256x32xf32, #blocked>
    %74 = arith.truncf %73 : tensor<256x32xf32, #blocked> to tensor<256x32xf16, #blocked>
    %75 = triton_gpu.convert_layout %74 : (tensor<256x32xf16, #blocked>) -> tensor<256x32xf16, #shared>
    %76 = triton_gpu.convert_layout %75 : (tensor<256x32xf16, #shared>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %77 = triton_gpu.view_slice %62[0, 32] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16, 1>, #blocked> to tensor<256x32x!tt.ptr<f16, 1>, #blocked>
    %78 = tt.load %77 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x32xf16, #blocked>
    %79 = arith.extf %78 : tensor<256x32xf16, #blocked> to tensor<256x32xf32, #blocked>
    %80 = arith.mulf %79, %4 : tensor<256x32xf32, #blocked>
    %81 = arith.truncf %80 : tensor<256x32xf32, #blocked> to tensor<256x32xf16, #blocked>
    %82 = triton_gpu.convert_layout %81 : (tensor<256x32xf16, #blocked>) -> tensor<256x32xf16, #shared>
    %83 = triton_gpu.convert_layout %82 : (tensor<256x32xf16, #shared>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %84 = triton_gpu.view_slice %62[0, 0] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16, 1>, #blocked> to tensor<256x32x!tt.ptr<f16, 1>, #blocked>
    %85 = tt.load %84 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x32xf16, #blocked>
    %86 = arith.extf %85 : tensor<256x32xf16, #blocked> to tensor<256x32xf32, #blocked>
    %87 = arith.mulf %86, %5 : tensor<256x32xf32, #blocked>
    %88 = arith.truncf %87 : tensor<256x32xf32, #blocked> to tensor<256x32xf16, #blocked>
    %89 = triton_gpu.convert_layout %88 : (tensor<256x32xf16, #blocked>) -> tensor<256x32xf16, #shared>
    %90 = triton_gpu.convert_layout %89 : (tensor<256x32xf16, #shared>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %91 = tt.broadcast %60 : (tensor<1x128xi64, #blocked2>) -> tensor<256x128xi64, #blocked2>
    %92 = tt.expand_dims %54 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi64, #blocked3>
    %93 = tt.addptr %20, %92 : tensor<128x1x!tt.ptr<f16, 1>, #blocked3>, tensor<128x1xi64, #blocked3>
    %94 = tt.broadcast %93 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked3>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked3>
    %95 = tt.broadcast %60 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
    %96:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<256x128xf32, #mfma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %107 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %108 = arith.addi %107, %56 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %109 = tt.expand_dims %108 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x128xi64, #blocked3>
      %110 = arith.muli %109, %30 : tensor<1x128xi64, #blocked3>
      %111 = tt.broadcast %110 : (tensor<1x128xi64, #blocked3>) -> tensor<128x128xi64, #blocked3>
      %112 = tt.addptr %94, %111 : tensor<128x128x!tt.ptr<f16, 1>, #blocked3>, tensor<128x128xi64, #blocked3>
      %113 = triton_gpu.view_slice %112[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %114 = triton_gpu.view_slice %112[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %115 = tt.load %114 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %116 = triton_gpu.view_slice %112[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %117 = tt.load %116 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %118 = triton_gpu.convert_layout %115 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared1>
      %119 = triton_gpu.convert_layout %118 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %120 = triton_gpu.convert_layout %117 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared1>
      %121 = triton_gpu.convert_layout %120 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %122 = tt.dot %90, %119, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %123 = tt.load %113 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %124 = triton_gpu.view_slice %112[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked3> to tensor<32x128x!tt.ptr<f16, 1>, #blocked3>
      %125 = tt.load %124 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked3>
      %126 = triton_gpu.convert_layout %123 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared1>
      %127 = triton_gpu.convert_layout %126 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %128 = tt.dot %83, %121, %122 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %129 = triton_gpu.convert_layout %125 : (tensor<32x128xf16, #blocked3>) -> tensor<32x128xf16, #shared1>
      %130 = triton_gpu.convert_layout %129 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %131 = tt.dot %76, %127, %128 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %132 = tt.dot %69, %130, %131 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %133 = "tt.reduce"(%132) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %185 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %185 : f32
      }) : (tensor<256x128xf32, #mfma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %134 = arith.maximumf %arg24, %133 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %135 = arith.subf %arg24, %134 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %136 = tt.extern_elementwise %135 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %137 = arith.mulf %arg23, %136 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %138 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256x1xf32, #mfma>
      %139 = tt.broadcast %138 : (tensor<256x1xf32, #mfma>) -> tensor<256x128xf32, #mfma>
      %140 = arith.subf %132, %139 : tensor<256x128xf32, #mfma>
      %141 = tt.extern_elementwise %140 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<256x128xf32, #mfma>) -> tensor<256x128xf32, #mfma>
      %142 = arith.truncf %141 : tensor<256x128xf32, #mfma> to tensor<256x128xf16, #mfma>
      %143 = tt.expand_dims %136 {axis = 1 : i32} : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256x1xf32, #mfma>
      %144 = tt.broadcast %143 : (tensor<256x1xf32, #mfma>) -> tensor<256x128xf32, #mfma>
      %145 = arith.mulf %arg22, %144 : tensor<256x128xf32, #mfma>
      %146 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %147 = arith.addi %146, %58 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %148 = tt.expand_dims %147 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
      %149 = arith.muli %148, %28 : tensor<128x1xi64, #blocked2>
      %150 = tt.addptr %22, %149 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
      %151 = tt.broadcast %150 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
      %152 = tt.addptr %151, %95 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
      %153 = triton_gpu.view_slice %152[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %154 = triton_gpu.view_slice %152[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %155 = triton_gpu.view_slice %142[0, 0] [256, 32] [1, 1] : tensor<256x128xf16, #mfma> to tensor<256x32xf16, #mfma>
      %156 = triton_gpu.convert_layout %155 : (tensor<256x32xf16, #mfma>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %157 = tt.load %154 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %158 = triton_gpu.view_slice %152[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %159 = tt.load %158 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %160 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared2>
      %161 = triton_gpu.convert_layout %160 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %162 = triton_gpu.convert_layout %159 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared2>
      %163 = triton_gpu.convert_layout %162 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %164 = tt.dot %156, %161, %145 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %165 = triton_gpu.view_slice %142[0, 32] [256, 32] [1, 1] : tensor<256x128xf16, #mfma> to tensor<256x32xf16, #mfma>
      %166 = triton_gpu.convert_layout %165 : (tensor<256x32xf16, #mfma>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %167 = tt.load %153 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %168 = triton_gpu.view_slice %152[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %169 = tt.load %168 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %170 = triton_gpu.convert_layout %167 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared2>
      %171 = triton_gpu.convert_layout %170 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %172 = tt.dot %166, %163, %164 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %173 = triton_gpu.view_slice %142[0, 64] [256, 32] [1, 1] : tensor<256x128xf16, #mfma> to tensor<256x32xf16, #mfma>
      %174 = triton_gpu.convert_layout %173 : (tensor<256x32xf16, #mfma>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %175 = triton_gpu.convert_layout %169 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared2>
      %176 = triton_gpu.convert_layout %175 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %177 = tt.dot %174, %171, %172 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %178 = triton_gpu.view_slice %142[0, 96] [256, 32] [1, 1] : tensor<256x128xf16, #mfma> to tensor<256x32xf16, #mfma>
      %179 = triton_gpu.convert_layout %178 : (tensor<256x32xf16, #mfma>) -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %180 = tt.dot %179, %176, %177 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %181 = "tt.reduce"(%141) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %185 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %185 : f32
      }) : (tensor<256x128xf32, #mfma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %182 = arith.addf %137, %181 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %183 = arith.addi %arg25, %c128_i64 : i64
      %184 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %180, %182, %134, %183, %184 : tensor<256x128xf32, #mfma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %97 = tt.extern_elementwise %96#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %98 = arith.addf %96#2, %97 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %99 = triton_gpu.convert_layout %98 : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256xf32, #blocked1>
    %100 = tt.expand_dims %96#1 {axis = 1 : i32} : (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<256x1xf32, #mfma>
    %101 = tt.broadcast %100 : (tensor<256x1xf32, #mfma>) -> tensor<256x128xf32, #mfma>
    %102 = arith.divf %96#0, %101 : tensor<256x128xf32, #mfma>
    %103 = arith.truncf %102 : tensor<256x128xf32, #mfma> to tensor<256x128xf16, #mfma>
    %104 = triton_gpu.convert_layout %103 : (tensor<256x128xf16, #mfma>) -> tensor<256x128xf16, #blocked2>
    tt.store %41, %99 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32, #blocked1>
    %105 = tt.broadcast %47 : (tensor<256x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<256x128x!tt.ptr<f16, 1>, #blocked2>
    %106 = tt.addptr %105, %91 : tensor<256x128x!tt.ptr<f16, 1>, #blocked2>, tensor<256x128xi64, #blocked2>
    tt.store %106, %104 {cache = 1 : i32, evict = 1 : i32} : tensor<256x128xf16, #blocked2>
    tt.return
  }
}"""


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    split_kernel = False
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
        split_kernel = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd" and TORCH_HAS_FP8:
            q = q.to(torch_dtype)
            k = k.to(torch_dtype)
        sm_scale = 1.3
        assert(False)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
            f.write(ir)
            f.flush()
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
            kernel = triton.compile(f.name)

        fn = lambda: kernel(q, k, v, causal, sm_scale, split_kernel)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    for D_HEAD in [128]:
        if mode == 'bwd' and D_HEAD == 128:
            continue
        for causal in [False]:
            if mode == 'bwd' and causal == False:
                continue
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H','N_CTX'],
                x_vals=[#(16, 16, 1024),
                #         (8, 16, 2048),
                #         (4, 16, 4096),
                #         (2, 16, 8192),
                #         (1, 16, 16384),
                #         (4, 48, 1024),
                #         (4, 48, 2048),
                        (4, 48, 4096),
                        # (4, 48, 8192),
                        # (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': torch.float16,
                    'mode': mode,
                    'causal': causal,
                },
            ))



# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
