"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd + bwd kernel with causal masking
2) Vector and matrix bias (currently fwd kernel only, no causal masking)
3) Any sequence lengths without padding (currently fwd kernel only, no causal masking)
4) fp8 (e5m2fnuz, QK GEMM in fwd kernel only)
5) Support for different sequence lengths for q and k

Not currently supported:

1) Nested / ragged tensors ("varlen")
2) Non power of two head dims

"""

import pytest
import random
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16

TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz

class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = None
    max_seqlens_k = None
    bias_type = None
    causal = False
    num_contexts = 0
    varlen = False

    def __init__(self, sm_scale):
        self.sm_scale = sm_scale
    
    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlens_q = max_seqlens_q
        self.max_seqlens_k = max_seqlens_k
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1

    def need_bias(self, bias_type):
        self.bias_type = bias_type
    
    def need_causal(self):
        self.causal = True

def check_args(q, k, v, o, metadata):
    assert q.dim() == 3
    assert q.dim() == k.dim() and q.dim() == v.dim()
    total_q, nheads_q, head_size = q.shape
    total_k, nheads_k, _ = k.shape
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    # TODO: Change assert if we support qkl f8 and v f16
    assert q.dtype == k.dtype and q.dtype == v.dtype
    # TODO: Fix assert to remove is-power-of-2 check once it is handled
    # TODO: Fix assert to check head size <=256 once supported
    assert head_size <= 128 and ((head_size & (head_size-1)) == 0)
    assert o.shape == q.shape
    if metadata.varlen:
        assert metadata.cu_seqlens_q is not None
        assert metadata.cu_seqlens_k is not None
        assert len(metadata.cu_seqlens_q) == len(metadata.cu_seqlens_k)
        #TODO: Remove once bias is supported with varlen
        assert metadata.bias_type == None
    assert (nheads_q % nheads_k) == 0

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep

# Only needed for testing
def prepare_bias(bias, batch, nheads, seqlen_q, seqlen_k):
    assert bias.is_cuda
    assert bias.dim() == 4
    if bias.shape[2:] == (1, seqlen_k):
        bias_type = "vector"
    elif bias.shape[2:] == (seqlen_q, seqlen_k):
        bias_type = "matrix"
    else:
        raise RuntimeError(
            "Last 2 dimensions of bias must be (1, seqlen)" " or (seqlen, seqlen)"
        )
    return bias.expand(batch, nheads, seqlen_q, seqlen_k), bias_type

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m,
    seqlen_k,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_BLOCK: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr
):
    # range of values handled by this stage
    if STAGE == 1: # "Solid" blocks of Causal masks
        lo, hi = 0, min(seqlen_k, start_m * BLOCK_M)
    elif STAGE == 2: # "Semi-solid", or "Transition" block of Causal mask
        # Must use BLOCK_M, because the starting position of semi-solid block
        # is determined by start_m * BLOCK_M
        lo, hi = start_m * BLOCK_M, min(seqlen_k, start_m * BLOCK_M + BLOCK_M)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
    # So here, we are computing the elements for that last irregular block.
    # In the loop,  we will mask the elements of BLOCK_N that do not exist.
    elif PADDED_BLOCK:
        lo, hi = seqlen_k, seqlen_k + BLOCK_N
        lo = tl.multiple_of(lo, BLOCK_N)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # causal = False
    else:
        lo, hi = 0, seqlen_k
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        if STAGE == 1 or STAGE == 3:
            start_n = tl.multiple_of(start_n, BLOCK_N)
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if PADDED_BLOCK:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            k = tl.load(K_block_ptr)
        if PRE_LOAD_V:
            if PADDED_BLOCK:
                v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
            else:
                v = tl.load(V_block_ptr)
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = OFFS_M[:, None] >= (start_n + OFFS_N[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        if PADDED_BLOCK:
            boundary_m = tl.full([BLOCK_M], TOTAL_TOKENS, dtype=tl.float32)
            size_n = start_n + OFFS_N[None,:]
            mask = size_n < boundary_m[:,None]
            qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # Note about the conflicts of Flash attention algorithm and PyTorch's CUDA implementation
        # PyTorch needs to return softmax(qk) (dropout mask encoded in sign bits)
        # While Flash attention paper compute the dropout AFTER exp2(qk- m_ij)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            if PADDED_BLOCK:
                v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
            else:
                v = tl.load(V_block_ptr)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, bias, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    cu_seqlens_q, cu_seqlens_k,
    H,
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    NEED_PADDING: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    if bias is not None:
        if BIAS_TYPE == "vector":
            bias_ptr = bias + ((off_hz % H) * stride_bh) + offs_n
        elif BIAS_TYPE == "matrix":
            bias_ptr = tl.make_block_ptr(
                base=bias + ((off_hz % H) * stride_bh),
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
    else:
        bias_ptr = None
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_hz * seqlen_q * seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    if STAGE & 1:
        # We don't currently support causal masking and padding.
        tl.static_assert((STAGE != 3) or not NEED_PADDING)
        # equal to N_CTX if N_CTX is already a multiple of block_M
        seqlen_aligned = seqlen_k - extra_tokens_n
        if seqlen_k >= BLOCK_N:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                start_m, seqlen_aligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                False, seqlen_aligned,
                bias_ptr,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX
            )
        tl.debug_barrier()
        if NEED_PADDING:
            if seqlen_k < BLOCK_N:
                seqlen_aligned = 0
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                start_m, seqlen_aligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                True, seqlen_k,
                bias_ptr,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX
            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            2, offs_m, offs_n,
            PRE_LOAD_V,
            False, seqlen_aligned,
            bias_ptr,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_hz * seqlen_q + offs_m
    # Check for last block_M
    overflow_size = (start_m * BLOCK_M) - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), overflow_size, dtype=tl.float32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = off_hz * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))

@triton.jit
def _attn_varlen_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vk, stride_vh, stride_vn,
    stride_om, stride_oh, stride_on,
    cu_seqlens_q_tensor, cu_seqlens_k_tensor,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    max_seqlens_q: tl.constexpr, max_seqlens_k: tl.constexpr,
    B: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
    MQA: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    cu_seqlens_q_start = tl.load(cu_seqlens_q_tensor + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q_tensor + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
    # We have a one-size-fits-all grid in id(0). Some seqlens might be too
    # small for all start_m so for those we return early.
    if start_m * BLOCK_M > seqlen_q:
        return
    cu_seqlens_k_start = tl.load(cu_seqlens_k_tensor + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k_tensor + off_z + 1)
    # cu_seqlens_q has the start m index of each context.
    # off_z is num_contexts.
    q_offset = cu_seqlens_q_start * stride_qm + off_h_q * stride_qh
    # This doesn't exceed bounds as off_z is set to len(cu_seqlens_k) - 1
    seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    need_padding = False
    extra_tokens_n = 0
    if seqlen_k < BLOCK_N:
        need_padding = True
        extra_tokens_n = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        need_padding = True
        extra_tokens_n = seqlen_k % BLOCK_N

    off_h_k = off_h_q % HK if MQA else off_h_q

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = cu_seqlens_k_start * stride_kn + off_h_k * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = cu_seqlens_k_start * stride_vk + off_h_k * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
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
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    # TODO: Fix dropout
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_h_q * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    # TODO: Fix encoded softmax
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    if STAGE & 1:
        # equal to N_CTX if N_CTX is already a multiple of block_M
        seqlen_aligned = seqlen_k - extra_tokens_n
        if seqlen_k >= BLOCK_N:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                start_m, seqlen_aligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                False, seqlen_aligned,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX
            )
        tl.debug_barrier()
        if need_padding:
            if seqlen_k < BLOCK_N:
                seqlen_aligned = 0
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                start_m, seqlen_aligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                True, seqlen_k,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX
            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            2, offs_m, offs_n,
            PRE_LOAD_V,
            False, seqlen_aligned,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_z * HQ * max_seqlens_q + off_h_q * max_seqlens_q + offs_m
    # Check for last block_M
    overflow_size = (start_m * BLOCK_M) - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), overflow_size, dtype=tl.float32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))

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
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr
):
    start_m = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = q_offset
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * seqlen_q
    l_ptrs = L + off_hz * seqlen_q
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(K_block_ptr.type.element_ty)
    vt = tl.load(VT_block_ptr)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = start_m if CAUSAL else 0
    hi = seqlen_q
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_n in range(lo, hi, BLOCK_M):
        offs_m_curr = offs_n[:, None] + start_n
        # -- load q, do --
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        if CAUSAL:
            qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_n * seqlen_k + start_m
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            # CAVEAT: do NOT update p, ds needs the original p
            dv += tl.dot(tl.where(tl.trans(keep), tl.trans(p) / (1 - dropout_p), 0.0).to(Q.dtype.element_ty), do)
        else:
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)#NAN WHY
        dp = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di)
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    # initialize pointers to output
    DK_block_ptr = tl.make_block_ptr(
        base=DK + k_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DK.type.element_ty))


@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * seqlen_q
    l_ptrs = L + off_hz * seqlen_q
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = min(start_m + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, v)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        dq += tl.dot(ds.to(Q.dtype.element_ty), tl.trans(k))
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ_block_ptr.type.element_ty))

empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, bias, sm_scale, dropout_p=0.0, return_encoded_softmax=False, split_kernel=True):
        # shape constraints
        batch, nheads, seqlen, Lq = q.shape
        Lk, Lv = k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        # We assume K sequlen == V seqlen
        assert k.shape[-2] == v.shape[-2]
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]

        # We've derived these previously from tuning the kernel
        BLOCK_M = 256
        BLOCK_N = 128 if Lq == 128 else 64
        waves_per_eu = 2 if Lq == 128 else 3
        num_warps = 8 if Lq == 128 else 4
        pre_load_v = False if Lq == 128 else True
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        # encoded_softmax is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
        # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
        # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
        # only.  This return holds no useful output aside from debugging.
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=torch.float32)
        else:
            encoded_softmax = None

        stage = 3 if causal else 1
        seqlen_k = k.shape[-2]
        if seqlen_k < BLOCK_N:
            need_padding = True
            extra_tokens_n = BLOCK_N - seqlen_k
        elif seqlen_k % BLOCK_N:
            need_padding = True
            extra_tokens_n = seqlen_k % BLOCK_N
        else:
            # We don't care if the M dim needs padding, as we
            # always boundary_check on Q and O
            need_padding = False
            extra_tokens_n = 0

        o = torch.empty_like(q, dtype=v.dtype)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        # Seed the RNG so we get reproducible results for testing.
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        if bias is not None:
            bias, bias_type = prepare_bias(bias, batch, nheads, seqlen_q, seqlen_k)
            bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
        else:
            bias, bias_type, bias_strides = None, None, (0,0,0,0)

        _attn_fwd[grid](
            q, k, v, bias, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            *bias_strides,
            extra_tokens_n,
            H=q.shape[1],
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            STAGE=stage,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N,
            PRE_LOAD_V=pre_load_v,
            NEED_PADDING=need_padding,
            BIAS_TYPE=bias_type,
            ENABLE_DROPOUT=dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=return_encoded_softmax,
            num_stages=1, num_warps=num_warps
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.split_kernel = split_kernel
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = return_encoded_softmax
        return o, encoded_softmax

    @staticmethod
    def backward(ctx, do, _):
        # configuration is not supported
        assert ctx.split_kernel # This implementation only supports split.
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        delta = torch.empty_like(L)
        do_scaled = torch.empty_like(do)
        _attn_bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            do_scaled, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        dq = torch.zeros_like(q)
        _bwd_kernel_dk_dv[(triton.cdiv(q.shape[2], BLOCK), ctx.grid[1])](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            BLOCK_N=BLOCK,
            CAUSAL=ctx.causal,
            ENABLE_DROPOUT=ctx.dropout_p > 0.0,
            num_warps=4,num_stages=1,
        )
        DQ_BLOCK_M = min(seqlen_q, BLOCK)
        _bwd_kernel_dq[ctx.grid](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,
            BLOCK_M=DQ_BLOCK_M,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            BLOCK_N=BLOCK,
            CAUSAL=ctx.causal,
            ENABLE_DROPOUT=ctx.dropout_p > 0.0,
            num_warps=4, waves_per_eu=1, num_stages=1,
        )
        #print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None, None, None, None

attention = _attention.apply

class _attention_varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, o, metadata):
        if o is None:
            o = torch.empty_like(q, dtype=v.dtype)
        check_args(q, k, v, o, metadata)

        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        # We've derived these previously from tuning the kernel
        BLOCK_M = 256
        BLOCK_N = 128 if head_size == 128 else 64
        waves_per_eu = 2 if head_size == 128 else 3
        num_warps = 8 if head_size == 128 else 4
        pre_load_v = False if head_size == 128 else True
        # There is some redundancy here. Not all WGs have useful work to do
        # This is due to the fact that different contexts are asymmetric, while
        # our 3D grid is uniform. If context 0 has seqlen 1024 and that is max
        # seqlen_q, then dim0 of the grid is 8, assuming BLOCK_M=128.
        # If context 1 has seqlen 512, dim0 is still 8 but WGs that belong to 4-7
        # have nothing to do. We handle this in the kernel by detecting this
        # and exiting early.
        grid = (
            triton.cdiv(metadata.max_seqlens_q, BLOCK_M),
            nheads_q,
            metadata.num_contexts
        )

        # TODO: Fix when causal is handled with varlen
        stage = 1

        M = torch.empty((metadata.num_contexts, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)
        # Seed the RNG so we get reproducible results for testing.
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42
        bias = None
        mqa = nheads_q != nheads_k

        #print(f"cu_seqlens_q = {metadata.cu_seqlens_q}")
        #print(f"cu_seqlens_k = {metadata.cu_seqlens_k}")
        #print(f"max_seqlens_q = {metadata.max_seqlens_q}")
        #print(f"max_seqlens_k = {metadata.max_seqlens_k}")
        #print(f"q shape = {q.shape}")
        #print(f"k shape = {k.shape}")
        #print(f"q stride = {q.stride()}")
        #print(f"k stride = {k.stride()}")
        #print(f"grid = {grid}")
        #print(f"mqa = {mqa}")

        _attn_varlen_fwd[grid](
            q, k, v, metadata.sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), 
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2), 
            o.stride(0), o.stride(1), o.stride(2), 
            metadata.cu_seqlens_q, metadata.cu_seqlens_k, 
            dropout_p=0, #dropout_p,
            philox_seed=0, #philox_seed,
            philox_offset_base=None, #philox_offset,
            encoded_softmax=None, #encoded_softmax,
            max_seqlens_q=metadata.max_seqlens_q, 
            max_seqlens_k=metadata.max_seqlens_k,
            B=metadata.num_contexts, HQ=nheads_q, HK=nheads_k,
            MQA=mqa,
            STAGE=stage,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_size, BLOCK_N=BLOCK_N,
            PRE_LOAD_V=pre_load_v,
            ENABLE_DROPOUT=False, #dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=False, #return_encoded_softmax,
            num_stages=1, num_warps=num_warps
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = metadata.causal
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

attention_varlen = _attention_varlen.apply

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 63, 64),
                          (4, 48, 987, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (4, 48, 3989, 64),
                          (4, 48, 1024, 128),
                          (4, 48, 1021, 128),
                          (4, 48, 2048, 128),
                          (4, 48, 4096, 128),
                          (4, 16, 8192, 64),
                          (4, 16, 8080, 64),
                          (1, 48, 16384, 64),
                          (4, 48, 127, 64),
                          ])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('bias_type', ["vector", "matrix"])
@pytest.mark.parametrize('qseqlen_not_equal_kseqlen', [512, None]) #dropout needs to be tested vs SPDA reference in torch
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, use_bias, bias_type, qseqlen_not_equal_kseqlen, dtype=torch.float16):
    torch.manual_seed(20)
    if qseqlen_not_equal_kseqlen is not None:
        seqlen_q = qseqlen_not_equal_kseqlen
    else:
        seqlen_q = N_CTX
    seqlen_k = N_CTX
    if use_bias:
        if bias_type == "vector":
            bias = torch.randn((1, H, 1, seqlen_k), dtype=torch.float32, device="cuda")
        elif bias_type == "matrix":
            bias = torch.randn((1, H, seqlen_q, seqlen_k), dtype=torch.float32, device="cuda")
    else:
        bias = None
    q = torch.randn((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    if TORCH_HAS_FP8E5:
        q = q.to(torch_dtype)
        k = k.to(torch_dtype)
    sm_scale = D_HEAD ** -0.5
    dout = torch.randn_like(q, dtype=torch.float16)
    # triton implementation
    tri_out, _ = attention(q, k, v, causal, bias, sm_scale, 0, False, True) #dropout tested against SDPA only
    # reference implementation
    M = torch.tril(torch.ones((seqlen_q, seqlen_k), device="cuda"))
    p = torch.matmul(q.half(), k.transpose(2, 3).half()) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    if use_bias:
        ref_bias, _ = prepare_bias(bias, Z, H, seqlen_q, seqlen_k)
        p += ref_bias

    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=4e-2, rtol=4e-2)

def varlen_input_helper(Z, HQ, HK, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    
    # Random sequence lengths. Using N_CTX as kind of max of sum of individual seqs
    max_seqlens = N_CTX // Z
    seqlens_q = torch.randint(1, max_seqlens + 1, (Z,), dtype=torch.int32)
    seqlens_k = torch.randint(1, max_seqlens + 1, (Z,), dtype=torch.int32)
    max_seqlens_q = torch.max(seqlens_q).item()
    max_seqlens_k = torch.max(seqlens_k).item()

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")
    # -1 because the last entry of cu_seqlens_q specifies the end of the last seq
    num_ctxs = len(cu_seqlens_q) - 1

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    return q, k, v, input_metadata


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 8192, 64),
                          (4, 48, 256, 64),
                          (4, 48, 512, 64),
                          (4, 48, 1024, 64),
                          (8, 48, 4096, 64),
                          (4, 48, 8192, 64),
                          (4, 48, 128, 128),
                          (4, 48, 4096, 128),
                          (4, 48, 16384, 128),
                          (4, 16, 1024, 128),
                          (4, 16, 8192, 128),
                          (32, 48, 8192, 128)
                          ])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, D_HEAD, dtype)
    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i+1], input_metadata.cu_seqlens_k[i+1]
        q_curr = q[start_q:end_q]
        k_curr = k[start_k:end_k]
        v_curr = v[start_k:end_k]
        scores = torch.einsum('qhd,khd->qhk', q_curr, k_curr) * input_metadata.sm_scale
        p = torch.softmax(scores.float(), dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
    attention_varlen(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD',
                         [(2, 48, 24, 128, 64),
                          (4, 48, 12, 256, 64),
                          (4, 48, 4, 512, 64),
                          (4, 48, 2, 1024, 64),
                          (8, 48, 6, 4096, 64),
                          (4, 48, 24, 8192, 64),
                          (4, 64, 16, 128, 128),
                          (4, 64, 4, 4096, 128),
                          (4, 64, 8, 16384, 128),
                          (4, 16, 4, 1024, 128),
                          (4, 16, 2, 8192, 128),
                          (32, 128, 32, 8192, 128)
                          ])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z, HQ, HK, N_CTX, D_HEAD, causal, dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, D_HEAD, dtype)
    sm_scale = D_HEAD ** -0.5
    tri_out = torch.full_like(q, float("nan"))
    ref_out = torch.full_like(q, float("nan"))
    # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so the
    # size aligns with Q.
    k_ref = k.view(k.shape[0], 1, k.shape[1], k.shape[2]).expand(-1, HQ // HK, -1, -1)
    v_ref = v.view(v.shape[0], 1, v.shape[1], v.shape[2]).expand(-1, HQ // HK, -1, -1)
    #print(f"kref outside = {k_ref[0][1][0][0]}, k = {k[0][1][0]}")
    temp = k_ref.reshape(k_ref.shape[0], -1, k_ref.shape[3])
    #print(f"kref reshaped = {temp[0][1][0]}, kref = {k_ref[0][0][0][0]}")
    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i+1], input_metadata.cu_seqlens_k[i+1]
        q_curr = q[start_q:end_q].permute(1,0,2)
        k_curr = k_ref[start_k:end_k]
        k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3]).permute(1,0,2)
        v_curr = v_ref[start_k:end_k]
        v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3]).permute(1,0,2)
        #print(f"q_curr shape = {q_curr.shape}")
        #print(f"k_curr shape = {k_curr.shape}")
        #print(f"v_curr shape = {v_curr.shape}")
        #if i == 0:
        #    print(f"kref = {k_curr[1][0][0]}, k = {k[0][1][0]}")
        scores = torch.bmm(q_curr, k_curr.transpose(-1, -2)).float() * sm_scale
        p = torch.softmax(scores.float(), dim=-1).half()
        ref_out[start_q:end_q] = torch.bmm(p, v_curr).permute(1,0,2).float()
    attention_varlen(q, k, v, tri_out, input_metadata)
    #print(f"ref shape = {ref_out.shape}, tri shape = {tri_out.shape}")
    #print(f"err = {torch.max(torch.abs(tri_out) - torch.abs(ref_out))}")
    #print(f"triout = {tri_out[0][1][0]}, ref_out = {ref_out[0][1][0]}")
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
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()

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
    tri_out, _ = attention(q, k, v, causal, None, sm_scale, 0, False, True)
    tri_out.backward(dout)#dout)
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

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

configs = []
for mode in ['fwd']:
    for D_HEAD in [128]:
        if mode == 'bwd' and D_HEAD == 128:
            continue
        for causal in [False]:
            if mode == 'bwd' and causal == False:
                continue
            for use_bias in [False, True]:
                configs.append(triton.testing.Benchmark(
                    x_names=['BATCH', 'H','N_CTX'],
                    x_vals=[(16, 16, 1024),
                            (8, 16, 2048),
                            (4, 16, 4096),
                            (2, 16, 8192),
                            (1, 16, 16384),
                            (2, 48, 1024),
                            (2, 48, 2048),
                            (2, 48, 4096),
                            (2, 48, 8192),
                            (2, 48, 16384),
                            (8, 16, 1989),
                            (4, 16, 4097),
                            (2, 16, 8122),
                            (1, 16, 16281),
                            (2, 48, 1021),
                            (2, 48, 2001),
                            (2, 48, 3996),
                            (2, 48, 8181),
                            ],
                    line_arg='provider',
                    line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                    line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                    styles=[('red', '-'), ('blue', '-')],
                    ylabel='ms',
                    plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}-bias={use_bias}',
                    args={
                        'D_HEAD': D_HEAD,
                        'dtype': torch.float16,
                        'mode': mode,
                        'causal': causal,
                        'use_bias' : use_bias})
                )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, D_HEAD, use_bias, causal, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    split_kernel = False
    bias_type = "vector"
    if use_bias:
        if bias_type == "vector":
            bias = torch.randn((1, H, 1, N_CTX), dtype=torch.float32, device="cuda")
        elif bias_type == "matrix":
            bias = torch.randn((1, H, N_CTX, N_CTX), dtype=torch.float32, device="cuda")
        else:
            raise RuntimeError(
                f"Got unsupported bias type: {bias_type}. Supported types are vector and matrix."
            )

    else: bias = None
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
        split_kernel = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd":
            q = q.to(torch_dtype)
            k = k.to(torch_dtype)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, bias, sm_scale, split_kernel)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
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


#bench_flash_attention.run(save_path=".", print_data=True)

configs = []
for mode in ['fwd']:
    for D_HEAD in [128]:
        configs.append(triton.testing.Benchmark(
            x_names=['BATCH', 'HQ', 'HK', 'N_CTX'],
            x_vals=[(16, 16, 4, 1024),
                    #(8, 16, 2, 2048),
                    #(4, 16, 8, 4096),
                    #(2, 16, 4, 8192),
                    #(2, 16, 8, 16384),
                    #(2, 48, 12, 1024),
                    #(2, 48, 24, 2048),
                    #(2, 48, 8, 4096),
                    #(2, 48, 4, 8192),
                    #(2, 48, 2, 16384),
                    #(8, 64, 32, 1024),
                    #(4, 64, 16, 2048),
                    #(4, 64, 8, 4096),
                    #(4, 64, 32, 8192),
                    #(16, 128, 16, 16384),
                    ],
            line_arg='provider',
            line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
            line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
            styles=[('red', '-'), ('blue', '-')],
            ylabel='ms',
            plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}-bias={use_bias}',
            args={
                'D_HEAD': D_HEAD,
                'dtype': torch.float16,
                'mode': mode})
        )


@triton.testing.perf_report(configs)
def bench_varlen_flash_attention(
    BATCH, HQ, HK, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    print("test")

    if provider == "triton":
        q, k, v, input_metadata = varlen_input_helper(BATCH, HQ, HK, N_CTX, D_HEAD, dtype)
        print("test1")
        tri_out = torch.empty_like(q)
        print("test2")
        fn = lambda: attention_varlen(q, k, v, tri_out, input_metadata)
        print("test3")
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print("test4")
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print("test5")
    flops_per_matmul = 0
    print("test6")
    for i in range (0, input_metadata.num_contexts):
        seqlen_q = input_metadata.cu_seqlens_q[i+1] - input_metadata.cu_seqlens_q[i]
        seqlen_k = input_metadata.cu_seqlens_k[i+1] - input_metadata.cu_seqlens_k[i]
        # x2 for 2 GEMMs
        flops_per_matmul += seqlen_q * seqlen_k * HQ * D_HEAD * 2
    # x2 for mul and add
    total_flops = 2 * flops_per_matmul
    print(f"tflops = {total_flops/ms*1e-9}")
    return total_flops / ms * 1e-9

bench_varlen_flash_attention.run(save_path=".", print_data=True)