#!/usr/bin/env python
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple
import pytest
import torch
import sys

import triton
import triton.language as tl

def _strides(x: torch.Tensor, *stride_names: str):
    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


# @triton.autotune(
#    configs=[
#     #    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'waves_per_eu': 2}, num_stages=1, num_warps=2),
#     #    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1, num_warps=2),
#     #    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_stages=1, num_warps=4),
#        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_stages=1, num_warps=2),
#        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_stages=1, num_warps=2),
#        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
#    ],
#    key=['Z', 'H', 'G', 'N_CTX_Q', 'N_CTX_K', 'BLOCK_DMODEL'],        
# )
@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out,
    lse,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_om,
    stride_og,
    stride_oh,
    stride_on,
    Z,
    N_CTX_Q,
    N_CTX_K,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G

    kv_len = N_CTX_K

    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
        shape=(N_CTX_Q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
    # Additional shift by 1 along the last dimension in the quantized case, since
    # the first element along that dim contains packed quantization coefficients.
    K_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(BLOCK_DMODEL, kv_len),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg
    V_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(kv_len, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(
        tl.advance(Q_block_ptr, (0, 0))
    )

    # loop over k, v and update accumulator
    for start_n in range(0, kv_len, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # for i in range(elem_num):  # noqa: F821
        qk += tl.dot(q, k)  # noqa: F821
        qk *= qk_scale

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + off_h * stride_oh + off_z * stride_oz + off_g * stride_og,
        shape=(N_CTX_Q, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(
        tl.advance(O_block_ptr, (0, 0)),
        (acc / l_i[:, None]).to(Out.type.element_ty),
        boundary_check=(0,),
    )

    l_ptrs = lse + start_m * BLOCK_M + tl.arange(0,BLOCK_M)
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

class _attention(torch.autograd.Function):

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 128
    # SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
    #     type(None),
    #     BlockDiagonalCausalWithOffsetPaddedKeysMask,
    # }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "triton_splitKF"

    @staticmethod
    def forward(cls, q, k, v, scale_float):

        cls.SPLIT_K: Optional[int] = None
        cls.BLOCK_M = 16
        cls.BLOCK_N = 64

        # Transpose in the case of MQA/GQA
        #print(f"q shape = {q.shape}")
        #print(f"k shape = {k.shape}")
        #print(f"v shape = {v.shape}")
        #print(f"q stride = {q.stride()}")
        #print(f"k stride = {k.stride()}")
        #print(f"v stride = {v.stride()}")
        mqa_swap_seqlen_head = False
        if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
            mqa_swap_seqlen_head = True
            assert q.shape[1] == 1
            q = q.transpose(1, 3)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]

        B, Mk, G, H, Kkv = k.shape
        B, M, G, H, Kq = q.shape
        Lk = k.shape[-1]
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N

        M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        o = torch.empty_like(q)
        lse = torch.empty((B * G * H, M), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), B * G * H)

        num_warps = 1
        #print(f"q shape after = {q.shape}")
        #print(f"k shape after = {k.shape}")
        #print(f"v shape after = {v.shape}")
        #print(f"q stride after = {q.stride()}")
        #print(f"k stride after = {k.stride()}")
        #print(f"v stride after = {v.stride()}")

        _fwd_kernel_splitK[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=scale_float,
            Out=o,
            lse=lse,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o, "oz", "om", "og", "oh", "on"),
            Z=B,
            H=H,
            G=G,
            N_CTX_Q=M,
            N_CTX_K=Mk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=1,
        )

        if mqa_swap_seqlen_head:
            o.transpose(1, 3)
        o = o.reshape(B, H, -1, Kq).transpose(1, 2).contiguous()
        return o

attention = _attention.apply

def get_input_shapes():
    cases = [
        # dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=1, K=128)
        (1, 1, 2**i, 2048, 1, 128)
        for i in [11] # Mkv = 2048 and 8192
    ]# + [
    #    # dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=2, K=128)
    #    (max(1, 2 ** (16 - i)), 1, 2**i, 16, 2, 128)
    #    for i in range(8, 18)
    #]

    # cases = [
    #     # dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=1, K=128)
    #     (max(1, 2 ** (16 - i)), 1, 2**i, 16, 1, 128)
    #     for i in range(17, 18)
    # ]

    return cases


@pytest.mark.parametrize('B, Mq, Mkv, Hq, Hkv, K',
                         get_input_shapes())
def test_op_fwd(B, Mq, Mkv, Hq, Hkv, K, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((B, Mq, Hkv, Hq // Hkv, K), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    ).expand(-1, -1, -1, Hq // Hkv, -1)
    v = (
        torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    ).expand(-1, -1, -1, Hq // Hkv, -1)
    scale = 1 / K**0.5
    tri_out = attention(q, k, v, scale)

    q = q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
    k = k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    v = v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-3, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    # for D_HEAD in [128]:
    for causal in [False]:
        configs.append(triton.testing.Benchmark(
            x_names=['B', 'Mq','Mkv', 'Hq', 'Hkv', 'K'],
            x_vals=get_input_shapes(),
            line_arg='provider',
            line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
            line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
            styles=[('red', '-'), ('blue', '-')],
            ylabel='ms',
            plot_name=f'fused-attention-d{128}-{mode}-causal={causal}',
            args={
                # 'D_HEAD': D_HEAD,
                'dtype': torch.float16,
                'mode': mode,
                'causal': causal})
        )


@triton.testing.perf_report(configs)
def bench_flash_attention(B, Mq, Mkv, Hq, Hkv, K, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 100
    rep = 400
    ms = 0
    if provider == "triton":
        q = torch.randn(
            [B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=False
        )
        k = torch.randn(
            [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
        ).expand(-1, -1, -1, Hq // Hkv, -1)
        v = torch.randn(
            [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
        ).expand(-1, -1, -1, Hq // Hkv, -1)

        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # if provider == "flash":
    #     qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    #     if FLASH_VER == 1:
    #         lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
    #         cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
    #         cu_seqlens[1:] = lengths.cumsum(0)
    #         qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
    #         fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
    #     elif FLASH_VER == 2:
    #         fn = lambda: flash_attn_func(qkv, causal=causal)
    #     else:
    #         raise ValueError(f'unknown {FLASH_VER = }')
    #     ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul =  2 * B * Hq * (Mq * K * Mkv + Mq * Mkv * K)
    total_flops = 2 * flops_per_matmul
    totalBytes = ((B * Mkv * Hkv * K * 2) + (B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2

    # return totalBytes / ms * 1e-9
    return ms * 1000


def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())

