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

    @staticmethod
    def forward(cls, q, k, v, scale_float):

        # Transpose in the case of MQA/GQA
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

        BLOCK_M = 16
        BLOCK_N = 64

        M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        o = torch.empty_like(q)
        lse = torch.empty((B * G * H, M), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), B * G * H)

        num_warps = 1

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

# Batch, Mq, Mkv, Hq, Hkv, K
def get_input_shapes():
    cases = [
        # dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=1, K=128)
        (1, 1, 2**i, 1, 1, 128)
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

# vary seq length for fixed head and batch=4
configs = triton.testing.Benchmark(
            x_names=['B', 'Mq','Mkv', 'Hq', 'Hkv', 'K'],
            x_vals=get_input_shapes(),
            line_arg='provider',
            line_vals=['triton'],
            line_names=['Triton'],
            styles=[('red', '-'), ('blue', '-')],
            ylabel='ms',
            plot_name=f'fused-attention',
            args={
                'dtype': torch.float16,
                }
        )

@triton.testing.perf_report(configs)
def bench_flash_attention(B, Mq, Mkv, Hq, Hkv, K, provider, dtype=torch.float16, device="cuda"):
    warmup = 25
    rep = 100
    ms = 0
    q = torch.randn(
        [B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
    ).expand(-1, -1, -1, Hq // Hkv, -1)
    v = torch.randn(
        [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
    ).expand(-1, -1, -1, Hq // Hkv, -1)

    sm_scale = K**-0.5
    fn = lambda: attention(q, k, v, sm_scale)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul =  2 * B * Hq * (Mq * K * Mkv + Mq * Mkv * K)
    total_flops = 2 * flops_per_matmul
    totalBytes = ((B * Mkv * Hkv * K * 2) + (B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2

    # return totalBytes / ms * 1e-9
    return ms * 1000


def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())

