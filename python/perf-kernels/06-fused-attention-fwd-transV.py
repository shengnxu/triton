"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import pytest
import torch
import sys

import triton
import triton.language as tl

from enum import Enum

# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
# float8:tl.constexpr = torch.float8_e4m3fnuz

USE_DTYPE = 2
class Dtype:
    float16 = 1
    float8_e5m2fnuz = 2
    float8_e4m3fnuz = 3
# set intended dtype and scaling factor if fp8
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
if USE_DTYPE == Dtype.float16 or (not TORCH_HAS_FP8E5 and not TORCH_HAS_FP8E4):
    torch_dtype:tl.constexpr = torch.float16
    fp8_scale = 1.0
elif USE_DTYPE == Dtype.float8_e5m2fnuz:
    float8:tl.constexpr = torch.float8_e5m2fnuz
    fp8_scale = 57344.0
elif USE_DTYPE == Dtype.float8_e4m3fnuz:
    float8:tl.constexpr = torch.float8_e4m3fnuz
    fp8_scale = 448.0
else:
    sys.exit("Specified datatype is not supported")

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _attn_fwd(
    # Q, K, V, sm_scale, M, Out,
    Q, K, V, sm_scale, q_scale, k_scale, v_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    use_fp8: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(0, 1)
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
    q = tl.load(Q_block_ptr)
    # it's even better to multiply the qk_scale and convert to f16
    # than doing it inside the loop
    # So conversion is quite cheap
    # if use_fp8, attention_scale (qk_scale) should be done after 1st gemm
    if not use_fp8:
        q = (q * qk_scale).to(q.dtype)
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        ##### Scale + Dot + Descale #####
        # Scale
        if use_fp8:
            k = (k * k_scale).to(k.dtype)
        if pre_load_v:
            v = tl.load(V_block_ptr)
            if use_fp8:
                v = (v * v_scale).to(v.dtype)
        # Dot
        qk += tl.dot(q, k)
        # Descale
        if use_fp8:  # descale qk
            qk = qk * qk_scale / q_scale / k_scale
        #################################

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # TODO: here is amax_s for next iteration
        # we calculate for current iter for 1st one but for next iter
        # use reduce kernel or atomic (entire tensor rather than each group)
        if use_fp8:
            s_scale0 = 1.0
            s_scaleN = 1.0
            if start_n == lo:
                s_scale0 = fp8_scale / tl.max(tl.max(p, 1), 0)
            else:
                s_scaleN = fp8_scale / tl.max(tl.max(p, 1), 0)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
            if use_fp8:
                v = (v * v_scale).to(v.dtype)

        ##### Scale + Dot + Descale #####
        # Scale
        if use_fp8:  # scale p and v
            s_scale = s_scale0 if start_n == lo else s_scaleN
            p = (p * s_scale).to(v.dtype)
            v = (v * v_scale).to(v.dtype)
        # Dot
        acc += tl.dot(p, v)
        # Descale
        if use_fp8:  # descale p & acc
            s_scale = s_scale0 if start_n == lo else s_scaleN
            p = p / s_scale
            acc = acc / s_scale / v_scale
        #################################

        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    acc = acc / l_i[:, None]
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, q_scale, k_scale, v_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        ## hardcoded best perf_configs for MI250
        if Lk == 64:
            ## D_HEAD = 64
            BLOCK_M = 128
            BLOCK_N = 64
            waves_per_eu = 3
            num_warps = 4
            num_stages = 1
            ## causal=False likes to pre load v but causal=True does not
            pre_load_v = False if causal else True
            use_fp8 = USE_DTYPE != 1
        else:
            ## D_HEAD = 128
            ## For fp16, pick BLOCK_M=256, num_warps=8
            ## For fp8, pick BLOCK_M=128, num_warps=4
            ## TODO (zhanglx): add tuning infra for FA
            BLOCK_M = 128 if q.dtype == float8 else 256
            BLOCK_N = 128
            waves_per_eu = 2
            num_warps = BLOCK_M // 32
            num_stages = 1
            pre_load_v = False
            use_fp8 = USE_DTYPE != 1

        grid = ( triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            q, k, v, sm_scale, q_scale, k_scale, v_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N,
            waves_per_eu = waves_per_eu,
            num_warps = num_warps,
            num_stages = num_stages,
            pre_load_v = pre_load_v,
            use_fp8=use_fp8,
        )

        return o


attention = _attention.apply

name_to_torch_types = {
    'fp16': torch.float16,
    'fp8': float8
}

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(4, 48, 1024, 128),
                  (4, 48, 2048, 128),
                  (4, 48, 4096, 128)]
    for dtype in ['fp8']])
def test_op_fwd(Z, H, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, D_HEAD, N_CTX), dtype=torch.float16, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    dout = torch.randn_like(q, dtype=torch.float16)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q.half(), k.transpose(2, 3).half()) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v.transpose(2,3))
    # triton implementation
    q_scale = fp8_scale * (q.float().amax()).item()
    k_scale = fp8_scale * (k.float().amax()).item()
    v_scale = fp8_scale * (v.float().amax()).item()
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    v = v.to(name_to_torch_types[dtype])
    tri_out = attention(q, k, v, sm_scale, q_scale, k_scale, v_scale)
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 0
    torch.testing.assert_close(ref_out, tri_out.to(torch.float16), atol=atol, rtol=0)


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
for dtype in [float8]:
    for D_HEAD in [128]:
        for causal in [False]:
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H','N_CTX'],
                x_vals=[(16, 16, 1024),
                        # (8, 16, 2048),
                        # (4, 16, 4096),
                        # (2, 16, 8192),
                        # (1, 16, 16384),
                        # (4, 48, 1024),
                        # (4, 48, 2048),
                        # (4, 48, 4096),
                        # (4, 48, 8192),
                        # (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'],
                line_names=['Triton'],
                #styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-fwd-d{D_HEAD}-causal={causal}-{dtype}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': dtype,
                    'causal': causal})
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, provider, dtype, device="cuda"):
    warmup = 25
    rep = 100
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=torch.float16, device="cuda", requires_grad=True)
    # use .item() because only python float, rather than pytorch float tensor can be recoginized as fp32 in triton
    q_scale = fp8_scale / (q.float().amax()).item()
    k_scale = fp8_scale / (k.float().amax()).item()
    v_scale = fp8_scale / (v.float().amax()).item()
    sm_scale = 1.3
    # TODO: scale qkv here first and then type casting
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    fn = lambda: attention(q, k, v, sm_scale, q_scale, k_scale, v_scale)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    return total_flops / ms * 1e-9


def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())
