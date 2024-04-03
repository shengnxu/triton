import unittest
import torch
import triton
import triton.language as tl
import triton.testing

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 8),
    ],
    key = ['D'],
    use_cuda_graph=True,
)
@triton.jit
def _rms_norm_kernel_orig(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_mask: tl.constexpr,
):
    b = tl.program_id(0)
    ## Attention RMS NORM

    _var = float(0.0)
    # _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + D * b + cols, mask=cols < D, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _var += tl.sum(a * a, axis=0)
    rstd = tl.math.rsqrt((_var / D) + eps)
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        a = tl.load(
            x_ptr + D * b + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        tl.store(y_ptr + D * b + cols, a * rstd * w, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 8),
    ],
    key = ['d'],
    use_cuda_graph=True,
)
@triton.jit
def _rms_norm_kernel_const_mask(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_mask: tl.constexpr,
):
    b = tl.program_id(0)

    ## attention rms norm
    _var = float(0.0)
    for offset in range(0, d, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        if use_mask:
            mask = cols < d
            a = tl.load(
                x_ptr + d * b + cols, mask=mask, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
        else:
            a = tl.load(x_ptr + d * b + cols)
        _var += tl.sum(a * a, axis = 0)
    rstd = tl.math.rsqrt((_var / d) + eps)
    for offset in range(0, d, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < d
        if use_mask:
            a = tl.load(
                x_ptr + d * b + cols, mask=mask, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(y_ptr + d * b + cols, a * rstd * w, mask=mask)
        else:
            a = tl.load(x_ptr + d * b + cols).to(tl.float32)
            w = tl.load(w_ptr + cols)
            tl.store(y_ptr + d * b + cols, a * rstd * w)



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 16),
    ],
    key = ['d'],
    use_cuda_graph=True,
)
@triton.jit
def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_mask: tl.constexpr,
):
    b = tl.program_id(0)

    ## attention rms norm
    _var_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, d, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        if use_mask:
            mask = cols < d
            a = tl.load(
                x_ptr + d * b + cols, mask=mask, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
        else:
            a = tl.load(x_ptr + d * b + cols)
        _var_vec += a * a
    _var = tl.sum(_var_vec, axis=0)
    rstd = tl.math.rsqrt((_var / d) + eps)
    for offset in range(0, d, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < d
        if use_mask:
            a = tl.load(
                x_ptr + d * b + cols, mask=mask, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(y_ptr + d * b + cols, a * rstd * w, mask=mask)
        else:
            a = tl.load(x_ptr + d * b + cols).to(tl.float32)
            w = tl.load(w_ptr + cols)
            tl.store(y_ptr + d * b + cols, a * rstd * w)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps = 16),
    ],
    key = ['D'],
    use_cuda_graph=True,
)
@triton.jit
def _rms_norm_kernel_bp(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_mask: tl.constexpr,
):
    b = tl.program_id(0)
    num = tl.num_programs(0)

    pointer = tl.make_block_ptr(base=x_ptr,
                                shape=(num * D),
                                strides=(1),
                                offsets=(b * D),
                                block_shape=(BLOCK_SIZE),
                                order=(0))
    
    w_pointer = tl.make_block_ptr(base=y_ptr,
                                shape=(num * D),
                                strides=(1),
                                offsets=(b * D),
                                block_shape=(BLOCK_SIZE),
                                order=(0))
    
    cols = tl.arange(0, BLOCK_SIZE)
    if use_mask:
        a = tl.load(pointer, boundary_check=(0)).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=cols<D)
    else:
        a = tl.load(pointer).to(tl.float32)
        w = tl.load(w_ptr + cols)

    _var_vec = a * a
    _var = tl.sum(_var_vec, axis = 0)
    rstd = tl.math.rsqrt((_var / D) + eps)
    out = a * rstd * w
    if use_mask:
        tl.store(w_pointer, out.to(tl.bfloat16), boundary_check=(0))
    else:
        tl.store(w_pointer, out.to(tl.bfloat16))


def rms_norm(x, w, y, eps=1.0e-5):
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert y.is_contiguous()
    (B, T, D) = x.shape
    assert w.shape == (D,)
    assert y.shape == x.shape
    block_size = triton.next_power_of_2(D)
    use_mask = D % block_size != 0
    _rms_norm_kernel_bp[(B * T,)](
        x,
        y,
        w,
        eps,
        D,
        # BLOCK_SIZE=block_size,
        use_mask = use_mask,
        # num_warps = 4,
    )
    return y

def test_rms_norm(D, B, T, name):
    x = torch.randn(size=(B, T, D), dtype=torch.bfloat16, device="cuda")
    w = torch.randn(size=(D,), dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)

    def ref_rms_norm(x, w):
        x_std = torch.sqrt(torch.mean(x**2, -1, keepdim=True))
        x_norm = x / (x_std + 1.0e-6)
        return w * x_norm

    # torch.testing.assert_close(ref_rms_norm(x, w), rms_norm(x, w, y))
    t_ms = triton.testing.do_bench(lambda: rms_norm(x, w, y))
    #print(f"best_config = {_rms_norm_kernel_const_mask.get_best_config()}")
    t_seconds = t_ms / 1.0e3
    bandwidth_gbs = x.numel() * x.element_size() * 2 / t_seconds / 1.0e9
    print(f"\n{name}: shape {x.shape}, time {t_seconds * 1.0e6:.2f}us, BW {bandwidth_gbs:.2f}GB/s", end='')

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rms_norm(x, w, y)

    for _ in range(10):
        g.replay()

s = [(8192, 2, 4096),
    (8192, 1, 8192),
    (8192, 2, 1)]
class LLamaTests(unittest.TestCase):

    def test_rms_norm_prefill1(self):
        D, B, T = s[0]
        test_rms_norm(D, B, T, "prefill1")

    def test_rms_normprefill2(self,):
        D, B, T = s[1]
        test_rms_norm(D, B, T, "prefill2")
    
    def test_rms_norm_decode(self):
        D, B, T = s[2]
        test_rms_norm(D, B, T, "decode  ")

if __name__ == "__main__":
    unittest.main()
