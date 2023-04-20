"""
Reduce Sum
====================
In this tutorial, you will write a high-performance reduce sum kernel.

In doing so, you will learn about:
- Implementing parallel reduction in Triton
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _reduce_sum(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride
    # Compute sum for each block
    sum = 0
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _sum += a
    sum = tl.sum(_sum, axis=0)
    tl.store(Y + row, sum)

def reduce_sum(x: torch.Tensor):
    M, N = x.shape
    x_type = x.dtype
    y = torch.empty((M, ), dtype=x_type, device='cuda')

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    if torch.version.hip is not None:
        num_warps = 2
        if BLOCK_SIZE >= 2048:
            num_warps = 4
        if BLOCK_SIZE >= 4096:
            num_warps = 8
        if BLOCK_SIZE >= 8192:
            num_warps = 16
    else:
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _reduce_sum[(M,)](x, y, N, N, BLOCK_SIZE, num_warps=num_warps)
    return y

def test_reduce_sum(M, N, dtype, device = 'cuda'):
    x_shape = (M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device = device)
    y_tri = reduce_sum(x)
    y_torch = torch.sum(x, 1)
    triton.testing.assert_almost_equal(y_tri, y_torch)
    return

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='reduce-sum',
        args={'M': 4096, 'dtype': torch.float16}
    )
)
def bench_reduce_sum(M, N, dtype, provider, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    # utility functions
    if provider == 'triton':
        y_fwd = lambda: reduce_sum(x)
    if provider == 'torch':
        y_fwd = lambda: torch.sum(x, 1)
    data_size = x.numel() * x.element_size() + M * x.element_size()
    gbps = lambda ms: data_size / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_reduce_sum(4096, 1024, torch.float16)
# test_reduce_sum(4096, 8192, torch.float16)
# bench_reduce_sum.run(save_path='.', print_data=True)

