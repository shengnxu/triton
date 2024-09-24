import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=16, num_stages=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, input_row_stride, output_row_stride, n_rows, n_cols, eps,
               BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_idx = row_start

    #Calculate squared mean by block
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row_sum = 0.0
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        row_block = row_block * row_block  #square every value the block
        row_sum += (tl.sum(row_block, axis=-1) / n_cols
                    )  #tl.sum across row, divide by block_size and add it running sum

    row_norm = row_sum + eps
    row_norm = tl.rsqrt(row_norm)

    #Blocked normalization
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")  #load block of input
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".cg")  #load block of g
        output = row_block * row_norm  #element wise multiply with rms_norm
        output = output * g  #element wise multiplication with g

        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, output, mask=mask)


def rmsnorm(x, epsilon=1e-6):
    n_rows, n_cols = x.shape
    #Restricting BLOCK_SIZE to 64Kb is an important optimization. Otherwise,
    #performance can drop significantly for larger n_cols.
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    y = torch.empty_like(x, device='cuda')
    g = torch.ones((1, n_cols), device='cuda')

    num_programs = n_rows
    grid = lambda meta: (num_programs, )
    rms_kernel[grid](y, x, g, x.stride(0), y.stride(0), n_rows, n_cols, epsilon, BLOCK_SIZE)

    return y


def run_rmsnorm(M, N):
    print(f"Running RMSNorm for shape ({M}, {N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = rmsnorm(x)

    return y_triton


@pytest.mark.parametrize('M, N', [(1, 4), (2, 10), (8192, 4096), (4096, 8192), (1, 8192), (873, 1245), (1, 98304)])
def test_rmsnorm(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = rmsnorm(x)

    rms_norm = torch.nn.RMSNorm(N, device='cuda')
    y_torch = rms_norm(x)

    print(f"y_triton={y_triton}")
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def torch_rmsnorm(x):
    M, N = x.shape
    rms_norm = torch.nn.RMSNorm(N, device='cuda')
    y_torch = rms_norm(x)

    return y_torch


def run_benchmark(args):
    config = []
    if (args.M_benchmark):
        val = args.M_start
        x_vals_list = []
        while val <= args.M_end:
            x_vals_list.append(val)
            val *= args.M_step
        mn_args = {'N': args.N_start}
        plot_name = str("rmsnorm-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    else:
        x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
        mn_args = {'M': args.M_start}
        x_names = ['N']
        plot_name = str("rmsnorm-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))

    dtype = arg_to_torch_dtype[args.dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_rmsnorm(x))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: rmsnorm(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RMSNorm",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="8192", type=int)
    parser.add_argument('-Ns', "--N_step", default="1024", type=int)
    parser.add_argument('-Ne', "--N_end", default="32768", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_benchmark:
        run_rmsnorm(args.M_start, args.N_start)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
