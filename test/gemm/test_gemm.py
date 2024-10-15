from gemm_a8w8 import gemm_a8w8_forward
import torch
import triton
import triton.language as tl
import re
import pytest


def get_shapes():
    shapes = [(512 * i, 512 * i, 512 * i) for i in range(1, 16)]
    return shapes


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8e4': tl.float8e4b8,
    'fp8e5': tl.float8e5b16,
}

def gen_input(M, N, ty_name, needTrans, seed, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if ty_name == 'int8':
        if needTrans:
            raw_data = torch.randint(-20, 20, (N, M), dtype=torch.int8, device='cuda').T
        else:
            raw_data = torch.randint(-20, 20, (M, N), dtype=torch.int8, device='cuda')

        return raw_data, raw_data.to(torch.half)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    # avoid type conversion rounding errors of subnormal values
    raw_data += 0.1
    if d_type == tl.float8e4b8:
        raw_data += torch.sign(raw_data)

    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
        (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1]

def num_tensors(M, N, K):
    size = M * N + M * K + N * K + M + N
    total_size = 512 * 1024 * 1024
    num = triton.cdiv(total_size, size)
    return num 


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=get_shapes(),
        line_arg='provider',
        line_vals=['triton(int8)', 'torch(int8)'],
        line_names=['Triton.int8', "Torch.int8"],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        args={},
        plot_name='gemm-a8w8',
    )
)
def benchmark(M, N, K, provider):
    in_dtype = get_type(provider)
    out_dtype = torch.half

    a, _ = gen_input(M, K, in_dtype, False, 1, device='cuda')
    b, _ = gen_input(K, N, in_dtype, True, 2, device='cuda')
    alpha_row = torch.rand([M, 1], dtype=torch.half).cuda()
    alpha_col = torch.rand([1, N], dtype=torch.half).cuda()
    out = torch.empty([M, N], dtype=out_dtype, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
 
    if 'torch' in provider:
        gemm_a8w8 = TorchGemmA8W8()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_a8w8(a, b, alpha_row, alpha_col), rep=100, quantiles=quantiles
        )
    else: 
        assert 'triton' in provider
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_a8w8_forward(out, a, b, alpha_row, alpha_col), rep=100, quantiles=quantiles
        )
    perf_us = lambda x: round(x * 1e3, 2)
    #perf_us = lambda x: round(2 * M * N * K / x * 1e-9, 2)
    return perf_us(ms), perf_us(min_ms), perf_us(max_ms)
 

class TorchGemmA8W8(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, a, b, alpha_row, alpha_col):
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        scale = torch.matmul(alpha_row, alpha_col)
        out = torch.mul(x, scale)
        return out.to(torch.half)


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)

 
@pytest.mark.parametrize('m, n, k', get_shapes())
def test_gemm_a8w8(m, n, k):
    torch.random.manual_seed(0)
    with torch.no_grad():
        a, _ = gen_input(m, k, 'int8', False, 1, device='cuda')
        b, _ = gen_input(k, n, 'int8', True, 2, device='cuda')

        alpha_row = torch.rand([m, 1], dtype=torch.half).cuda()
        alpha_col = torch.rand([1, n], dtype=torch.half).cuda()

        gemm_a8w8 = TorchGemmA8W8()
        out_torch = gemm_a8w8(a, b, alpha_row=alpha_row, alpha_col=alpha_col)
        out_triton = torch.empty([a.shape[0], b.shape[1]], dtype=torch.half, device=a.device)
        gemm_a8w8_forward(out_triton, a, b, alpha_row, alpha_col)

        assert torch.allclose(out_triton.half(), out_torch.half(), rtol=1e-2)
