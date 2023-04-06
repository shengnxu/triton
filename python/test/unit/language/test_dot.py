# flake8: noqa: F821,F841
import itertools
import os
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']

# ---------------
# test dot
# ---------------


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)

def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")

# Restrictions of supported dim:
# 1. M and N must be a power of 2, which is a restriction from triton
#    https://github.com/openai/triton/blob/7f3f58f3322d537125c6f6a18d50f070d643994b/include/triton/Dialect/Triton/IR/TritonOps.td#L19
# 2. num_warps must be a power of 2, which is a restriction from triton
#    https://github.com/openai/triton/blob/7f3f58f3322d537125c6f6a18d50f070d643994b/python/triton/runtime/jit.py#L291
# 3. The largest MxN is 128x64 or 64x128 due to limitation of LDS size
# 4. When M and N is fixed, the largest K can be found in the following list.
#    For example, when M=128, N=64, the largest supported K is 64.
#    This is also due to limited LDS size
# 5. The smallest M or N dim is 16. This is due to the mfma instruction
#    selection logic in rocMLIR. Now we only select 32x32 or 16x16 mfma
#    output matrix as the building block for larger matrices.
# 6. The smallest K dim is 16, which is a restriction from triton
#    https://github.com/openai/triton/blob/7f3f58f3322d537125c6f6a18d50f070d643994b/python/triton/language/semantic.py#L1182
@pytest.mark.parametrize("M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, dtype",
                         [(*shape_nw, col_a, col_b, epilogue, allow_tf32, dtype)
                          for shape_nw in [[128, 64, 16, 16], [64, 128, 16, 16],
                                           [128, 64, 16, 8], [64, 128, 16, 8],
                                           [128, 64, 16, 4], [64, 128, 16, 4],
                                           [128, 64, 16, 2], [64, 128, 16, 2],
                                           [128, 64, 16, 1], [64, 128, 16, 1],
                                           [128, 64, 32, 16], [64, 128, 32, 16],
                                           [128, 64, 32, 8], [64, 128, 32, 8],
                                           [128, 64, 32, 4], [64, 128, 32, 4],
                                           [128, 64, 32, 2], [64, 128, 32, 2],
                                           [128, 64, 32, 1], [64, 128, 32, 1],
                                           [128, 64, 64, 16], [64, 128, 64, 16],
                                           [128, 64, 64, 8], [64, 128, 64, 8],
                                           [128, 64, 64, 4], [64, 128, 64, 4],
                                           [128, 64, 64, 2], [64, 128, 64, 2],
                                           [128, 64, 64, 1], [64, 128, 64, 1],
                                           [128, 32, 16, 8], [32, 128, 16, 8],
                                           [128, 32, 16, 4], [32, 128, 16, 4],
                                           [128, 32, 16, 2], [32, 128, 16, 2],
                                           [128, 32, 16, 1], [32, 128, 16, 1],
                                           [128, 32, 32, 8], [32, 128, 32, 8],
                                           [128, 32, 32, 4], [32, 128, 32, 4],
                                           [128, 32, 32, 2], [32, 128, 32, 2],
                                           [128, 32, 32, 1], [32, 128, 32, 1],
                                           [128, 32, 64, 8], [32, 128, 64, 8],
                                           [128, 32, 64, 4], [32, 128, 64, 4],
                                           [128, 32, 64, 2], [32, 128, 64, 2],
                                           [128, 32, 64, 1], [32, 128, 64, 1],
                                           [128, 16, 16, 4], [16, 128, 16, 4],
                                           [128, 16, 16, 2], [16, 128, 16, 2],
                                           [128, 16, 16, 1], [16, 128, 16, 1],
                                           [128, 16, 32, 4], [16, 128, 32, 4],
                                           [128, 16, 32, 2], [16, 128, 32, 2],
                                           [128, 16, 32, 1], [16, 128, 32, 1],
                                           [128, 16, 64, 4], [16, 128, 64, 4],
                                           [128, 16, 64, 2], [16, 128, 64, 2],
                                           [128, 16, 64, 1], [16, 128, 64, 1],
                                           [64, 64, 16, 8],
                                           [64, 64, 16, 4],
                                           [64, 64, 16, 2],
                                           [64, 64, 16, 1],
                                           [64, 64, 32, 8],
                                           [64, 64, 32, 4],
                                           [64, 64, 32, 2],
                                           [64, 64, 32, 1],
                                           [64, 64, 64, 8],
                                           [64, 64, 64, 4],
                                           [64, 64, 64, 2],
                                           [64, 64, 64, 1],
                                           [64, 64, 128, 8],
                                           [64, 64, 128, 4],
                                           [64, 64, 128, 2],
                                           [64, 64, 128, 1],
                                           [64, 32, 16, 4], [32, 64, 16, 4],
                                           [64, 32, 16, 2], [32, 64, 16, 2],
                                           [64, 32, 16, 1], [32, 64, 16, 1],
                                           [64, 32, 32, 4], [32, 64, 32, 4],
                                           [64, 32, 32, 2], [32, 64, 32, 2],
                                           [64, 32, 32, 1], [32, 64, 32, 1],
                                           [64, 32, 64, 4], [32, 64, 64, 4],
                                           [64, 32, 64, 2], [32, 64, 64, 2],
                                           [64, 32, 64, 1], [32, 64, 64, 1],
                                           [64, 32, 128, 4], [32, 64, 128, 4],
                                           [64, 32, 128, 2], [32, 64, 128, 2],
                                           [64, 32, 128, 1], [32, 64, 128, 1],
                                           [64, 16, 16, 2], [16, 64, 16, 2],
                                           [64, 16, 16, 1], [16, 64, 16, 1],
                                           [64, 16, 32, 2], [16, 64, 32, 2],
                                           [64, 16, 32, 1], [16, 64, 32, 1],
                                           [64, 16, 64, 2], [16, 64, 64, 2],
                                           [64, 16, 64, 1], [16, 64, 64, 1],
                                           [64, 16, 128, 2], [16, 64, 128, 2],
                                           [64, 16, 128, 1], [16, 64, 128, 1],
                                           [32, 32, 32, 1],
                                           [32, 32, 64, 1],
                                           [32, 32, 128, 1],
                                           [32, 32, 256, 1],
                                           [32, 16, 16, 1], [16, 32, 16, 1],
                                           [32, 16, 32, 1], [16, 32, 32, 1],
                                           [32, 16, 64, 1], [16, 32, 64, 1],
                                           [32, 16, 128, 1], [16, 32, 128, 1],
                                           [32, 16, 256, 1], [16, 32, 256, 1],
                                           [16, 16, 16, 1],
                                           [16, 16, 32, 1],
                                           [16, 16, 64, 1],
                                           [16, 16, 128, 1],
                                           [16, 16, 256, 1],
                                           [16, 16, 512, 1]]
                          for epilogue in ['none']
                          for allow_tf32 in [True]
                          for col_a in [True, False]
                          for col_b in [True, False]
                          for dtype in ['int8', 'float16', 'float32']] +
                         # The following configs use too much LDS for float32
                         [(*shape_nw, col_a, col_b, epilogue, allow_tf32, dtype)
                          for shape_nw in [[128, 64, 128, 8], [64, 128, 128, 8],
                                           [128, 64, 128, 4], [64, 128, 128, 4],
                                           [128, 64, 128, 2], [64, 128, 128, 2],
                                           [128, 64, 128, 1], [64, 128, 128, 1],
                                           [128, 32, 128, 4], [32, 128, 128, 4],
                                           [128, 32, 128, 2], [32, 128, 128, 2],
                                           [128, 32, 128, 1], [32, 128, 128, 1],
                                           [128, 16, 128, 4], [16, 128, 128, 4],
                                           [128, 16, 128, 2], [16, 128, 128, 2],
                                           [128, 16, 128, 1], [16, 128, 128, 1],
                                           [64, 64, 256, 4],
                                           [64, 64, 256, 2],
                                           [64, 64, 256, 1],
                                           [64, 32, 256, 2], [32, 64, 256, 2],
                                           [64, 32, 256, 1], [32, 64, 256, 1],
                                           [64, 16, 256, 2], [16, 64, 256, 2],
                                           [64, 16, 256, 1], [16, 64, 256, 1],
                                           [32, 32, 512, 1],
                                           [32, 16, 512, 1], [16, 32, 512, 1],
                                           [16, 16, 1024, 1]]
                          for epilogue in ['none']
                          for allow_tf32 in [True]
                          for col_a in [True, False]
                          for col_b in [True, False]
                          for dtype in ['int8', 'float16']])
def test_dot(M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, dtype, device='cuda'):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8:
        if dtype == 'int8':
            pytest.skip("Only test int8 on devices with sm >= 80")
        elif dtype == 'float32' and allow_tf32:
            pytest.skip("Only test tf32 on devices with sm >= 80")
    if capability[0] == 7:
        if (M, N, K, num_warps) == (128, 256, 32, 8):
            pytest.skip("shared memory out of resource")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               COL_A: tl.constexpr, COL_B: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(Ws)
            z = tl.dot(z.to(w.dtype), w)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    if col_a:
        x = numpy_random((K, M), dtype_str=dtype, rs=rs).T
    else:
        x = numpy_random((M, K), dtype_str=dtype, rs=rs)
    if col_b:
        y = numpy_random((N, K), dtype_str=dtype, rs=rs).T
    else:
        y = numpy_random((K, N), dtype_str=dtype, rs=rs)
    w = numpy_random((N, N), dtype_str=dtype, rs=rs)
    #print(x)
    if 'int' not in dtype:
        x *= .1
        y *= .1
    if dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    #print(x)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    # triton result
    if dtype == 'int8':
        z = 1 + numpy_random((M, N), dtype_str='int32', rs=rs)
    else:
        z = 1 + numpy_random((M, N), dtype_str=dtype, rs=rs) * .1

    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         COL_A=col_a, COL_B=col_b,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps)
    # torch result
    if dtype == 'int8':
        z_ref = np.matmul(x.astype(np.float32),
                          y.astype(np.float32())).astype(np.int32)
    else:
        z_ref = np.matmul(x, y)

    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        z_ref = np.matmul(z_ref, w)
    # compare
    if dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-4)
