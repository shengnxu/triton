import triton
import triton.language as tl
import torch


@triton.jit
def kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, W, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ADD_MATRIX: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
    Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
    Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
    x = tl.load(Xs)
    y = tl.load(Ys)
    z = tl.dot(x, y)
    tl.store(Zs, z)


M = 128
N = 128
K = 128

x = torch.zeros((M, K), dtype=torch.float16, device="cuda")
for i in range(M):
    x[i, i] = 1
y = torch.zeros((K, N), dtype=torch.float16, device="cuda")
for i in range(K):
    for j in range(N):
        y[i, j] = i + j * K
z = torch.zeros((M, N), dtype=torch.float32, device="cuda")

kernel[(1, 1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0), z.stride(1),
                  enable_moe_lds_bypass=False)

ref = torch.matmul(x, y)

assert_allclose(ref, z)
