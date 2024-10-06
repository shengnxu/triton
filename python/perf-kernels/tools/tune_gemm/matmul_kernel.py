import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, P, locks, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_bias, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr,
                  EVEN_K: tl.constexpr, GRID_MN: tl.constexpr, NUM_XCDS: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = GRID_MN // NUM_XCDS
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        pid = xcd * pids_per_xcd + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)
    if BIAS:
        c += bias[:, None]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        if pid_z == 0:
            for iter in range(1, SPLIT_K):
                loc_index = pid * SPLIT_K + iter
           #     while tl.atomic_cas(locks + pid*iter, 0, 1) != 0:
                while tl.load(locks + loc_index, cache_modifier = ".cv", volatile=True) != 1:
                    pass
                offs_am1 = tl.arange(0, BLOCK_SIZE_M)
                offs_bn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(offs_am1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(offs_bn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + loc_index * BLOCK_SIZE_M * BLOCK_SIZE_N + offs_am1[:, None] * BLOCK_SIZE_N + offs_bn1[None, :]
        #        accumulator += tl.load(tl.multiple_of(P_, (1, 16)))
                accumulator += tl.load(tl.multiple_of(P_, (1, 16)), cache_modifier = '.cv')

            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
            mask = (offs_am < M)[:, None] & (offs_bn < N)[None, :]
            tl.store(C_, accumulator, mask=mask)
        else:
            loc_index = pid * SPLIT_K + pid_z
            offs_am1 = tl.arange(0, BLOCK_SIZE_M)
            offs_bn1 = tl.arange(0, BLOCK_SIZE_N)
            offs_am1 = tl.max_contiguous(tl.multiple_of(offs_am1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn1 = tl.max_contiguous(tl.multiple_of(offs_bn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + loc_index * BLOCK_SIZE_M * BLOCK_SIZE_N +  offs_am1[:, None] * BLOCK_SIZE_N + offs_bn1[None, :]
            tl.store(P_, accumulator, cache_modifier=".wt")
            tl.store(locks + loc_index, 1, cache_modifier=".wt")
         #   tl.store(P_, accumulator)
         #   tl.debug_barrier()
         #   tl.atomic_xchg(locks + pid * pid_z, 1)
