import torch
from triton.runtime import driver
from triton.ops import matmul_perf_model
import amdsmi
from triton.testing import (get_dram_gbps, get_max_simd_tflops, get_max_tensorcore_tflops)


if __name__ == "__main__":
    device = torch.cuda.current_device()
    prop = driver.active.utils.get_device_properties(device)
    print(prop)
    amdsmi.amdsmi_init()
    d = amdsmi.amdsmi_get_processor_handles()[0]
    clock_info = amdsmi.amdsmi_get_clock_info(d, amdsmi.AmdSmiClkType.GFX)
    print(clock_info)
    sm_clk_max = matmul_perf_model.get_clock_rate_in_khz()
    print(f"DRAM: {get_dram_gbps()} GB/s")
    print(f"MFMA max throughput: {get_max_tensorcore_tflops(torch.bfloat16, sm_clk_max)} TFlops")