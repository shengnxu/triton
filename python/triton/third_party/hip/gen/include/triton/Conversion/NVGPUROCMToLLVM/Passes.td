#ifndef NVGPUROCM_CONVERSION_PASSES
#define NVGPUROCM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertNVGPUROCMToLLVM : Pass<"convert-nv-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert NVGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton_rocm::createConvertNVGPUROCMToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton_rocm::nvgpu::NVGPUROCMDialect"];
}

#endif
