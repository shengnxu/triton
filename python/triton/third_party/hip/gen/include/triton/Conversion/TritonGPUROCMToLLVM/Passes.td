#ifndef TRITONGPUROCM_CONVERSION_PASSES
#define TRITONGPUROCM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonGPUROCMToLLVM : Pass<"convert-triton-gpurocm-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton_rocm::createConvertTritonGPUROCMToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton_rocm::TritonROCMDialect",
                             "mlir::triton_rocm::gpu_rocm::TritonGPUROCMDialect",
                             "mlir::triton_rocm::nvidia_gpu::TritonNvidiaGPUROCMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"target", "target", "enum Target", "mlir::triton_rocm::Target::Default",
               "compile for target compatible LLVM",
               "llvm::cl::values("
               "clEnumValN(mlir::triton_rocm::Target::NVVM, \"nvvm\", \"compile for "
               "NVVM-compatible LLVM\"), "
               "clEnumValN(mlir::triton_rocm::Target::ROCDL, \"rocdl\", \"compile for "
               "ROCDL-compatible LLVM\"))">,
    ];
}

#endif
