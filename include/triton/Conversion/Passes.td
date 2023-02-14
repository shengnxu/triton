#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPU: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUPass()";

    let dependentDialects = ["mlir::arith::ArithmeticDialect",
                             "mlir::math::MathDialect",
                             "mlir::StandardOpsDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">
   ];
}


def ConvertTritonGPUToRock : Pass<"convert-triton-gpu-to-rock", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to Rock";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToRockPass()";

    let dependentDialects = ["mlir::arith::ArithmeticDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::StandardOpsDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">
    ];
}


def ConvertRockToLLVM : Pass<"convert-rock-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Rock to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertRockToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithmeticDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::StandardOpsDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">
    ];
}

#endif
