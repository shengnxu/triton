#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPU: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,
       Option<"kpack", "kpack",
              "int32_t", /*default*/"4",
              "rock tuning parameter">,
       Option<"mPerWave", "mPerWave",
              "int32_t", /*default*/"16",
              "rock tuning parameter">
   ];
}


def ConvertTritonGPUToRock : Pass<"convert-triton-gpu-to-rock", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to Rock";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToRockPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::rock::RockDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">
    ];
}


def ConvertRockToLLVM : Pass<"convert-rock-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Rock To LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertRockToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">
    ];
}

#endif
