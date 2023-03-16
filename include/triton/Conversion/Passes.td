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
              "number of warps">
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

//===----------------------------------------------------------------------===//
// RockToGPU
//===----------------------------------------------------------------------===//

def ConvertRockToGPUPass : Pass<"convert-rock-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower the operations from the Rock dialect into the GPU "
                "dialect";

  let options = [
      Option<"kernelName", "kernel-name", "std::string",
             "\"rock_conv2d_kcyx_nchw_nkhw\"",
             "kernel name to be lowered">,
      Option<"gpuModuleName", "gpu-module-name", "std::string",
             "\"rock_kernel_module\"",
             "GPU kernel module name to be lowered">,
  ];

  let dependentDialects = ["mlir::rock::RockDialect",
                           "mlir::amdgpu::AMDGPUDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::memref::MemRefDialect"];
}


def TritonConvertArithToIndex : Pass<"triton-convert-arith-to-index", "mlir::ModuleOp"> {

    let summary = "Convert arith to index";
    
    let constructor = "mlir::triton::createTritonConvertArithToIndexPass()";

    let description = [{
      Convert arith operation on index values to corresponding ops in the index dialect.
      We need this because SCFToCF conversion currently generates arith ops on indices.
    }];

    let dependentDialects = ["mlir::index::IndexDialect"];
}

#endif
