#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static std::string getKPackAttrName() { return "triton_gpu.kpack"; }
    static std::string getMPerWaveAttrName() { return "triton_gpu.mPerWave"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return mod->getAttr("triton_gpu.num-warps").cast<IntegerAttr>().getInt();
    }
    static int getKPack(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.kpack"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.kpack attribute");
      return mod->getAttr("triton_gpu.kpack").cast<IntegerAttr>().getInt();
    }
    static int getMPerWave(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.mPerWave"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.mPerWave attribute");
      return mod->getAttr("triton_gpu.mPerWave").cast<IntegerAttr>().getInt();
    }
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif
