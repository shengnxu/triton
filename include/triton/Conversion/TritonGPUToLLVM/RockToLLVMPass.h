#ifndef TRITON_CONVERSION_ROCK_TO_LLVM_PASS_H
#define TRITON_CONVERSION_ROCK_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

#define GEN_PASS_DECL_CONVERTROCKTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertRockToLLVMPass(int computeCapability = 80);

} // namespace triton

} // namespace mlir

#endif
