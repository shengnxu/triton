#ifndef TRITON_CONVERSION_TRITONGPU_TO_ROCK_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_ROCK_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

#define GEN_PASS_DECL_CONVERTTRITONGPUTOROCK
#include "triton/Conversion/Passes.h.inc"

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToRockPass(int computeCapability = 80);

} // namespace triton

} // namespace mlir

#endif
