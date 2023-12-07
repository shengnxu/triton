#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"

using llvm::MapVector;
using namespace mlir;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct DotSlicingPass : public TritonGPUDotSlicingBase<DotSlicingPass> {
  DotSlicingPass() = default;

  void runOnOperation() override {
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUDotSlicingPass() {
  return std::make_unique<DotSlicingPass>();
}
