#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;
  if (isa<triton::gpu::DotOperandEncodingAttr>(cvt.getType().getEncoding()))
    return true;
  return false;
}

static Operation *gatherDFG(Operation *op, SmallVector<Operation *> &ops) {
  Operation *lastOp = nullptr;
  // auto getLast = [&lastOp](Operation *op) {
  //   if (!lastOp || op->getIndex() > lastOp->getIndex()) {
  //     lastOp = op;
  //   }
  // };
  // BFS (filo)
  Block *block = op->getBlock();
  SmallVector<Operation *> oprs;
  for (auto operand : op->getOperands()) {
    if (Operation *pop = operand.getDefiningOp()) {
      if (pop->getBlock() == block) {
        // must reside in same block
        oprs.push_back(pop);
        ops.push_back(pop);
      // } else {
      //   getLast(pop);
      }
    }
  }
  for (auto *op : oprs) {
    gatherDFG(op, ops);
  }
  return lastOp;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if (std::distance(user_begin, user_end) != 1) // hasOneUse??
        return;
      if (user_begin->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, *user_begin});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();
    // Move LocalLoadOp and LocalAllocOp immediately after their operands.
    m.walk([&](Operation *op) {
      if (!isa<triton::gpu::LocalLoadOp, triton::gpu::LocalAllocOp>(op) ||
          op->getNumOperands() < 1) {
        return;
      }
      Operation *argOp = op->getOperand(0).getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move global loads early (prefetch)
    m.walk([&](triton::LoadOp op) {
      // 0. gather cone
      SmallVector<Operation *> ops{op};
      gatherDFG(op, ops);
      Block *block = op->getBlock();
      {
        // 1. move right after last dep
        for (auto *op : ops)
          op->moveAfter(block, block->begin());
      }
    });
    return;
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
