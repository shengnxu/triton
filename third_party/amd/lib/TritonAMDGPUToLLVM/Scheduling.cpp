#include "TritonAMDGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_SCHEDULEOPS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
void insertIgplIntrinsics(LLVM::LLVMFuncOp func, int mode) {
  OpBuilder builder(func);
  RewriterBase::InsertionGuard guard(builder);

  auto loc = func.getLoc();
  auto intrinsicName =
      StringAttr::get(func.getContext(), "llvm.amdgcn.iglp.opt");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = builder.getI32Type();

  DenseSet<Block *> blocks;
  func->walk([&](Operation *op) {
    RegisteredOperationName name = op->getName().getRegisteredInfo().value();
    if (name.getStringRef().contains("rocdl.mfma")) {
      Block *block = op->getBlock();
      if (!blocks.contains(block)) {
        builder.setInsertionPoint(op);
        auto option = builder.create<LLVM::ConstantOp>(
            loc, builder.getIntegerAttr(i32, mode));
        builder.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                              ValueRange{option}, defaultFlags);

        blocks.insert(block);
      }
    }
  });
}
} // namespace

namespace {
struct ScheduleOps : public triton::impl::ScheduleOpsBase<ScheduleOps> {
  ScheduleOps(int mode)
      : mode(mode), triton::impl::ScheduleOpsBase<ScheduleOps>(){};
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mod->walk([this](LLVM::LLVMFuncOp func) {
      insertIgplIntrinsics(func, this->mode);
    });
  }

private:
  int mode;
};
} // namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createSchedulingOpsPass(int mode) {
  return std::make_unique<ScheduleOps>(mode);
}
} // namespace triton
} // namespace mlir
