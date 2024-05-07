#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-transpose-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

class ThreadRakeSMEM : public mlir::RewritePattern {
  int kPack;

public:
  explicit ThreadRakeSMEM(mlir::MLIRContext *context, int kPack)
    : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, context),
      kPack(kPack) {}

  mlir::LogicalResult matchAndRewrite(
    mlir::Operation *op,
    mlir::PatternRewriter &rewriter) const override {

    auto dotOp = dyn_cast<triton::DotOp>(op);
    if (!dotOp)
      return mlir::failure();

    // For C = A * B, where A is (M, K) and B is (K, N), the optimal layout for
    // the matrices is K-major.
    // For operand A, optimal would be row-major, i.e. order of (1, 0)
    SmallVector<unsigned int> optimalAOrder = {1, 0};
    // auto opA = dotOp.getA();
    // auto opAEncoding = cast<RankedTensorType>(opA.getType()).getEncoding();
    // auto szPerTdA = ttg::getSizePerThread(opAEncoding);
    // auto orderA = ttg::getOrder(opAEncoding);
    // LDBG("Order A: " << orderA[0] << " " << orderA[1]);
    // LDBG("sizePerThread A: " << szPerTdA[0] << " " << szPerTdA[1]);
    // auto cvtLayoutOpA = cast<ttg::ConvertLayoutOp>(opA.getDefiningOp());
    // auto loadOpA = dyn_cast<tt::LoadOp>(cvtLayoutOpA.getOperand().getDefiningOp());
    // auto opATy = cast<RankedTensorType>(loadOpA.getPtr().getType());
    // opAEncoding = opATy.getEncoding();
    // szPerTdA = ttg::getSizePerThread(opAEncoding);
    // auto ctx = opA.getContext();
    // LDBG("sizePerThread A: " << szPerTdA[0] << " " << szPerTdA[1]);
    auto opA = cast<RankedTensorType>(dotOp.getA().getType());
    auto opAEncoding = opA.getEncoding();
    auto orderA = ttg::getOrder(opA.getEncoding());
    auto szPerTdA = ttg::getSizePerThread(opAEncoding);
    LDBG("kPack: " << kPack);
    LDBG("Order A: " << orderA[0] << " " << orderA[1]);
    LDBG("sizePerThread A: " << szPerTdA[0] << " " << szPerTdA[1]);
    if (orderA != optimalAOrder)
      return failure();

    // For operand B, optimal would be row-major, i.e. order of (1, 0)
    SmallVector<unsigned int> optimalBOrder({0, 1});
    auto opB = dotOp.getB();
    auto opBEncoding = cast<RankedTensorType>(opB.getType()).getEncoding();
    auto szPerTdB = ttg::getSizePerThread(opBEncoding);
    auto orderB = ttg::getOrder(opBEncoding);
    LDBG("Order B: " << orderB[0] << " " << orderB[1]);
    LDBG("sizePerThread B: " << szPerTdB[0] << " " << szPerTdB[1]);
    if (orderB != optimalBOrder) {
      LDBG("B is not optimal");
      // Assume the chain is tt.load -> ttg.convert_layout -> tt.dot
      auto cvtLayoutOpB = cast<ttg::ConvertLayoutOp>(opB.getDefiningOp());
      auto loadOpB = cast<tt::LoadOp>(
        cvtLayoutOpB.getOperand().getDefiningOp());
      auto b = loadOpB.getResult();
      auto opBTy = cast<RankedTensorType>(b.getType());
      opBEncoding = opBTy.getEncoding();
      szPerTdB = ttg::getSizePerThread(opBEncoding);
      auto ctx = b.getContext();
      LDBG("sizePerThread B: " << szPerTdB[0] << " " << szPerTdB[1]);
      SmallVector<unsigned int> newSzPerTdB{szPerTdB[0], 2};
      auto newBlkEncodingB = ttg::BlockedEncodingAttr::get(
        ctx, newSzPerTdB, ttg::getThreadsPerWarp(opBEncoding),
        ttg::getWarpsPerCTA(opBEncoding), orderB,
        ttg::getCTALayout(opBEncoding));
      auto newBType = RankedTensorType::get(
        opBTy.getShape(), opBTy.getElementType(), newBlkEncodingB);
      b = rewriter.create<ttg::ConvertLayoutOp>(loadOpB.getLoc(), newBType, loadOpB.getResult());
      szPerTdB = ttg::getSizePerThread(newBlkEncodingB);
      LDBG("New sizePerThread B: " << szPerTdB[0] << " " << szPerTdB[1]);
      return success();
    }
    return success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUTransposeCoalescePass
    : public TritonAMDGPUTransposeCoalesceBase<
          TritonAMDGPUTransposeCoalescePass> {
public:
  TritonAMDGPUTransposeCoalescePass() = default;
  TritonAMDGPUTransposeCoalescePass(int kPack) {
    this->kPack = kPack;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ThreadRakeSMEM>(context, kPack);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUTransposeCoalescePass(
    int kPack) {
  return std::make_unique<TritonAMDGPUTransposeCoalescePass>(kPack);
}