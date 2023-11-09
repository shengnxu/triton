#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Utility.h"

using namespace mlir;

namespace {

// convert(val) : mma -> blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> mma
// convert(mask) : blocked -> mma
// tt.store(ptr, val, mask, ...) : mma
//
// Store with mma layout directly
class BypassEpilogueSMEM : public mlir::RewritePattern {

public:
  explicit BypassEpilogueSMEM(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton_rocm::StoreOp::getOperationName(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto stOp = dyn_cast<triton_rocm::StoreOp>(op);
    if (!stOp)
      return mlir::failure();
    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = ptr.getType().dyn_cast<RankedTensorType>();
    auto valType = val.getType().dyn_cast<RankedTensorType>();
    if (!ptrType || !valType ||
        !ptrType.getEncoding().isa<triton_rocm::gpu_rocm::BlockedEncodingAttr>() ||
        !valType.getEncoding().isa<triton_rocm::gpu_rocm::BlockedEncodingAttr>())
      return mlir::failure();

    auto cvtOp = dyn_cast<triton_rocm::gpu_rocm::ConvertLayoutOp>(val.getDefiningOp());
    if (!cvtOp)
      return mlir::failure();

    auto encoding =
        cvtOp.getSrc().getType().cast<RankedTensorType>().getEncoding();

#if 1
    if (!encoding.isa<triton_rocm::gpu_rocm::MfmaEncodingAttr>())
      return mlir::failure();
#else
    if (!encoding.isa<triton_rocm::gpu_rocm::MmaEncodingAttr>())
      return mlir::failure();
#endif

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto newEncoding =
        cvtOp.getOperand().getType().cast<RankedTensorType>().getEncoding();

    auto newVal = cvtOp.getOperand();

    auto newPtrType = RankedTensorType::get(
        ptrType.getShape(), ptrType.getElementType(), newEncoding);
    Value newPtr = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(
        ptr.getLoc(), newPtrType, ptr);

    Value newMask = mask;
    if (mask) {
      auto maskType = mask.getType().dyn_cast<RankedTensorType>();
      auto newMaskType = RankedTensorType::get(
          maskType.getShape(), maskType.getElementType(), newEncoding);
      newMask = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(
          mask.getLoc(), newMaskType, mask);
    }

    rewriter.replaceOpWithNewOp<triton_rocm::StoreOp>(
        stOp, newPtr, newVal, newMask, stOp.getCache(), stOp.getEvict());
    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h.inc"

class TritonGPUROCMOptimizeEpiloguePass
    : public TritonGPUROCMOptimizeEpilogueBase<TritonGPUROCMOptimizeEpiloguePass> {

public:
  TritonGPUROCMOptimizeEpiloguePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUROCMOptimizeEpiloguePass() {
  return std::make_unique<TritonGPUROCMOptimizeEpiloguePass>();
}
