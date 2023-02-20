#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToRockPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

namespace {

class ConvertTritonGPUToRock
    : public ConvertTritonGPUToRockBase<ConvertTritonGPUToRock> {

public:
  explicit ConvertTritonGPUToRock(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // Step 1: Decompose unoptimized layout conversions to use shared memory
    // Step 2: Decompose insert_slice_async to use load + insert_slice for
    //   pre-Ampere architectures or unsupported vectorized load sizes

    // Step 1
    decomposeMmaToDotOperand(mod, numWarps);
    decomposeBlockedToDotOperand(mod);

    // Step 2
    if (failed(decomposeInsertSliceAsyncOp(mod)))
      return signalPassFailure();
  }

private:
  int computeCapability{};

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcMma, dstDotOp)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) const {
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  LogicalResult decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return failure();
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis->getPtrContiguity(src);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
#ifndef USE_ROCM
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth))
        return;
#endif

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());

      // insert_slice
      auto axis = insertSliceAsyncOp.getAxis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto index = insertSliceAsyncOp.getIndex();
      if (auto constantOp = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(index.getDefiningOp())) {
          auto integerAttr = constantOp.getValue().dyn_cast_or_null<mlir::IntegerAttr>();
          int64_t intVal = integerAttr.getValue().getSExtValue();
          offsets[axis] = intAttr(intVal);
      }
      //offsets[axis] = insertSliceAsyncOp.getIndex();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.getDst(),
          offsets, sizes, strides);

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
#ifdef USE_ROCM
      assert(decomposed &&
             "AsyncWait is not supported for ROCM and should be removed");
      asyncWaitOp.erase();
#else
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
#endif
    });
    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToRockPass(int computeCapability) {
  return std::make_unique<::ConvertTritonGPUToRock>(computeCapability);
}

} // namespace triton
} // namespace mlir
