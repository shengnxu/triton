#include "RegReallocOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton_rocm;

struct RegAllocOpConversion
    : public ConvertTritonGPUROCMOpToLLVMPattern<triton_rocm::nvidia_gpu::RegAllocOp> {
  using ConvertTritonGPUROCMOpToLLVMPattern<
      triton_rocm::nvidia_gpu::RegAllocOp>::ConvertTritonGPUROCMOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::nvidia_gpu::RegAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton_rocm::nvgpu::RegAllocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};

struct RegDeallocOpConversion
    : public ConvertTritonGPUROCMOpToLLVMPattern<triton_rocm::nvidia_gpu::RegDeallocOp> {
  using ConvertTritonGPUROCMOpToLLVMPattern<
      triton_rocm::nvidia_gpu::RegDeallocOp>::ConvertTritonGPUROCMOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::nvidia_gpu::RegDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton_rocm::nvgpu::RegDeallocOp>(
        op, adaptor.getRegCount());
    return success();
  }
};

void populateRegReallocOpToLLVMPatterns(
    TritonGPUROCMToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const ModuleAllocation &allocation, PatternBenefit benefit) {
  patterns.add<RegAllocOpConversion>(typeConverter, benefit);
  patterns.add<RegDeallocOpConversion>(typeConverter, benefit);
  return;
}
