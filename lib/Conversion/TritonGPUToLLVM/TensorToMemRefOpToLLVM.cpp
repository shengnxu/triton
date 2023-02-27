#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "TensorToMemRefOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::LDSEncodingAttr;

struct TensorToMemRefOpConversion 
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::TensorToMemRefOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::TensorToMemRefOp>::ConvertTritonGPUOpToLLVMPattern;

  TensorToMemRefOpConversion(LLVMTypeConverter &converter,
                             AxisInfoAnalysis &axisAnalysisPass, 
                             PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::TensorToMemRefOp>(converter, benefit) {}
  LogicalResult 
  matchAndRewrite(triton::gpu::TensorToMemRefOp op, 
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    Attribute srcEncoding = srcTy.getEncoding();
    if (!(srcEncoding && srcEncoding.isa<LDSEncodingAttr>())) return failure();

    Value smemBase = getSharedMemoryBase(loc, rewriter, src);

    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<MemRefType>();

    MemRefDescriptor desc = MemRefDescriptor::fromStaticShape(rewriter, loc,
		                                              *getTypeConverter(), resTy,
					                      smemBase, 
							      smemBase);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};
