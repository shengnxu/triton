#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "TensorMemRefOpToLLVM.h"

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

struct MemRefToTensorOpConversion 
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::MemRefToTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::MemRefToTensorOp>::ConvertTritonGPUOpToLLVMPattern;
  MemRefToTensorOpConversion(LLVMTypeConverter &converter,
                             AxisInfoAnalysis &axisAnalysisPass, 
                             PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::MemRefToTensorOp>(converter, benefit) {}
  LogicalResult 
  matchAndRewrite(triton::gpu::MemRefToTensorOp op, 
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = src.getType().dyn_cast<MemRefType>();
    auto srcRank = srcTy.getRank();
    auto srcShape = srcTy.getShape();  //!< could be removed. Same shape is guaranteed.
    
    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<RankedTensorType>();
    Type resElemTy = typeConverter->convertType(resTy.getElementType());
    auto resRank = srcTy.getRank();
    auto resShape = resTy.getShape();

    //!< TODO: Need to check the integere of private memory space.
    // unsigned srcAddrSpace = srcTy.getMemorySpaceAsInt();
    // if (addrSpace != PRIVATE_PLACEHOLDER) {
    //   llvm::errs() << "Private address space is needed.\n";
    //   return failure();
    // }

    //! The following check may be redundant because this op ensures the same shape.
    if ((srcRank != resRank) || (srcRank == 0)) {
      llvm::errs() << "Non-scalar and same rank are required\n";
      return failure();
    }

    long numElems = product<long>(resShape);

    MemRefDescriptor desc{adaptor.getSrc()};
    Value srcPtr = desc.alignedPtr(rewriter, loc);

    SmallVector<Value> valVec;

    for (auto i = 0; i < numElems; i++){
      valVec.push_back(extract_val(resElemTy, srcPtr, i));
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = getStructFromElements(loc, valVec, rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

void populateTensorMemRefOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<TensorToMemRefOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MemRefToTensorOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}
