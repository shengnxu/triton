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
    auto srcShape = srcTy.getShape();
    auto srcElemTy = srcTy.getElementType().dyn_cast<VectorType>();
    Type llSrcElemTy = typeConverter->convertType(srcElemTy);
    auto srcElemSize  = srcElemTy.getNumElements();

    auto gpuAttr = srcTy.getMemorySpace().dyn_cast<mlir::gpu::AddressSpaceAttr>();
    if (gpuAttr.getValue() != mlir::gpu::GPUDialect::getPrivateAddressSpace()) {
      llvm::errs() << "private address space is required.\n";
      return failure();
    }
    
    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<RankedTensorType>();
    Type resElemTy = typeConverter->convertType(resTy.getElementType());
    auto resRank = resTy.getRank();

    if (( resRank == 0) || (srcRank == 0)) {
      llvm::errs() << "Non-scalar rank is required\n";
      return failure();
    }

    unsigned numElems = getElemsPerThread(resTy);
    auto srcNumElems = product<long>(srcShape);
    if (numElems != srcNumElems) {
      llvm::errs() << "The number of elements is not consistent.\n";
      return failure();
    }

    MemRefDescriptor desc{adaptor.getSrc()};
    Value srcPtr = desc.alignedPtr(rewriter, loc);

    SmallVector<Value> valVec;

    for (auto i = 0; i < numElems; i++){
      Value vec = extract_val(llSrcElemTy, srcPtr, i);
      for (auto j = 0; j < srcElemSize; j++){
        valVec.push_back(extract_val(resElemTy, vec, j));
      }
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
