#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "TensorMemRefOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::LDSEncodingAttr;

struct TensorToMemRefOpConversion 
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::TensorToMemRefOp> {
  using ConvertTritonGPUOpToLLVMPattern<triton::gpu::TensorToMemRefOp>::ConvertTritonGPUOpToLLVMPattern;
  LogicalResult 
  matchAndRewrite(triton::gpu::TensorToMemRefOp op, 
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    Attribute srcEncoding = srcTy.getEncoding();
    if (!(srcEncoding && srcEncoding.isa<LDSEncodingAttr>())) {
      return failure();
    }

    Value smemBase = getSharedMemoryBase(loc, rewriter, src);

    Value llSrc = adaptor.getSrc();

    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<MemRefType>();
    //!< maybe need it later.
    //!< auto memSpaceAttr = resTy.getMemorySpace().dyn_cast<mlir::gpu::AddressSpaceAttr>();


    auto elemTy = resTy.getElementType();
    Value ptr = bitcast(smemBase, 
		        getTypeConverter()->getPointerType(
			  elemTy,  
	                  *getTypeConverter()->getMemRefAddressSpace(resTy)
			  //!< may be need it later. memSpaceAttr.getValue())
			) 
		       );

    MemRefDescriptor desc = MemRefDescriptor::fromStaticShape(rewriter, loc,
        	                                              *getTypeConverter(), resTy,
        						      ptr);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

struct MemRefToTensorOpConversion 
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::MemRefToTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::MemRefToTensorOp>::ConvertTritonGPUOpToLLVMPattern;
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

    // auto gpuAttr = srcTy.getMemorySpace().dyn_cast<mlir::gpu::AddressSpaceAttr>();
    // if (gpuAttr.getValue() != mlir::gpu::GPUDialect::getPrivateAddressSpace()) {
    //   llvm::errs() << "private address space is required.\n";
    //   return failure();
    // }

    unsigned memSpace = srcTy.getMemorySpaceAsInt();
    if (memSpace != 5){
      llvm::errs() <<"Wrong memory space\n";
      return failure();
    }
    
    
    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<RankedTensorType>();
    Type resElemTy = typeConverter->convertType(resTy.getElementType());
    auto resRank = resTy.getRank();

    // memreftype rank == 1
    if (( resRank == 0) || (srcRank == 0)) {
      llvm::errs() << "Non-scalar rank is required\n";
      return failure();
    }

    unsigned numElems = getElemsPerThread(resTy);
    unsigned srcNumElems = product<long>(srcShape) * srcElemSize;

    MemRefDescriptor desc{adaptor.getSrc()};
    Value srcPtr = desc.alignedPtr(rewriter, loc);

    SmallVector<Value> valVec; 

    // memref<mxvector<nxf32>>
    for (auto i = 0; i < srcShape[0]; i++){
      Value ptr = gep(srcPtr.getType(), llSrcElemTy, srcPtr, i32_val(i));
      Value vec = load(llSrcElemTy, ptr);
      // Value vec = ptr[i]; //!<  tried to make it vector or array-like.
      for (auto j = 0; j < srcElemSize; j++){ 
        valVec.push_back(extract_element(resElemTy, vec, i32_val(j)));
      }
    }

    llvm::outs()<<"retrieved value from vector\n";

    Value resultStruct = getTypeConverter()->packLLElements(loc, valVec, rewriter, resTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

void populateTensorMemRefOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<TensorToMemRefOpConversion>(typeConverter, allocation, smem, indexCacheInfo, benefit);
  patterns.add<MemRefToTensorOpConversion>(typeConverter, allocation, smem, indexCacheInfo, benefit);
}
