#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "TensorMemRefOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::LDSEncodingAttr;

struct TensorToMemRefOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::TensorToMemRefOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::TensorToMemRefOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::TensorToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    Attribute srcEncoding = srcTy.getEncoding();
    if (!(srcEncoding && srcEncoding.isa<LDSEncodingAttr>()))
      return failure();

    Value smemBase;
    if (allocation->getBufferId(src) == Allocation::InvalidBufferId) {
      auto smemObj =
          getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
      smemBase = smemObj.base;
    } else {
      smemBase = getSharedMemoryBase(loc, rewriter, src);
    }
    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<MemRefType>();

    MemRefDescriptor desc = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), resTy, smemBase, smemBase);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

struct MemRefToTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::MemRefToTensorOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::MemRefToTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemRefToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = src.getType().dyn_cast<MemRefType>();
    auto srcRank = srcTy.getRank();
    auto srcShape = srcTy.getShape();
    auto srcElemTy = srcTy.getElementType().dyn_cast<VectorType>();
    auto vecElemTy = srcElemTy.getElementType();
    Type llSrcElemTy = typeConverter->convertType(srcElemTy);
    auto srcElemSize = srcElemTy.getNumElements();

    auto gpuAttr =
        srcTy.getMemorySpace().dyn_cast<mlir::gpu::AddressSpaceAttr>();
    if (gpuAttr.getValue() != mlir::gpu::GPUDialect::getPrivateAddressSpace()) {
      llvm::errs() << "private address space is required.\n";
      return failure();
    }

    Value res = op.getResult();
    auto resTy = res.getType().dyn_cast<RankedTensorType>();
    Type resElemTy = typeConverter->convertType(resTy.getElementType());
    auto resRank = resTy.getRank();

    if ((resRank == 0) || (srcRank == 0)) {
      llvm::errs() << "Non-scalar rank is required\n";
      return failure();
    }

    unsigned numElems = srcTy.getNumElements();

    MemRefDescriptor desc{adaptor.getSrc()};
    Value srcPtr = desc.alignedPtr(rewriter, loc);

    // bitcast the llvm.ptr<5> to llvm.ptr<f32, 5>
    srcPtr = bitcast(srcPtr, ptr_ty(vecElemTy, 5));
    auto vecElemPtrTy = ptr_ty(vecElemTy, 5);

    SmallVector<Value> valVec;

    for (auto i = 0; i < numElems * srcElemSize; i++) {
      Value elemPtr = gep(vecElemPtrTy, srcPtr, i32_val(i));
      Value elem = load(elemPtr);
      valVec.push_back(elem);
    }
    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = getTypeConverter()->packLLElements(
        loc, valVec, rewriter, llvmResultStructTy);
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
  patterns.add<TensorToMemRefOpConversion>(typeConverter, allocation, smem,
                                           indexCacheInfo, benefit);
  patterns.add<MemRefToTensorOpConversion>(typeConverter, allocation, smem,
                                           indexCacheInfo, benefit);
}
