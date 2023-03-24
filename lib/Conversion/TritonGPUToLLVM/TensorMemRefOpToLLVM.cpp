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

    Value smemBase = getSharedMemoryBase(loc, rewriter, src);
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

// Copied from GpuOpsLowering.h
/// A function that maps a MemorySpace enum to a target-specific integer value.
using MemorySpaceMapping =
    std::function<unsigned(mlir::gpu::AddressSpace gpuAddressSpace)>;

/// Copied from GpuOpsLowering.cpp
static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

// Copied from GpuOpsLowering.cpp
void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type,
                mlir::gpu::AddressSpaceAttr memorySpaceAttr) {
        mlir::gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

void populateTensorMemRefOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  // Copied from LowerGpuOpsToROCDLOps.cpp
  // We need this function to teach the typeConverter how to lower the
  // enum-style gpu memory space into integers. Otherwise, fromStaticShape
  // will complain.
  populateGpuMemorySpaceAttributeConversions(
      typeConverter, [](mlir::gpu::AddressSpace space) {
        switch (space) {
        case mlir::gpu::AddressSpace::Global:
          return 1;
        case mlir::gpu::AddressSpace::Workgroup:
          return 3;
        case mlir::gpu::AddressSpace::Private:
          return 5;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });
  patterns.add<TensorToMemRefOpConversion>(typeConverter, allocation, smem,
                                           indexCacheInfo, benefit);
  patterns.add<MemRefToTensorOpConversion>(typeConverter, allocation, smem,
                                           indexCacheInfo, benefit);
}
