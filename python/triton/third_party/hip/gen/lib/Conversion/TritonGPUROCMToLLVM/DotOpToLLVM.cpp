#include "DotOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton_rocm;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton_rocm::gpu_rocm::DotOperandEncodingAttr;
using ::mlir::triton_rocm::gpu_rocm::getShapePerCTA;
using ::mlir::triton_rocm::gpu_rocm::MmaEncodingAttr;

LogicalResult convertFMADot(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                            TritonGPUROCMToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA884(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                            TritonGPUROCMToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA1688(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                             TritonGPUROCMToLLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

LogicalResult convertMMA16816(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                              TritonGPUROCMToLLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

#if 1
LogicalResult convertMFMA(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                          TritonGPUROCMToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
#endif
LogicalResult convertWGMMA(triton_rocm::DotOp op, triton_rocm::DotOp::Adaptor adaptor,
                           TritonGPUROCMToLLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

LogicalResult convertAsyncWGMMA(triton_rocm::nvidia_gpu::DotAsyncOp op,
                                triton_rocm::nvidia_gpu::DotAsyncOp::Adaptor adaptor,
                                TritonGPUROCMToLLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                Value thread);

struct DotOpConversion : public ConvertTritonGPUROCMOpToLLVMPattern<triton_rocm::DotOp> {
  using ConvertTritonGPUROCMOpToLLVMPattern<
      triton_rocm::DotOp>::ConvertTritonGPUROCMOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.isVolta())
        return convertMMA884(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isTuring())
        return convertMMA1688(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isAmpere())
        return convertMMA16816(op, adaptor, getTypeConverter(), rewriter);
      if (mmaLayout.isHopper())
        return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                            getThreadId(rewriter, loc));

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

#if 1
    MfmaEncodingAttr mfmaLayout = D.getType()
                                      .cast<RankedTensorType>()
                                      .getEncoding()
                                      .dyn_cast<MfmaEncodingAttr>();
    if (!isOuter && mfmaLayout && supportMFMA(op)) {
      return convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }
#endif

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct DotAsyncOpConversion
    : public ConvertTritonGPUROCMOpToLLVMPattern<triton_rocm::nvidia_gpu::DotAsyncOp> {
  using ConvertTritonGPUROCMOpToLLVMPattern<
      triton_rocm::nvidia_gpu::DotAsyncOp>::ConvertTritonGPUROCMOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::nvidia_gpu::DotAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout &&
        supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      if (mmaLayout.isHopper()) {
        return convertAsyncWGMMA(op, adaptor, getTypeConverter(), rewriter,
                                 getThreadId(rewriter, loc));
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotAsyncOp to LLVM.");
    }

    llvm::report_fatal_error(
        "Unsupported DotAsyncOp found when converting TritonGPU to LLVM.");
  }
};

struct DotWaitOpConversion
    : public ConvertTritonGPUROCMOpToLLVMPattern<triton_rocm::nvidia_gpu::DotWaitOp> {
  using ConvertTritonGPUROCMOpToLLVMPattern<
      triton_rocm::nvidia_gpu::DotWaitOp>::ConvertTritonGPUROCMOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::nvidia_gpu::DotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() <= 1) {
      Value intput =
          adaptor.getInputs().size() == 1 ? adaptor.getInputs()[0] : Value();
      rewriter.replaceOpWithNewOp<triton_rocm::nvgpu::WGMMAWaitGroupOp>(op, intput,
                                                                   pendings);
      return success();
    }
    std::vector<Type> types;
    // Pack the inputs into a single struct.
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().dyn_cast<LLVM::LLVMStructType>();
      if (!structType)
        return failure();
      for (Type type : structType.getBody())
        types.push_back(type);
    }
    auto packedType =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    unsigned outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().dyn_cast<LLVM::LLVMStructType>();
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], input, i);
        packed = rewriter.create<LLVM::InsertValueOp>(
            loc, packedType, packed, value, outputStructIndex++);
      }
    }
    Value packedOutput =
        rewriter.create<triton_rocm::nvgpu::WGMMAWaitGroupOp>(loc, packed, pendings);
    // Unpack the output into the original struct types.
    SmallVector<Value> outputs;
    outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().cast<LLVM::LLVMStructType>();
      Value unpacked = rewriter.create<LLVM::UndefOp>(loc, structType);
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, packedType.getBody()[outputStructIndex], packedOutput,
            outputStructIndex);
        outputStructIndex++;
        unpacked = rewriter.create<LLVM::InsertValueOp>(loc, structType,
                                                        unpacked, value, i);
      }
      outputs.push_back(unpacked);
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

void populateDotOpToLLVMPatterns(TritonGPUROCMToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 ModuleAllocation &allocation,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, benefit);
  patterns.add<DotAsyncOpConversion>(typeConverter, allocation, benefit);
  patterns.add<DotWaitOpConversion>(typeConverter, allocation, benefit);
}
