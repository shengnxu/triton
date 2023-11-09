#ifndef TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_ROCM_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypesROCM.h"

using namespace mlir;
using namespace mlir::triton_rocm;

class TritonGPUROCMToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUROCMToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type getElementTypeForStruct(RankedTensorType type);
  Type convertTritonPointerType(triton_rocm::PointerType type);

  Value packLLElements(Location loc, ValueRange resultVals,
                       ConversionPatternRewriter &rewriter, Type type);

  SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                      ConversionPatternRewriter &rewriter,
                                      Type type);

  Type convertTritonTensorType(RankedTensorType type);

  SmallVector<Value> packMfmaOperand(
    const SmallVector<Value> &inValues, Type srcTy,
    ConversionPatternRewriter &rewriter, Location loc);
};

#endif
