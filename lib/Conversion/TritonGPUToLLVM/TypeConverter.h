#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

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
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping);

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type getElementTypeForStruct(RankedTensorType type);
  Type convertTritonPointerType(triton::PointerType type);

  Value packLLElements(Location loc, ValueRange resultVals,
                       ConversionPatternRewriter &rewriter, Type type);

  SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                      ConversionPatternRewriter &rewriter,
                                      Type type);

  Type convertTritonTensorType(RankedTensorType type);
};

#endif
