//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPUROCM_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONGPUROCM_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonGPUROCMTypeConverter : public TypeConverter {
public:
  TritonGPUROCMTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                         int numCTAs);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

class TritonGPUROCMConversionTarget : public ConversionTarget {

public:
  explicit TritonGPUROCMConversionTarget(MLIRContext &ctx,
                                     TritonGPUROCMTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
