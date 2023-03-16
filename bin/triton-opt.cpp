#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "triton/Dialect/Rock/IR/Rock.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Rock/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::registerTritonGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerConvertTritonGPUToRockPass();
  mlir::triton::registerConvertRockToLLVMPass();
  mlir::triton::registerConvertRockToGPUPass();
  mlir::rock::registerPasses();

  // TODO: register Triton & TritonGPU passes
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::gpu::GPUDialect>();
  // Register dialects used by Rock
  registry.insert<mlir::rock::RockDialect, mlir::memref::MemRefDialect,
                  mlir::amdgpu::AMDGPUDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
