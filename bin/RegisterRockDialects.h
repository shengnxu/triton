#pragma once
#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "triton/Dialect/Rock/IR/Rock.h"

#include "triton/Dialect/Rock/Passes.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Dialect.h"

#include "mlir/InitAllPasses.h"

inline void registerRockDialects(mlir::DialectRegistry &registry) {
  mlir::rock::registerPasses();

  // Register dialects used by Rock
  registry.insert<mlir::rock::RockDialect, mlir::memref::MemRefDialect,
                  mlir::amdgpu::AMDGPUDialect, mlir::vector::VectorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect>();
}
