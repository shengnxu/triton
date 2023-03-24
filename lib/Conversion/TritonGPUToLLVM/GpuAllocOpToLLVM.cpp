#include "GpuAllocOpToLLVM.h"

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Dialect/Rock/IR/Rock.h"

using namespace mlir;
using namespace mlir::triton;

struct MIGPUAllocRewritePattern
    : public ConvertTritonGPUOpToLLVMPattern<mlir::rock::GpuAllocOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::rock::GpuAllocOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult matchAndRewrite(rock::GpuAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto type = op.getOutput().getType();
    assert(type && type.hasStaticShape() && "unexpected type in rock.alloc()");
    auto gpuAttr =
        type.getMemorySpace().dyn_cast<mlir::gpu::AddressSpaceAttr>();
    Location loc = op->getLoc();

    if (gpuAttr.getValue() ==
        mlir::gpu::GPUDialect::getWorkgroupAddressSpace()) {
      llvm::errs() << "rock.alloc() for LDS is not supported!\n";
    } else if (gpuAttr.getValue() ==
               mlir::gpu::GPUDialect::getPrivateAddressSpace()) {
      Type elementType = getTypeConverter()->convertType(type.getElementType());
      auto ptrType =
          ptr_ty(op.getContext(),
                 mlir::ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace);
      Value numElements =
          rewriter.create<LLVM::ConstantOp>(loc, i64_ty, type.getNumElements());
      Value allocated = rewriter.create<LLVM::AllocaOp>(
          loc, ptrType, elementType, numElements, /*alignment=*/0);
      auto descr = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), type, allocated);
      rewriter.replaceOp(op, {descr});
    } else {
      // TBD: return failure.
      llvm::errs() << "unsupported addrspace!\n";
    }
    return success();
  }
};

template <typename Tmi, typename Tgpu>
struct MIIdRewritePattern : public OpRewritePattern<Tmi> {
  using OpRewritePattern<Tmi>::OpRewritePattern;

  LogicalResult matchAndRewrite(Tmi op, PatternRewriter &b) const override {
    Value nop =
        b.create<Tgpu>(op.getLoc(), b.getIndexType(), mlir::gpu::Dimension::x);
    b.replaceOp(op, nop);
    return success();
  }
};

void populateGpuAllocOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem, PatternBenefit benefit,
    MLIRContext *ctx) {
  patterns.add<MIGPUAllocRewritePattern>(typeConverter, benefit);
  // patterns.add<WorkitemIdRewritePattern>(typeConverter, benefit);
  patterns
      .add<MIIdRewritePattern<mlir::rock::WorkitemIdOp, mlir::gpu::ThreadIdOp>>(
          ctx);
}
