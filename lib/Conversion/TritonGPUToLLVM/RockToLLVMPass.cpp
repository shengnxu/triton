#include "triton/Conversion/TritonGPUToLLVM/RockToLLVMPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Rock/IR/Rock.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "GpuAllocOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
#include "TensorMemRefOpToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
#define GEN_PASS_DEF_CONVERTROCKTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx, bool isROCM)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    if (isROCM) {
      addLegalDialect<ROCDL::ROCDLDialect>();
      addLegalDialect<mlir::scf::SCFDialect>();
    } else {
      addLegalDialect<NVVM::NVVMDialect>();
    }
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    // Set an attribute to indicate this function is a kernel entry.
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
#ifndef USE_ROCM
    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid", rewriter.getI32ArrayAttr(32 * numWarps));
#endif

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx, bool isROCM)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    if (isROCM) {
      addLegalDialect<ROCDL::ROCDLDialect>();
      addLegalDialect<mlir::scf::SCFDialect>();
    } else {
      addLegalDialect<NVVM::NVVMDialect>();
    }
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class ConvertRockToLLVM
    : public impl::ConvertRockToLLVMBase<ConvertRockToLLVM> {

public:
  using impl::ConvertRockToLLVMBase<ConvertRockToLLVM>::ConvertRockToLLVMBase;
  explicit ConvertRockToLLVM(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget target(*context, /*isROCM*/ true);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    /* allocate shared memory and set barrier */
    Allocation allocation(mod);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    /* lower functions */
    {
      mlir::LowerToLLVMOptions option(context);
      option.overrideIndexBitwidth(32);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context, /*isROCM*/ true);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps,
                                         /*benefit=*/1);
      funcPatterns.add<ReturnOpConversion>(typeConverter);
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
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    /* rewrite ops */
    RewritePatternSet patterns(context);
    // TritonGPU lowering patterns
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};
    auto populatePatterns1 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, *axisInfoAnalysis,
                   &allocation, smem, indexCacheInfo, /*benefit*/ 1);
    };
    auto populatePatterns2 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, *axisInfoAnalysis,
                   &allocation, smem, /*benefit*/ 1);
    };
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
    populateGpuAllocOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                     *axisInfoAnalysis, &allocation, smem,
                                     /*benefit*/ 1, context);
    populatePatterns1(populateTritonGPUToLLVMPatterns);
    populatePatterns1(populateTensorMemRefOpToLLVMPatterns);
    populatePatterns1(populateConvertLayoutOpToLLVMPatterns);
    // populatePatterns2(populateDotOpToLLVMPatterns);
    populatePatterns2(populateElementwiseOpToLLVMPatterns);
    populatePatterns1(populateLoadStoreOpToLLVMPatterns);
    populatePatterns1(populateReduceOpToLLVMPatterns);
    populatePatterns2(populateViewOpToLLVMPatterns);

    // Native lowering patterns
#ifdef USE_ROCM
    // GPUToROCDL lowering patterns
    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse("gfx90a");
    populateAMDGPUToROCDLConversionPatterns(typeConverter, patterns,
                                            *maybeChipset);
    populateVectorToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateGpuToROCDLConversionPatterns(typeConverter, patterns,
                                         mlir::gpu::amd::HIP);
#else
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
#endif
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  Value smem;

  using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;

  int computeCapability{};
  bool isROCM{};

  void initSharedMemory(size_t size,
                        TritonGPUToLLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/0,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
    SmallVector<LLVM::LLVMFuncOp> funcs;
    mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
    assert(funcs.size() == 1 &&
           "Inliner pass is expected before TritonGPUToLLVM");
    b.setInsertionPointToStart(&funcs[0].getBody().front());
    smem = b.create<LLVM::AddressOfOp>(loc, global);
    auto ptrTy = LLVM::LLVMPointerType::get(b.getContext(), 3);
    smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertRockToLLVMPass(int computeCapability) {
  return std::make_unique<::ConvertRockToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
