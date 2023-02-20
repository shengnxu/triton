#include "triton/Conversion/TritonGPUToLLVM/RockToLLVMPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

namespace mlir {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
#ifdef USE_ROCM
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
#else
    addLegalDialect<NVVM::NVVMDialect>();
#endif
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
#ifdef USE_ROCM
    addLegalDialect<ROCDL::ROCDLDialect>();
#else
    addLegalDialect<NVVM::NVVMDialect>();
#endif
    addIllegalOp<mlir::func::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

} // namespace mlir

namespace {

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

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

class ConvertRockToLLVM
    : public ConvertRockToLLVMBase<ConvertRockToLLVM> {

public:
  explicit ConvertRockToLLVM(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    TritonLLVMConversionTarget target(*context);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // The first 2 steps are done in TritonGPUToRockPass
    // Step 1: Decompose unoptimized layout conversions to use shared memory
    // Step 2: Decompose insert_slice_async to use load + insert_slice for
    //   pre-Ampere architectures or unsupported vectorized load sizes
    // Step 3: Allocate shared memories and insert barriers
    // Step 4: Convert SCF to CFG
    // Step 5: Convert FuncOp to LLVMFuncOp via partial conversion
    // Step 6: Get axis and shared memory info
    // Step 7: Convert the rest of ops via partial conversion
    //
    // The reason for putting step 3 before step 4 is that the membar
    // analysis currently only supports SCF but not CFG. The reason for a
    // separation between 5/7 is that, step 6 is out of the scope of Dialect
    // Conversion, thus we need to make sure the smem is not revised during the
    // conversion of step 7.

    // Step 3
    Allocation allocation(mod);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Step 4
    RewritePatternSet scf_patterns(context);
    mlir::populateSCFToControlFlowConversionPatterns(scf_patterns);
    mlir::ConversionTarget scf_target(*context);
    scf_target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp,
                            scf::WhileOp, scf::ExecuteRegionOp>();
    scf_target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(
            applyPartialConversion(mod, scf_target, std::move(scf_patterns))))
      return signalPassFailure();

    // Step 5
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(typeConverter, numWarps, /*benefit=*/1);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    // Step 6 - get axis and shared memory info
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    // Step 7 - rewrite rest of ops
    // We set a higher benefit here to ensure triton's patterns runs before
    // arith patterns for some encoding not supported by the community
    // patterns.
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};

    RewritePatternSet patterns(context);

    // Normal conversions
    populateTritonGPUToLLVMPatterns(typeConverter, patterns, numWarps,
                                    *axisInfoAnalysis, &allocation, smem,
                                    indexCacheInfo, /*benefit=*/10);
    // ConvertLayoutOp
    populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                          *axisInfoAnalysis, &allocation, smem,
                                          indexCacheInfo, /*benefit=*/10);
    // DotOp
    populateDotOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                *axisInfoAnalysis, &allocation, smem,
                                /*benefit=*/10);
    // ElementwiseOp
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                        *axisInfoAnalysis, &allocation, smem,
                                        /*benefit=*/10);
    // LoadStoreOp
    populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                      *axisInfoAnalysis, &allocation, smem,
                                      indexCacheInfo, /*benefit=*/10);
    // ReduceOp
    populateReduceOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                   *axisInfoAnalysis, &allocation, smem,
                                   indexCacheInfo, /*benefit=*/10);
    // ViewOp
    populateViewOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                 *axisInfoAnalysis, &allocation, smem,
                                 /*benefit=*/10);

    // Add arith/math's patterns to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
#ifdef USE_ROCM
    mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns, mlir::gpu::amd::HIP);
#else
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
#endif

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // Take care of scf pattern introduced by LoadStoreOp
#ifdef USE_ROCM
    RewritePatternSet scf_patterns_extra(context);
    mlir::populateSCFToControlFlowConversionPatterns(scf_patterns_extra);
    if (failed(
            applyPartialConversion(mod, scf_target, std::move(scf_patterns_extra))))
      return signalPassFailure();
    RewritePatternSet patterns_extra(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns_extra);
    if (failed(
            applyPartialConversion(mod, target, std::move(patterns_extra))))
      return signalPassFailure();
#endif
  }

private:
  Value smem;

  using IndexCacheKeyT = std::pair<Attribute, SmallVector<int64_t>>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;

  int computeCapability{};

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
    auto ptrTy =
        LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 3);
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
