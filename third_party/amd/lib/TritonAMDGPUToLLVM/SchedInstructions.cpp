#include "TritonAMDGPUToLLVM/Passes.h"

#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_SCHEDGROUPBARRIERSINSERTION
#define GEN_PASS_DEF_SCHEDGROUPBARRIERSLOWERING
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
enum InstructionMaskEnum {
  NONE = 0x0000000,
  ALL_ALU = 0x00000001,
  VALU = 0x00000002,
  SALU = 0x00000004,
  MFMA = 0x00000008,
  ALL_VMEM = 0x00000010,
  VMEM_READ = 0x00000020,
  VMEM_WRITE = 0x00000040,
  ALL_DS = 0x00000080,
  DS_READ = 0x00000100,
  DS_WRITE = 0x00000200
};

#define MOD_SCHED
//#define ENABLE_IGLP_OPT 0

#ifdef MOD_SCHED
const bool modifyScheduling{true};
#else
const bool modifyScheduling{false};
#endif // MOD_SCHED

#if defined(ENABLE_IGLP_OPT) && defined(MOD_SCHED)
const bool useIglpOpt{true};
const int iglpValue{ENABLE_IGLP_OPT};
#else
const bool useIglpOpt{false};
const int iglpValue{0};
#endif // ENABLE_IGLP_OPT

void buildSchedGroupBarrier(PatternRewriter &builder,
                            InstructionMaskEnum maskValue, int sizeValue,
                            int groupIdValue) {
  MLIRContext *ctx = builder.getContext();
  Location loc = builder.getUnknownLoc();
  auto intrinsicName = StringAttr::get(ctx, "llvm.amdgcn.sched.group.barrier");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = builder.getI32Type();
  auto mask = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, static_cast<int64_t>(maskValue)));
  auto size = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, sizeValue));
  auto groupId = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, groupIdValue));
  builder.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                        ValueRange{mask, size, groupId},
                                        defaultFlags);
}

Operation *generatedSchedBarrier(PatternRewriter &rewriter,
                                 int64_t maskValue) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = rewriter.getUnknownLoc();
  auto intrinsicName = StringAttr::get(ctx, "llvm.amdgcn.sched.barrier");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = rewriter.getI32Type();
  auto mask = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32, maskValue));
  return rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                                ValueRange{mask}, defaultFlags);
}

Operation *generatedIglpOpt(PatternRewriter &rewriter,
                            int value) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = rewriter.getUnknownLoc();
  auto intrinsicName = StringAttr::get(ctx, "llvm.amdgcn.iglp.opt");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = rewriter.getI32Type();
  auto mask = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32, static_cast<int64_t>(value)));
  return rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                                ValueRange{mask}, defaultFlags);
}

struct SchedGroupBarriersRewriter
    : public OpRewritePattern<triton::gpu::GroupSched> {
  using OpRewritePattern<triton::gpu::GroupSched>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::gpu::GroupSched schedBarrier,
                                PatternRewriter &rewriter) const override {
    
    if (useIglpOpt) {
      rewriter.setInsertionPointAfter(schedBarrier);
      auto *op = generatedIglpOpt(rewriter, iglpValue);
      rewriter.eraseOp(schedBarrier);
      return mlir::success();
    }

    Block *block = schedBarrier->getBlock();

    size_t numGlbLoads = 0;
    block->walk([&numGlbLoads](LLVM::CallOp callOp) {
      StringRef calleeName = callOp.getCallee().value();
      if (calleeName.contains("__predicated_load_vector"))
        ++numGlbLoads;
    });

    size_t numDsReads = 0;
    block->walk([&numDsReads](LLVM::LoadOp op) {
      auto operandType = op.getOperand().getType();
      if (auto ptr = llvm::dyn_cast<LLVM::LLVMPointerType>(operandType))
        if (ptr.getAddressSpace() == 3) {
          ++numDsReads;
        }
    });

    size_t numDsWrites = 0;
    block->walk([&numDsWrites](LLVM::StoreOp op) {
      auto operandType = op.getOperand(1).getType();
      if (auto ptr = llvm::dyn_cast<LLVM::LLVMPointerType>(operandType))
        if (ptr.getAddressSpace() == 3) {
          ++numDsWrites;
        }
    });

    size_t numMfmas = 0;
    block->walk([&numMfmas](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (opName.contains("mfma"))
        ++numMfmas;
    });

    llvm::dbgs() << "group scheduling info: ["
                 << "numGlbLoads: " << numGlbLoads << ", "
                 << "numDsReads: " << numDsReads << ", "
                 << "numDsWrites: " << numDsWrites << ", "
                 << "numMfmas: " << numMfmas << "]\n";

    rewriter.setInsertionPointToStart(block);
    Operation *op;
    op = generatedSchedBarrier(rewriter, InstructionMaskEnum::NONE);
    
    /*
    constexpr size_t num_mfma_per_issue = 3;
    constexpr size_t num_buffer_load_inst_a = 4;
    constexpr size_t num_dswrite_per_issue_a = 1;
    constexpr size_t num_buffer_load_inst_b = 4;
    constexpr size_t num_dswrite_per_issue_b = 1;
    constexpr size_t num_dsread_a_mfma = 4;
    constexpr size_t num_dsread_b_mfma = 4;
    constexpr size_t num_ds_read_inst_a = 8;
    constexpr size_t ds_read_a_mfma_rate = 2;
    constexpr size_t num_ds_read_inst_b = 8;
    constexpr size_t ds_read_b_mfma_rate = 2;
    for (size_t i = 0; i < num_buffer_load_inst_a; ++i) {
      for (size_t j = 0; num_dswrite_per_issue_a < 1; ++j) {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 1, 0);
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      }
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, num_mfma_per_issue - num_dswrite_per_issue_a, 0);
    }


    for (size_t i = 0; i < num_buffer_load_inst_b; ++i) {
      for (size_t j = 0; num_dswrite_per_issue_b < 1; ++j) {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 1, 0);
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      }
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, num_mfma_per_issue - num_dswrite_per_issue_b, 0);
    }

    for (size_t i = 0; i < num_dsread_a_mfma; ++i) {
      if ((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >= ds_read_a_mfma_rate) {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, ds_read_a_mfma_rate, 0);
      } else {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, num_ds_read_inst_a - (num_dsread_a_mfma - 1) *
                                                                              ds_read_a_mfma_rate, 0);
      }
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
    }

    for (size_t i = 0; i < num_dsread_b_mfma; ++i) {
      if ((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >= ds_read_b_mfma_rate) {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, ds_read_b_mfma_rate, 0);
      } else {
        buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, num_ds_read_inst_b - (num_dsread_b_mfma - 1) *
                                                                              ds_read_b_mfma_rate, 0);
      }
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
    }
    */
    
    /*
    for (size_t i = 0; i < 16; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 1, 0);
    }

    for (size_t i = 0; i < 4; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 2, 0);
    }

    for (size_t i = 0; i < 12; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, 2, 0);
    }
    */


    /*
    for (size_t j = 0; j < 8; ++j) {
      //for (size_t i = 0; i < 2; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 2, 0);
      //}
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 1, 0);
    }


    for (size_t i = 0; i < 8; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, 2, 0);
    }

    for (size_t i = 0; i < 8; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, 1, 0);
    }
    */

    rewriter.setInsertionPoint(block, std::prev(block->end()));

    for (size_t i = 0; i < 7; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 2, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 3, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 3, 0);
    }

    for (size_t i = 0; i < 5; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 3, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::ALL_VMEM, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 3, 0);
    }

    for (size_t i = 0; i < 7; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 3, 0);
    }

    for (size_t i = 0; i < 28; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
    }

    for (size_t i = 0; i < 7; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
    }

    op = generatedSchedBarrier(rewriter, InstructionMaskEnum::NONE);
    rewriter.eraseOp(schedBarrier);
    return mlir::success();
  }
};

struct SchedGroupBarriersLowering
    : public triton::impl::SchedGroupBarriersLoweringBase<
          SchedGroupBarriersLowering> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    if (!modifyScheduling)
      return;

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::gpu::GroupSched>();

    RewritePatternSet patterns(ctx);
    patterns.add<SchedGroupBarriersRewriter>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct SchedGroupBarriersInsertion
    : public triton::impl::SchedGroupBarriersInsertionBase<
          SchedGroupBarriersInsertion> {

  void insertPlaceholder(mlir::OpBuilder &builder, triton::DotOp dot) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(dot);
    Location loc = builder.getUnknownLoc();
    builder.create<triton::gpu::GroupSched>(loc);
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    if (!modifyScheduling)
      return;

    mlir::OpBuilder builder(ctx);
    mod.walk(
        [this, &builder](triton::DotOp op) { insertPlaceholder(builder, op); });
  }
};
} // namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>>
createSchedGroupBarriersLoweringPass() {
  return std::make_unique<SchedGroupBarriersLowering>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createSchedGroupBarriersInsertionPass() {
  return std::make_unique<SchedGroupBarriersInsertion>();
}
} // namespace triton
} // namespace mlir
