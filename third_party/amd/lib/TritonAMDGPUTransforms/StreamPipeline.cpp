#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/LoopSchedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

#include <list>

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.


class StreamSchedule : public tt::LoopSchedule {
protected:
  bool isLoadOp(Operation *op) override {
    return isa<tt::LoadOp>(op);
  }
  
  void createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                       Value insertIdx, Value extractIdx,
                       tt::CoarseSchedule::Cluster prefetchCluster) override;
  void createTMAAsyncCopy(tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
                          Value insertIdx, Value extractIdx, Value barrier,
                          Operation *waitOp, Value phase) override { assert(0); }
  void createTMABarrierAndWait(SmallVector<AsyncLoad> &asyncLoads,
                               Value insertIdx, Value extractIdx, Value phase,
                               int numBuffers) override {}
public:
  using tt::LoopSchedule::LoopSchedule;
};

void StreamSchedule::createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                     Value insertIdx, Value extractIdx,
                                     tt::CoarseSchedule::Cluster prefetchCluster) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  if (!isExpensiveLoadOrStore(loadOp) && loadToInfo[loadOp].blockedEncoding) {
    // For inexpensive loads that do not directly feed into dot ops
    // we want to use optimal layout for the data.
    ttg::BlockedEncodingAttr encoding = loadToInfo[loadOp].blockedEncoding;
    auto convertBlockLayout = [&](Value src) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), encoding);
      auto cvt =
          builder.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src);
    if (mask)
      mask = convertBlockLayout(mask);
    if (other)
      other = convertBlockLayout(other);
  }

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  Operation *copy = builder.clone(*loadOp);

  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  Operation *lds_store =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);
  {
    // Clean up old local caches.
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        alloc.replaceAllUsesWith(viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad =
        builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    Value other = loadOp.getOther();
    if (other && !isZeroConst(other)) {
      auto select = builder.create<arith::SelectOp>(
          loc, loadOp.getType(), mask, sharedLoad.getResult(), other);
      result = select->getResults();
    }

    loadOp->replaceAllUsesWith(result);

    // Prefetch load if is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      schedule.insert(lds_store, numStages - 2, prefetchCluster);
      schedule.insert(viewLoad, numStages - 2, prefetchCluster);
    }
  }
  loadOp.erase();
}


static bool
preProcessLoopAndGetSchedule2(scf::ForOp &forOp, int numStages,
                              mlir::triton::PipeliningOption &options) {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.

  StreamSchedule sched(forOp, numStages, true);

  if (!sched.compute())
    return false;

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  auto schedule = sched.getSchedule();

  forOp = sched.getNewForLoop();

  // Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::triton::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};
  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : sched.getAllocs())
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

// Return true if the preconditions for pipelining the loop are met.
static bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

static void tryAndPipelineOuterLoop(scf::ForOp forOp) {
  mlir::triton::PipeliningOption options;
  bool foundSchedule = false;
  // Limit 2 stages to not require extra shared memory.
  foundSchedule = getOuterLoopSchedule(forOp, /*numStage=*/2, options);
  if (!foundSchedule)
    return;
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);
}

static bool pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  if (!preCondition(forOp))
    return false;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule2(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    return false;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return false;
  return true;
}

namespace {
struct PipelinePass : public TritonAMDGPUStreamPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int32_t numStages) { this->numStages = numStages; }

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (auto attr =
            forOp->getAttrOfType<IntegerAttr>(mlir::triton::kNumStagesAttrName))
      return attr.getInt();
    return numStages;
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;

    llvm::SmallSetVector<scf::ForOp, 8> outerLoops;
    for (scf::ForOp forOp : loops) {
      auto outerLoop = dyn_cast<scf::ForOp>(forOp->getParentOp());
      int loopNumStages = getNumStagesOrDefault(forOp);
      bool pipelined = pipelineLoop(forOp, loopNumStages);
      if (pipelined && outerLoop && getNumStagesOrDefault(outerLoop) > 1)
        outerLoops.insert(outerLoop);
    }

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed())
      return signalPassFailure();

    // Try to pipeline the outer loop to overlap the prologue and epilogue of
    // the inner loop.
    for (scf::ForOp outerLoop : outerLoops)
      tryAndPipelineOuterLoop(outerLoop);
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPUStreamPipelinePass(int numStages) {
  return std::make_unique<PipelinePass>(numStages);
}
