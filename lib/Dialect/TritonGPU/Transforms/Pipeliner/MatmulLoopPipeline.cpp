#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/LoopSchedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "triton-matmul-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

class MatmulSchedule : public tt::LoopSchedule {
protected:
  void createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                       Value insertIdx, Value extractIdx,
                       tt::CoarseSchedule::Cluster prefetchCluster) override;
  void createTMAAsyncCopy(tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
                          Value insertIdx, Value extractIdx, Value barrier,
                          Operation *waitOp, Value phase) override;
  void createTMABarrierAndWait(SmallVector<AsyncLoad> &asyncLoads,
                               Value insertIdx, Value extractIdx, Value phase,
                               int numBuffers) override;
public:
  using tt::LoopSchedule::LoopSchedule;
  SmallVector<Value> barriers;

};

void MatmulSchedule::createAsyncCopy(tt::LoadOp loadOp, Value alloc,
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
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commmit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;
  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);
  schedule.insert(commmit, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
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

    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
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

    // Prefetch load if is not MMAV3 and is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      schedule.insert(wait, numStages - 2, prefetchCluster);
      schedule.insert(viewLoad, numStages - 2, prefetchCluster);
    }
  }
  loadOp.erase();
}

void MatmulSchedule::createTMAAsyncCopy(
    tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    Value phase) {
  assert(phase && "Phase value is required for TMA async copy.");
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);

  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Operation *copy = builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
      loc, loadOp.getDescPtr(), loadOp.getIndices(), barrier, view, pred);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;
  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  builder.setInsertionPointAfter(waitOp);
  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
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

    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  loadOp.erase();
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(scf::ForOp &forOp, unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = forOp.getLoc();
  auto context = forOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = tt::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      tt::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                           sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(forOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

// Create barriers and wait ops for the async loads. Barriers may be shared by
// multiple loads is the schedule allows it.
void MatmulSchedule::createTMABarrierAndWait(
    SmallVector<AsyncLoad> &asyncLoads, Value insertIdx,
    Value extractIdx, Value phase, int numBuffers) {
  llvm::SmallDenseMap<Operation *, AsyncLoad *> loadToAsyncLoad;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    loadToAsyncLoad[asyncLoad.loadOp] = &asyncLoad;
  }
  SmallVector<SmallVector<AsyncLoad *>> loadGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (!asyncLoad.isTMALoad || visited.count(asyncLoad.loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<AsyncLoad *> group;
    Block *loadBlock = asyncLoad.loadOp->getBlock();
    auto addToGroup = [&](AsyncLoad *loadInfo) {
      group.push_back(loadInfo);
      visited.insert(loadInfo->loadOp);
      for (Operation *user : loadInfo->loadOp->getUsers()) {
        auto it = loadToInfo.find(loadInfo->loadOp);
        if (it != loadToInfo.end()) {
          // Special case for MMAv3 loads, we can ignore the alloc and only
          // consider uses of the alloc op since it will be removed.
          if (it->second.loadIsMMAV3) {
            auto alloc = cast<ttg::LocalAllocOp>(
                (*loadInfo->loadOp->getUsers().begin()));
            if (alloc->getBlock() == loadBlock) {
              users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
              continue;
            }
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(&asyncLoad);
    Operation *nextOp = asyncLoad.loadOp->getNextNode();
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isa<tt::ExperimentalDescriptorLoadOp>(nextOp)) {
        auto it = loadToAsyncLoad.find(nextOp);
        if (it != loadToAsyncLoad.end() && it->second->isTMALoad) {
          addToGroup(it->second);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    loadGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<AsyncLoad *> &group : loadGroups) {
    int sizeInBytes = 0;
    for (AsyncLoad *asyncLoad : group) {
      auto tensorTy =
          cast<RankedTensorType>(asyncLoad->loadOp->getResult(0).getType());
      int loadSize = product(tensorTy.getShape());
      sizeInBytes +=
          loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
    }

    Value barrierAlloc = createBarrierAlloc(forOp, numBuffers);
    barriers.push_back(barrierAlloc);
    Location loc = forOp.getLoc();
    OpBuilder builder(forOp);
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    tt::MemDescType barrierTy = tt::MemDescType::get(
        {1}, builder.getI64Type(),
        cast<tt::MemDescType>(barrierAlloc.getType()).getEncoding(),
        sharedMemorySpace,
        /*mutableMemory=*/true);
    builder.setInsertionPoint(group[0]->loadOp);
    Value barrier = builder.create<ttg::MemDescSubviewOp>(
        loc, barrierTy, barrierAlloc, ArrayRef<Value>({insertIdx}));
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Operation *expect = builder.create<ttng::BarrierExpectOp>(
        forOp.getLoc(), barrier, sizeInBytes, pred);
    auto [stage, cluster] = schedule[asyncLoads[0].loadOp];
    schedule.insert(expect, stage, cluster);

    builder.setInsertionPointAfter(group.back()->loadOp);
    Value barrierViewWait = builder.create<ttg::MemDescSubviewOp>(
        loc, barrierTy, barrierAlloc, ArrayRef<Value>({extractIdx}));
    Operation *wait =
        builder.create<ttng::WaitBarrierOp>(loc, barrierViewWait, phase);
    // Update the async loads info.
    for (AsyncLoad *asyncLoad : group) {
      asyncLoad->barrier = barrier;
      asyncLoad->waitOp = wait;
    }
  }
}


static void invalidateBarriers(OpBuilder &builder,
                               SmallVector<Value> &barriers) {
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
  for (Value barrier : barriers) {
    int numBarriers = cast<tt::MemDescType>(barrier.getType()).getShape()[0];
    for (int i = 0; i < numBarriers; i++) {
      Value idx = builder.create<arith::ConstantIntOp>(barrier.getLoc(), i, 32);
      tt::MemDescType barrierTy = tt::MemDescType::get(
          {1}, builder.getI64Type(),
          cast<tt::MemDescType>(barrier.getType()).getEncoding(),
          sharedMemorySpace,
          /*mutableMemory=*/true);
      Value barrierView = builder.create<ttg::MemDescSubviewOp>(
          barrier.getLoc(), barrierTy, barrier, idx);
      builder.create<ttng::InvalBarrierOp>(barrier.getLoc(), barrierView);
    }
  }
}

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.

  MatmulSchedule sched(forOp, numStages);

  // return true if something to do..
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
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), ValueRange({}), 0);
  // Invalidate any mbarrier create
  invalidateBarriers(builder, sched.barriers);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : sched.getAllocs())
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value value assigned to the argument coming from outside the
      // loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  int minCommits = minOverHistories(waitOp->getOperand(0), waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.back());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<tt::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

// Determines whether a given MMAv3 dot op, represented as ttng.warp_group_dot,
// needs a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either ttng.warp_group_dot {isAsync=True} or ttng.warp_group_dot
// {isAsync=False}.  But even if we use ttng.warp_group_dot {isAsync=True}, the
// conservative thing is to make a dot "effectively synchronous" by inserting a
// `ttng.warp_group_dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//
//  2. If the dot is used by any op in the loop, it must be used under an `if`,
//     and will be synced with a `wait 0` at the beginning of the `if` block.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.warp_group_dot_wait {pendings=0}` op, the result of the dot from
//     iteration i-1 is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = warp_group_dot %a1, %b1, %c1
//        %accum = warp_group_dot %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another warp_group_dot, and this only works for
//     the `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::WarpGroupDotOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two
//     ttng::WarpGroupDotOps don't have to correspond to wgmma.async ops with
//     the same shapes as specified in the docs, because there's an intervening
//     fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::WarpGroupDotOp dotOp,
                                                scf::ForOp forOp) {
  LDBG("Considering whether to make MMAv3 dot properly async: " << dotOp);

  // Rule 1: All shmem operands are multi-buffered.
  auto checkOperand = [&](Value operand) {
    if (!isa<ttg::SharedEncodingAttr>(
            cast<TensorOrMemDesc>(operand.getType()).getEncoding())) {
      return true;
    }

    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescSubview op.  Only ConvertLayout and Trans ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, tt::TransOp>(
        transitiveOperand.getDefiningOp())) {
      transitiveOperand = transitiveOperand.getDefiningOp()->getOperand(0);
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           isa<ttg::MemDescSubviewOp>(transitiveOperand.getDefiningOp());
  };

  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB())) {
    LDBG("Can't make dot async because shmem operands aren't multi-buffered");
    return std::nullopt;
  }

  // Rule 2: The dot cannot be unconditionally used by any op in the loop.
  // Uses under `if` are allowed, as can be explicitly synced with a `wait 0`.
  int iterArgIdx = -1;
  Value iterArg = nullptr;
  SmallVector<std::pair<Operation *, int>> queue;
  for (auto &use : dotOp->getUses()) {
    queue.push_back({use.getOwner(), use.getOperandNumber()});
  }
  while (!queue.empty()) {
    auto [user, argIdx] = queue.pop_back_val();
    if (user->getParentOp() == forOp) {
      if (isa<scf::YieldOp>(user)) {
        if (iterArg) {
          // The dot is used by the loop's yield, but we can't have any other
          // uses.
          LDBG("Can't make dot async because dot is used by multiple ops in "
               "the loop.");
          return std::nullopt;
        }
        iterArgIdx = argIdx;
        iterArg = forOp.getRegionIterArg(argIdx);
        continue;
      }
      LDBG("Can't make dot async because dot is unconditionally used in the "
           "loop.");
      return std::nullopt;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
      if (isa<scf::YieldOp>(user)) {
        // The result is returned by the if, follow it further.
        auto uses = ifOp.getResult(argIdx).getUses();
        for (auto &use : uses) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
      }
    } else {
      return std::nullopt;
    }
  }

  // Rule 3a: Are the only users of the dot's result from iteration i-1 other
  // MMAv3 dots?  If so, we're done, this dot can be properly async.
  if (llvm::all_of(iterArg.getUses(), [&](OpOperand &use) {
        return isa<ttng::WarpGroupDotOp>(use.getOwner()) &&
               use.getOperandNumber() == 2;
      })) {
    return iterArgIdx;
  }

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `warp_group_dot_wait {pendings=0}` op?  If so, the dot can be
  // properly async, but we have to thread its result from iteration i-1 through
  // the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>();
  auto firstWaitOpIter = llvm::find_if(
      waitOps, [&](auto waitOp) { return waitOp.getPendings() == 0; });
  if (firstWaitOpIter != waitOps.end() &&
      llvm::all_of(iterArg.getUsers(), [&](Operation *user) {
        assert(forOp->isAncestor(user));
        while (user->getParentOp() != forOp) {
          user = user->getParentOp();
        }
        return (*firstWaitOpIter)->isBeforeInBlock(user);
      })) {
    LDBG("MMAv3 dot can be properly async because it follows a "
         "warp_group_dot_wait "
         "{pendings=0}.\n"
         << "  wait: " << *firstWaitOpIter << "\n"
         << "  dot: " << dotOp);
    threadValuesThroughWait(*firstWaitOpIter, {iterArg});
    return iterArgIdx;
  }

  LDBG("Can't make dot async because its result from i-1 is used by "
       "something other than another MMAv3 dot as the `c` operand.");
  return std::nullopt;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the properly-async dots from iteration i-1 to complete.  (We pipeline to
// depth 2, so there are at most 2 copies of each warp_group_dot in flight at a
// time.)
//
// We can skip inserting the wait if we have a `warp_group_dot_wait
// {pendings=0}` somewhere in the loop.  To see why, consider:
//
//   warp_group_dot
//   warp_group_dot; wait 0  // synchronous dot
//   warp_group_dot
//   warp_group_dot
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncWarpGroupDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
  }

  // Insert waits before the users of the properly async dots other than loop
  // yield.
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    SmallVector<OpOperand *> uses;
    for (auto &use : asyncDot->getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        continue;
      }
      uses.push_back(&use);
    }

    DenseMap<Block *, SmallVector<Value>> blockToUsers;
    for (auto use : uses) {
      auto block = use->getOwner()->getBlock();
      blockToUsers[block].push_back(use->get());
    }

    for (auto [block, users] : blockToUsers) {
      OpBuilder builder(block, block->begin());
      auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
          asyncDot->getLoc(), ArrayRef<Value>{}, 0);

      threadValuesThroughWait(newWait, users);
    }
  }

  // Add the wait right after the last properly-async dot.  This only needs to
  // wait for all properly-async dots from the i-1'th iteration to complete, IOW
  // we wait until there are most `asyncDots.size()` dots in flight.
  //
  // (You might want to put the wait at the end of the loop instead of right
  // after the last dot, but there could be a load into shmem between the last
  // async dot and the end of the loop, and that could clobber memory being used
  // by a dot.)
  IRRewriter builder(forOp.getContext());
  auto lastAsyncDot = properlyAsyncDots.back().first;
  builder.setInsertionPointAfter(lastAsyncDot);
  auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
      lastAsyncDot->getLoc(),
      /*inputs=*/ArrayRef<Value>{}, properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto WarpGroupDotOp : forOp.getBody()->getOps<ttng::WarpGroupDotOp>()) {
    WarpGroupDotOp.setIsAsync(true);
    if (auto iterArgIdx = dotCanBeProperlyAsync(WarpGroupDotOp, forOp)) {
      properlyAsyncDots[WarpGroupDotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(WarpGroupDotOp);
      auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
          WarpGroupDotOp.getLoc(), ArrayRef<Value>{},
          /*pendings=*/0);
      SmallVector<Value> waitOperands = {WarpGroupDotOp.getResult()};
      threadValuesThroughWait(wait, waitOperands);
    }
  }

  if (properlyAsyncDots.empty()) {
    LDBG("No properly async dots.");
    return;
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncWarpGroupDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto WarpGroupDotWaitAfterLoop = builder.create<ttng::WarpGroupDotWaitOp>(
      forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);
}
