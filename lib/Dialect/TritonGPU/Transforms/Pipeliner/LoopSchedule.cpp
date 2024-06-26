#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/LoopSchedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.


void tt::CoarseSchedule::insertDepsOfOp(Operation *op, int stage,
                                        tt::CoarseSchedule::Cluster cluster,
                                        bool includeArg) {
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = dyn_cast<BlockArgument>(v)) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      if (insertIfAbsent(defOp, stage, cluster)) {
        insertDepsOfOp(defOp, stage, cluster, includeArg);
      }
    }
  }
}

SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
tt::CoarseSchedule::getOpsInOrder(scf::ForOp forOp) {
  SmallVector<SmallVector<std::tuple<Operation *, int, Cluster>>, 8>
      orderClusters(clusters.size());
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opToStageAndCluster.count(&op) == 0) {
      continue;
    }
    assert(opToStageAndCluster[&op].first < numStages &&
           "Op with invalid stage!");
    int clusterId = *opToStageAndCluster[&op].second;
    assert(clusterId == std::distance(clusters.begin(),
                                      opToStageAndCluster[&op].second) &&
           "Cluster ID mismatch!");
    orderClusters[clusterId].push_back(make_tuple(
        &op, opToStageAndCluster[&op].first, opToStageAndCluster[&op].second));
  }
  SmallVector<std::tuple<Operation *, int, Cluster>> opsInOrder;
  for (int i = 0; i < orderClusters.size(); i++) {
    for (auto [op, stage, cluster] : orderClusters[i]) {
      opsInOrder.push_back({op, stage, cluster});
    }
  }

  return opsInOrder;
}

std::vector<std::pair<Operation *, unsigned>>
tt::CoarseSchedule::createFinalSchedule(scf::ForOp forOp) {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = getOpsInOrder(forOp);
  std::vector<std::pair<Operation *, unsigned>> schedule;
  for (auto [op, stage, cluster] : opsInOrder)
    schedule.push_back({op, stage});
  return schedule;
}

void tt::CoarseSchedule::dump() {
  for (int i = 0; i < numStages; i++) {
    llvm::dbgs() << "\n---- Ops in stage " << i << "\n";
    for (auto &[op, stageAndCluster] : opToStageAndCluster) {
      if (i == stageAndCluster.first) {
        llvm::dbgs() << "        cluster: " << *stageAndCluster.second
                     << ":\n\t" << *op << "\n";
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//  LoopSchedule management class
////////////////////////////////////////////////////////////////////////////////

bool tt::LoopSchedule::isLoadOp(Operation *op) {
  return isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op);
}

// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the shared encoding that needs to be
// used to be compatible with users' layouts. If there are imcompatible shared
// encodings set `incompatible` to true.
static std::optional<ttg::SharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val, bool &incompatible) {
  ttg::SharedEncodingAttr attr;
  incompatible = false;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0), incompatible)
               .has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<TensorOrMemDesc>(user->getResult(0).getType()).getEncoding());
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = cast<TensorOrMemDesc>(val.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(),
          ttg::getOrder(srcTy.getEncoding()),
          ttg::getCTALayout(srcTy.getEncoding()),
          srcTy.getElementType().getIntOrFloatBitWidth(), /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (attr != nullptr && attr != tempAttr) {
      incompatible = true;
      return std::nullopt;
    }
    attr = tempAttr;
  }
  return attr;
}

static ttg::BlockedEncodingAttr
getBlockedEncoding(tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value src = loadOp.getPtr();
  auto ty = cast<RankedTensorType>(src.getType());
  auto mod = loadOp->getParentOfType<ModuleOp>();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  tt::AxisInfo::DimVectorT contiguity =
      axisInfo.getAxisInfo(src)->getContiguity();
  SmallVector<unsigned> order = argSort(contiguity);
  unsigned currPerThread = getNumElementsPerThread(loadOp, order, axisInfo);
  SmallVector<unsigned> sizePerThread(order.size(), 1);
  sizePerThread[order[0]] = currPerThread;
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  return ttg::BlockedEncodingAttr::get(loadOp->getContext(), ty.getShape(),
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, ctaLayout);
}

static std::optional<ttg::SharedEncodingAttr>
getSharedEncoding(Operation *loadOp, bool isMMAV3) {
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto blockedOrder = ttg::getOrder(ty.getEncoding());
  SmallVector<unsigned> order;
  if (blockedOrder.size() == 3) {
    for (unsigned i = 0; i < blockedOrder.size(); ++i) {
      if (blockedOrder[i] == 0)
        continue;
      order.push_back(blockedOrder[i]);
    }
    order.push_back(0);
  } else {
    order = blockedOrder;
  }
  if (isMMAV3) {
    return ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(), order,
                                        ctaLayout, ty.getElementType());
  }

  // If the load is used by a LocalAllocOp, use the same encoding as the allocs.
  // If the allocs don't all have the same encoding, bail.
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    ttg::SharedEncodingAttr localAllocEnc;
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingAttr>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc)
        return std::nullopt;
    }
    return localAllocEnc;
  }

  // Use non-swizzled layout for loads that do not feed into dot ops.
  // TODO: This won't be optimal for 2D tensors.
  return ttg::SharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                      ctaLayout);
}

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
tt::LoopSchedule::loadOpsToIndirectionLevelAndUse() {
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse;
  DenseSet<Operation *> seen;

  std::function<void(Operation *, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (isLoadOp(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.push_back(std::make_tuple(op, distance, use));
          use = op;
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasTrait<OpTrait::DotLike>())
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isLoadOp(&op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToIndLevelAndUse;
}

static bool loadIsMMAv3(Operation *loadOp) {
  if (!loadOp->hasOneUse())
    return false;
  auto alloc = dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
  if (!alloc)
    return false;
  auto sharedEnc = cast<ttg::SharedEncodingAttr>(alloc.getType().getEncoding());
  if (!sharedEnc.getHasLeadingOffset())
    return false;

  // MMA V3 case.
  auto newOrder = sharedEnc.getOrder();
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  auto oldOrder = ttg::getOrder(ty.getEncoding());

  // The operand of MMAv3 is in SharedEncoding and its order should not
  // be changed after FuseTranspositions Pass. So we only pipeline the
  // load if the order of the loaded BlockedEncoding is the same as the
  // order of the SharedEncoding it is converted to.
  return oldOrder == newOrder;
}

void tt::LoopSchedule::assignMemoryLayouts(
     llvm::SmallVector<std::tuple<Operation *, int, Operation *>> &loadOpToIndLevelAndUse,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(op))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;
    LoadInfo loadInfo;

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(!isLoadFromTensorPtr(loadOp) &&
             "Block ptr should have been lowered before this pass.");
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy)
        continue;
      auto ty =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();

      // We do not pipeline all loads for the following reasons:
      // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
      // 2. It's likely that pipling small loads won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      LDBG("Load " << *loadOp << " has width " << width);
      if (width < 32)
        continue;
    }

    if (use->hasTrait<OpTrait::DotLike>()) {
      loadInfo.usedByDot = true;
      if (loadIsMMAv3(op)) {
        loadInfo.loadIsMMAV3 = true;
        loadInfo.sharedEncoding =
            getSharedEncoding(op, /*loadIsMMAv3=*/true).value_or(nullptr);
      } else if (isa<tt::ExperimentalDescriptorLoadOp>(op)) {
        loadInfo.sharedEncoding =
            getSharedEncoding(op, /*loadIsMMAv3=*/true).value_or(nullptr);
      } else if (auto dot = dyn_cast<tt::DotOp>(use)) {
        bool incompatible = false;
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
                .value_or(nullptr);
        // If we can't agree on a shared encoding skip pipelinig the load.
        if (incompatible)
          continue;
        // HACK: Triton LLVM codegen has a bug where local_loads from #shared to
        // #mma layout can lead to invalid code if the loaded shape is smaller
        // than the mma tile (e.g. loading a 128x1 tensor for an MMAv2 dot with
        // tile {16,8} is bad because 1 < 8).  To work around this, don't
        // pipeline such loads.
        //
        // The codegen bug is caught by an assertion, so if you think you've
        // fixed it, feel free to delete this code and see if the assert still
        // fails.  :)
        if (!loadInfo.sharedEncoding) {
          if (auto dotEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(
                  dot.getResult().getType().getEncoding())) {
            auto loadTy = cast<RankedTensorType>(op->getResultTypes()[0]);
            auto mmaInstrShape = dotEnc.getInstrShape();
            if (loadTy.getRank() < mmaInstrShape.size())
              continue;
            bool ok = true;
            for (int i = 0; i < mmaInstrShape.size(); i++) {
              if (loadTy.getShape()[loadTy.getRank() - mmaInstrShape.size() +
                                    i] < mmaInstrShape[i]) {
                ok = false;
                break;
              }
            }
            // If this load might trigger the bug, don't do the fallback logic
            // below, which might allow the load to be pipelined.
            if (!ok)
              continue;
          }
        }
      }
    } else if (auto loadOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadsToPipeline already, it means that the use is not valid for
      // pipelining for some reason. We should skip this loadOp, too. Note that
      // we have an assumption that distAndUse.second (i.e. the use of this
      // loadOp) has already be processed in a previous loop iteration. This
      // assumption is held by how loadOpsToIndirectionLevelAndUse recursively
      // collects loadOpToIndLevelAndUse using DFS.
      if (loadToInfo.count(loadOp) == 0) {
        continue;
      }
    }

    // If we still don't have a shared encoding, try a "generic" shared
    // encoding.
    if (!loadInfo.sharedEncoding && !isa<ttng::WarpGroupDotOp>(use)) {
      if (!canRegisterBuffer) {
        loadInfo.sharedEncoding =
            getSharedEncoding(op, /*isMMAV3=*/loadInfo.loadIsMMAV3)
                .value_or(nullptr);
      }
      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        loadInfo.blockedEncoding = getBlockedEncoding(loadOp, axisInfoAnalysis);
      }
    }

    // If that still didn't work, bail on pipelining this load unless register
    // buffering is desired.
    if (!loadInfo.sharedEncoding && !canRegisterBuffer) {
      continue;
    }
    loadToInfo[op] = loadInfo;
  }
}

void tt::LoopSchedule::scheduleLoads() {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse();
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return;

  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  assignMemoryLayouts(loadOpToIndLevelAndUse, axisInfoAnalysis);

  if (loadToInfo.empty())
    return;

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }

  // Distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
tt::CoarseSchedule::Cluster
tt::LoopSchedule::schedulePrologueAndEpilogue() {
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  tt::CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }

  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void tt::LoopSchedule::scheduleDependencies() {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void tt::LoopSchedule::scheduleDistanceOneDependencies() {
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<tt::CoarseSchedule::Cluster *, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

void
tt::LoopSchedule::scheduleRemainingToLastStage(tt::CoarseSchedule::Cluster afterPrologue) {
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        tt::CoarseSchedule::Cluster userCluster = opToCluster[user];
        tt::CoarseSchedule::Cluster opCluster = schedule[op].second;
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = mlir::triton::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, sharedMemorySpace,
      /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loadOp->getLoc(), memdescType, Value());
  return alloc;
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
void tt::LoopSchedule::createAsyncOps() {
  // Calculate the number of buffers needed for each load.
  // TODO pawel: we could do more fine-grained allocation here and
  // allocate only the number of buffers that specific loads need.
  // Instead, we allocate the maximum number of buffers needed by any load.
  int numBuffers =
      llvm::max_element(llvm::make_second_range(loadToInfo), [](auto &lhs,
                                                                auto &rhs) {
        return lhs.distToUse < rhs.distToUse;
      })->distToUse;
  bool hasMMAV3 =
      llvm::any_of(loadToInfo, [](auto &kv) { return kv.second.loadIsMMAV3; });
  if (hasMMAV3) {
    // For MMAv3, we need an extra buffer as this is assumed in the wgmma
    // pipelining post-processing.
    numBuffers++;
  };

  SmallVector<AsyncLoad> asyncLoads;
  bool hasTMALoad = false;
  for (auto &[loadOp, info] : loadToInfo) {
    if (info.sharedEncoding) {
      Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
      assert(alloc && "Failed to create alloc for the async load.");
      allocs.push_back(alloc);
      asyncLoads.emplace_back(loadOp, alloc);
      if (isa<tt::ExperimentalDescriptorLoadOp>(loadOp)) {
        hasTMALoad = true;
        asyncLoads.back().isTMALoad = true;
      }
    }
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value phase = Value();
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  if (hasTMALoad) {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    newOperands.push_back(phase);
  }
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  insertIdx = newForOp.getBody()->getArgument(newOperandIndex);
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex + 1);
  if (phase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex + 2);
  }

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
  if (phase) {
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }
  createTMABarrierAndWait(asyncLoads, insertIdx, extractIdx, phase,
                          numBuffers);

  // Create a cluster for the prefetches. It may end up being empty, but this
  // is OK.
  tt::CoarseSchedule::Cluster prefetchCluster = schedule.clusters.newAtBack();

  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(asyncLoad.loadOp)) {
      createAsyncCopy(loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      prefetchCluster);
    } else {
      auto descLoad = cast<tt::ExperimentalDescriptorLoadOp>(asyncLoad.loadOp);
      createTMAAsyncCopy(descLoad, asyncLoad.alloc, insertIdx, extractIdx,
                         asyncLoad.barrier, asyncLoad.waitOp, phase);
    }
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (phase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);
}

//bool mlir::triton::preProcessLoopAndGetSchedule(
//    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {

bool tt::LoopSchedule::compute() {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  scheduleLoads();
  if (loadToInfo.empty())
    return false;

  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    schedule.dump();
  });

  // Convert the loads into async loads and create the allocs.
  createAsyncOps();

  LLVM_DEBUG({
    LDBG("Coarse schedule with async loads:");
    schedule.dump();
  });

  tt::CoarseSchedule::Cluster afterPrologue = schedulePrologueAndEpilogue();
  LLVM_DEBUG({
    LDBG("Coarse schedule with prologue and epilogue:");
    schedule.dump();
  });

  scheduleDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    schedule.dump();
  });

  scheduleDistanceOneDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    schedule.dump();
  });

  scheduleRemainingToLastStage(afterPrologue);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    schedule.dump();
  });

  return true;
}

