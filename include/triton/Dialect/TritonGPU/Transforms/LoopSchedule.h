#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_LOOPSCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_LOOPSCHEDULE_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include <list>
#include <vector>

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {

class CoarseSchedule {
public:
  class ClusterList {
    std::list<int> orderClusters;

  public:
    using iterator = decltype(orderClusters)::iterator;
    ClusterList() = default;
    iterator begin() { return orderClusters.begin(); }
    iterator end() { return orderClusters.end(); }
    size_t size() { return orderClusters.size(); }
    iterator newAtBack() {
      orderClusters.push_back(orderClusters.size());
      return std::prev(orderClusters.end());
    }
    iterator newAtFront() {
      orderClusters.push_front(-1);
      for (auto &clusterId : orderClusters) {
        clusterId++;
      }
      return orderClusters.begin();
    }
    iterator newBefore(iterator cluster) {
      auto ret = orderClusters.insert(cluster, *cluster);
      for (auto &clusterId : llvm::make_range(cluster, orderClusters.end())) {
        clusterId++;
      }
      return ret;
    }
  };

  CoarseSchedule(int numStages) : numStages(numStages) {}
  int numStages;
  ClusterList clusters;
  using Cluster = decltype(clusters)::iterator;

  DenseMap<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void insert(Operation *op, int stage, Cluster cluster) {
    opToStageAndCluster[op] = {stage, cluster};
  }

  bool insertIfAbsent(Operation *op, int stage, Cluster cluster) {
    if (opToStageAndCluster.count(op))
      return false;
    insert(op, stage, cluster);
    return true;
  }

  void insertDepsOfOp(Operation *op, int stage, CoarseSchedule::Cluster cluster,
                      bool includeArg);

  void erase(Operation *op) { opToStageAndCluster.erase(op); }

  int count(Operation *op) { return opToStageAndCluster.count(op); }

  std::pair<int, Cluster> operator[](Operation *op) {
    return opToStageAndCluster[op];
  }

  SmallVector<std::tuple<Operation *, int, Cluster>>
  getOpsInOrder(scf::ForOp forOp);
  std::vector<std::pair<Operation *, unsigned>>
  createFinalSchedule(scf::ForOp forOp);
  void dump();
};

// Schedule encapsulation for Loops
class LoopSchedule {
protected:
  struct LoadInfo {
    // Layout of the data in the shared memory.
    ttg::SharedEncodingAttr sharedEncoding = nullptr;
    // Blocked encoding is used for loads not used by the dot.
    ttg::BlockedEncodingAttr blockedEncoding = nullptr;
    bool loadIsMMAV3 = false;
    int distToUse = 0;
    bool usedByDot = false;
  };

  struct AsyncLoad {
    AsyncLoad(Operation *loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    Operation *loadOp;
    Value alloc;
    Value barrier;
    Operation *waitOp = nullptr;
    bool isTMALoad = false;
  };

  scf::ForOp forOp;
  int numStages;
  bool canRegisterBuffer;
  CoarseSchedule schedule;
  DenseSet<Operation *> rootUsers;
  SmallVector<Value> allocs;
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

public:

  LoopSchedule(scf::ForOp forOp, int numStages, bool canRegBuf = false)
    : forOp(forOp), numStages(numStages), canRegisterBuffer(canRegBuf),
      schedule(numStages) {}

  bool compute();

  std::vector<std::pair<Operation *, unsigned>> getSchedule() {
    return schedule.createFinalSchedule(forOp);
  }

  scf::ForOp getNewForLoop() { return forOp; }
  SmallVector<Value> &getAllocs() { return allocs; }

protected:
  virtual bool isLoadOp(Operation *op);

  virtual void createAsyncCopy(LoadOp loadOp, Value alloc,
                               Value insertIdx, Value extractIdx,
                               CoarseSchedule::Cluster prefetchCluster) = 0;
  virtual void createTMAAsyncCopy(ExperimentalDescriptorLoadOp loadOp,
                                  Value alloc, Value insertIdx, Value extractIdx,
                                  Value barrier, Operation *waitOp, Value phase) = 0;
  virtual void createTMABarrierAndWait(SmallVector<AsyncLoad> &asyncLoads,
                                       Value insertIdx, Value extractIdx,
                                       Value phase, int numBuffers) = 0;

private:
  void createAsyncOps();
  void assignMemoryLayouts(
     llvm::SmallVector<std::tuple<Operation *, int, Operation *>> &loadOpToIndLevelAndUse,
     ModuleAxisInfoAnalysis &axisInfoAnalysis);
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>> loadOpsToIndirectionLevelAndUse();
  void scheduleLoads();
  CoarseSchedule::Cluster schedulePrologueAndEpilogue();
  void scheduleDependencies();
  void scheduleDistanceOneDependencies();
  void scheduleRemainingToLastStage(CoarseSchedule::Cluster afterPrologue);

};

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_LOOPSCHEDULE_H_
