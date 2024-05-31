#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap,
                       bool notTransposed) {
    Value ptr = getMemAccessPtr(op);
    auto refTensorType = cast<RankedTensorType>(ptr.getType());

    LDBG("Considering op: " << *op);
    LLVM_DEBUG({
      DBGS() << "axis info of pointer: ";
      axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = argSort(contiguity);
    LDBG("order=[" << triton::join(order, ", ") << "]");

    auto matchesShape = [&refTensorType](const Value &val) {
      auto rttType = dyn_cast<RankedTensorType>(val.getType());
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility among all dependent
    // pointers which have the same shape and order as `ptr`.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder =
            argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread = getNumElementsPerThread(op, order, axisInfoAnalysis);
    LDBG("perThread for op: " << perThread);

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread =
          getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    LDBG("perThread: " << perThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      unsigned elemNumBits = getElementBitWidth(refTensorType);
      perThread = std::min<int>(
          perThread, getNumElementsPerThread(op, order, axisInfoAnalysis));
    }
    SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;
    llvm::outs() << "Encoding Order: " << sizePerThread[0] << " " << sizePerThread[1] << "\n";
    if (notTransposed)
      sizePerThread[order[1]] = 4;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  bool checkTransposition(Value ptr) {
    SmallVector<unsigned int> optimalOrderA = {1, 0};
    SmallVector<unsigned int> optimalOrderB = {0, 1};

    LDBG("Checking this op for transposition: " << ptr);
    // the memAccessPtr can be load/store or addptr
    auto memOps = ptr.getUsers();
    for (auto memOp : memOps) {
      // only start tracking down dotOp from a
      if (!isa<triton::LoadOp>(memOp))
        continue;
      // this assumes the load -> ConvertLayoutOp -> dot chain
      auto cvtOps = memOp->getUsers();
      for (auto cvtOp : cvtOps) {
        // here should be only 1 ConvertLayoutOp presumably
        if (!isa<triton::gpu::ConvertLayoutOp>(cvtOp))
          continue;
        LDBG("cvtOp: " << *cvtOp);
        auto dots = cvtOp->getUsers();
        // here should be only 1 dotOp presumably, but multiple are fine
        for (auto dot : dots) {
          if (!isa<triton::DotOp>(dot))
            continue;
          auto dotOp = cast<triton::DotOp>(dot);
          auto opA = dotOp.getOperand(0);
          auto opB = dotOp.getOperand(1);
          if (cvtOp->getResult(0) == opA) {
            LDBG("This is opA");
            auto opAEncoding = cast<RankedTensorType>(opA.getType()).getEncoding();
            auto orderA = triton::gpu::getOrder(opAEncoding);
            LDBG("Order A: " << orderA[0] << " " << orderA[1]);
            if (orderA != optimalOrderA) {
              llvm::outs() << "opB is not transposed\n";
              return true;
            }
          }
          else if (cvtOp->getResult(0) == opB) {
            LDBG("This is opB");
            auto opBEncoding = cast<RankedTensorType>(opB.getType()).getEncoding();
            auto orderB = triton::gpu::getOrder(opBEncoding);
            LDBG("Order B: " << orderB[0] << " " << orderB[1]);
            if (orderB != optimalOrderB) {
              llvm::outs() << "opB is not transposed\n";
              return true;
            }
          }
          else {
            return false;
          }
        }
      }
    }
    return false;
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` or `tt.ptr<tensor<>>` load/store
      bool isPtrTensor = false, isTensorPointer = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (auto ptrType = dyn_cast<PointerType>(ptr.getType()))
        isTensorPointer = isa<RankedTensorType>(ptrType.getPointeeType());
      if (!isPtrTensor && !isTensorPointer)
        return;
      bool notTransposed = checkTransposition(ptr);
      auto mod = curr->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap, notTransposed);
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      coalesceOp(kv.second, kv.first);
    }
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createCoalescePass() {
  return std::make_unique<CoalescePass>();
}
