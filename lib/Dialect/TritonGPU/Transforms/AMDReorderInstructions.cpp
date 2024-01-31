#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void sortOperandsByDominance(OperandRange operands,
                               SmallVector<Value> &operandsSorted) {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    for (auto operand : operands) {
      operandsSorted.push_back(operand);
    }

    if (operandsSorted.size() == 1) {
      return;
    }

    std::sort(operandsSorted.begin(), operandsSorted.end(),
              [&](const Value &a, const Value &b) {
                Operation *operandA = a.getDefiningOp();
                Operation *operandB = b.getDefiningOp();
                if (operandA && operandB) {
                  return dom.dominates(operandA, operandB);
                }
                return false;
              });
  }

  void moveAfter(Operation *lhs, Operation *rhs) {
    auto lhsId = getWSRoleId(lhs);
    auto rhsId = getWSRoleId(rhs);
    if (lhsId == rhsId)
      lhs->moveAfter(rhs);
  }

  void moveBefore(Operation *lhs, Operation *rhs) {
    auto lhsId = getWSRoleId(lhs);
    auto rhsId = getWSRoleId(rhs);
    if (lhsId == rhsId)
      lhs->moveBefore(rhs);
  }

  bool isFAChainDot(tt::DotOp &dotOp) const {
    SetVector<Operation *> slices;
    getForwardSlice((Operation *)dotOp, &slices);

    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp)) {
        auto operandA = op->getOperand(0).getDefiningOp();
        auto containsOperandA =
            std::find(slices.begin(), slices.end(), operandA) != slices.end();
        if (containsOperandA) {
          return true;
        }
      }
    }
    return false;
  }

  void moveImmediatelyAfterOperands(Operation *op,
                                    SmallVector<Operation *> &movedOperations) {

    if (std::find(movedOperations.begin(), movedOperations.end(), op) !=
        movedOperations.end()) {
      return;
    }
    auto operands = op->getOperands();
    if (operands.empty()) {
      return;
    }
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    for (auto operandVal : operands) {
      Operation *argOp = operandVal.getDefiningOp();
      if (!argOp) {
        continue;
      }
      moveImmediatelyAfterOperands(argOp, movedOperations);
    }

    SmallVector<Value> operandsSorted;
    sortOperandsByDominance(operands, operandsSorted);

    if (!operandsSorted.empty() &&
        operandsSorted[operandsSorted.size() - 1].getDefiningOp()) {

      moveAfter(op, operandsSorted[operandsSorted.size() - 1].getDefiningOp());
      if (failed(mlir::verify(m))) {
        assert(false);
      }
    }

    movedOperations.push_back(op);
  }

  void moveQTensorOutOfTheLoop(ModuleOp m) {
    m.walk([&](tt::DotOp dotOp) {
      if (isFAChainDot(dotOp)) {
        Operation *operandA = dotOp->getOperand(0).getDefiningOp();
        SmallVector<Operation *> movedOperations;
        moveImmediatelyAfterOperands(operandA, movedOperations);
        return;
      }
    });
  }

  bool contains(const SmallVector<Operation *> &vec, Operation *element) {
    return std::find(vec.begin(), vec.end(), element) != vec.end();
  }

  bool containsInAnyChain(SmallVector<SmallVector<Operation *>> dotChains,
                          Operation *element) {
    for (auto chain : dotChains) {
      if (contains(chain, element)) {
        return true;
      }
    }
    return false;
  }

  bool isLDSWrite(Operation *op) {
    auto cvtLayoutOp = dyn_cast<ttg::ConvertLayoutOp>(op);
    if (!cvtLayoutOp) {
      return false;
    }
    auto srcType = cvtLayoutOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvtLayoutOp.getResult().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    auto dstEncoding = dstType.getEncoding();
    if (srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() &&
        dstEncoding.isa<triton::gpu::SharedEncodingAttr>())
      return true;
    return false;
  }

  bool isLDSRead(Operation *op) {
    auto cvtLayoutOp = dyn_cast<ttg::ConvertLayoutOp>(op);
    if (!cvtLayoutOp) {
      return false;
    }
    auto srcType = cvtLayoutOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvtLayoutOp.getResult().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    auto dstEncoding = dstType.getEncoding();
    if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>() &&
        dstEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
      return true;
    return false;
  }

  void moveLoadStoreBeforeDot(Operation *currDot, Operation *moveBeforeDot,
                              SmallVector<Operation *> &operations,
                              int operandIdx) {
    auto operandB = currDot->getOperand(operandIdx).getDefiningOp();
    Operation *currOp = operandB;
    Operation *moveBeforeOp = moveBeforeDot;

    auto moveOp = [&](Operation *op, Operation *&opType) {
      if (opType) {
        moveAfter(op, opType);
      } else {
        moveBefore(op, moveBeforeOp);
      }
      opType = op;
    };

    for (int i = 0; !isa<ttg::ViewSliceOp>(currOp); i++) {
      moveOp(currOp, operations[i]);
      moveBeforeOp = currOp;
      currOp = currOp->getOperand(0).getDefiningOp();
    }
    moveOp(currOp, operations[operations.size() - 1]);
  }

  void initOperations(Operation *currOp, SmallVector<Operation *> &vec,
                      int operandIdx) {
    while (!isa<ttg::ViewSliceOp>(currOp)) {
      if (operandIdx == 0) {
        vec.push_back(currOp);
      } else {
        vec.push_back(nullptr);
      }
      currOp = currOp->getOperand(0).getDefiningOp();
    }
    if (operandIdx == 0) {
      vec.push_back(currOp);
    } else {
      vec.push_back(nullptr);
    }
  }

  void processStage(Operation *currDot, Operation *moveBeforeDot,
                    SmallVector<Operation *> &operations, bool init,
                    int operandIdx) {
    if (init) {
      initOperations(currDot->getOperand(operandIdx).getDefiningOp(),
                     operations, operandIdx);
      if (operandIdx == 0) {
        return;
      }
    }
    moveLoadStoreBeforeDot(currDot, moveBeforeDot, operations, operandIdx);
  }

  unsigned getNumUsers(Value value) {
    return std::distance(value.user_begin(), value.user_end());
  }

  void scheduleSlicedDot(ModuleOp m, int stages, bool sinkLDSRd, bool sinkLDSWr) {
    SmallVector<SmallVector<Operation *>> dotChains;

    m.walk([&](tt::DotOp dotOp) {
      if (!containsInAnyChain(dotChains, dotOp)) {
        SmallVector<Operation *> newChain;
        Operation *currOp = dotOp;
        newChain.push_back(currOp);

        if (getNumUsers(dotOp->getResult(0)) == 1) {
          auto user = *currOp->getUsers().begin();
          while (isa<tt::DotOp>(user)) {
            newChain.push_back(user);
            if (getNumUsers(user->getResult(0)) > 1) {
              break;
            }
            // TODO: check that  user is accumulator
            // of the dot.
            user = *user->getUsers().begin();
          }
        }
        if (newChain.size() >= 2) {
          dotChains.push_back(newChain);
        }
      }
    });

    for (auto chain : dotChains) {
      for (int i = 0; i < chain.size() / stages; i++) {
        SmallVector<Operation *> operations;
        SmallVector<Operation *> operationsIdx0;
        for (int j = 0; j < stages; j++) {
          processStage(chain[i * stages + j], chain[i], operationsIdx0, j == 0,
                       0);
          processStage(chain[i * stages + j], chain[i], operations, j == 0, 1);
        }
      }

      int startDotIdx = (chain.size() / stages) * stages;
      SmallVector<Operation *> operations;
      SmallVector<Operation *> operationsIdx0;
      for (int i = 0; i < chain.size() % stages; i++) {
        processStage(chain[startDotIdx + i], chain[chain.size() / stages],
                     operationsIdx0, i == 0, 0);
        processStage(chain[startDotIdx + i], chain[chain.size() / stages],
                     operations, i == 0, 1);
      }
    }

    if (!sinkLDSRd) {
      return;
    }

    for (auto chain : dotChains) {
      for (int i = 0; i < chain.size(); i++) {
        Operation *dotOp = chain[i];
        Operation *ldsRd = dotOp->getOperand(1).getDefiningOp();
        assert(isLDSRead(ldsRd));
        moveBefore(ldsRd, dotOp);
        if (sinkLDSWr) {
          Operation *ldsWr = ldsRd->getOperand(0).getDefiningOp();
          assert(isLDSWrite(ldsWr));
          moveBefore(ldsWr, ldsRd);
        }
      }
    }
  }

  void runOnOperation() override {
    SmallVector<Operation *> movedOperations;
    ModuleOp m = getOperation();

    moveQTensorOutOfTheLoop(m);
    int stages = 4;
    bool sinkLDSRd = true;
    bool sinkLDSWr = true;
    scheduleSlicedDot(m, stages, sinkLDSRd, sinkLDSWr);
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}

// m.walk([&](tt::DotOp dotOp) {
//   auto *operandA = dotOp.getOperand(0).getDefiningOp();
//   auto convert = dyn_cast<ttg::ConvertLayoutOp>(operandA);
//   auto srcTy = convert.getSrc().getType().cast<RankedTensorType>();
//   Attribute srcLayout = srcTy.getEncoding();

//   if (isa<ttg::MfmaEncodingAttr>(srcLayout)) {
//     Operation *currOp = operandA;
//     Operation *moveBeforeOp = dotOp;
//     while (!isa<ttg::ViewSliceOp>(currOp)) {
//       moveBefore(currOp, moveBeforeOp);
//       moveBeforeOp = currOp;
//       currOp = currOp->getOperand(0).getDefiningOp();
//     }
//     moveBefore(currOp, moveBeforeOp);
//   }
// });
