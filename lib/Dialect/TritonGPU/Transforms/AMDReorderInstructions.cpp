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

static int num = 0;
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
    if(num <= 7){
      num += 1;
      return true;
    }else{
      return false;
    }
    SetVector<Operation *> slices;
    getForwardSlice((Operation *)dotOp, &slices);

    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp)) {
        auto operandA = op->getOperand(0).getDefiningOp();
        auto containsOperandA =
            std::find(slices.begin(), slices.end(), operandA) != slices.end();
        if (containsOperandA) {
          num += 1;
          return true;
        }
      }
    }
    num += 1;
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

    // FIX THIS TO WORK PROPERLY:

    // for (auto chain : dotChains) {
    //   for (int i = 0; i < (chain.size() - 1) / (stages - 1); i++) {
    //     SmallVector<Operation *> operations;
    //     SmallVector<Operation *> operationsIdx0;
    //     for (int j = 0; j < stages - 1; j++) {
    //       processStage(chain[i * stages + j], chain[i], operationsIdx0, j == 0,
    //                    0);
    //       processStage(chain[i * stages + j], chain[i], operations, j == 0, 1);
    //     }
    //   }

    //   int startDotIdx = (chain.size() / stages) * stages;
    //   SmallVector<Operation *> operations;
    //   SmallVector<Operation *> operationsIdx0;
    //   for (int i = 0; i < chain.size() % stages; i++) {
    //     processStage(chain[startDotIdx + i], chain[chain.size() / stages],
    //                  operationsIdx0, i == 0, 0);
    //     processStage(chain[startDotIdx + i], chain[chain.size() / stages],
    //                  operations, i == 0, 1);
    //   }
    // }
    // for (auto chain : dotChains) {
    //   SmallVector<Operation *> operations;
    //   SmallVector<Operation *> operationsIdx0;
    //   for (int i = 0; i < chain.size(); i++) {
    //     auto prevIdx = i - 1;
    //     if(prevIdx < 0){
    //       prevIdx = 0;
    //     }
    //     bool shouldInit = (i % 2 == 0);
    //     if (shouldInit) {
    //       operations.clear();
    //       operationsIdx0.clear();
    //     }
    //     processStage(chain[i], chain[prevIdx], operationsIdx0, shouldInit, 0);
    //     processStage(chain[i], chain[prevIdx], operations, shouldInit, 1);
    //   }
    // }

    // if (!sinkLDSRd) {
    //   return;
    // }

    // for (auto chain : dotChains) {
    //   for (int i = 0; i < chain.size(); i++) {
    //     Operation *dotOp = chain[i];
    //     Operation *ldsRd = dotOp->getOperand(1).getDefiningOp();
    //     assert(isLDSRead(ldsRd));
    //     moveBefore(ldsRd, dotOp);
    //     if (sinkLDSWr) {
    //       Operation *ldsWr = ldsRd->getOperand(0).getDefiningOp();
    //       assert(isLDSWrite(ldsWr));
    //       moveBefore(ldsWr, ldsRd);
    //     }
    //   }
    // }

    for (int i = 0; i < 3; i++) {
      Operation *firstDotFirstGEMM = dotChains[i][0];
      // firstDotSecondGEMM->dump();
      Operation *currOp = firstDotFirstGEMM->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOp)) {
        currOp = currOp->getOperand(0).getDefiningOp();
      }
      Operation *loadOp = currOp;
      Operation *ldsRead = firstDotFirstGEMM->getOperand(1).getDefiningOp();
      Operation *ldsWrite = ldsRead->getOperand(0).getDefiningOp();
      Operation *viewSlice = loadOp->getOperand(0).getDefiningOp();

      Operation *firstDotFirstGEMM2 = dotChains[i][1];
      // firstDotSecondGEMM->dump();
      Operation *currOp2 = firstDotFirstGEMM2->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOp2)) {
        currOp2 = currOp2->getOperand(0).getDefiningOp();
      }
      Operation *ldsRead2 = firstDotFirstGEMM2->getOperand(1).getDefiningOp();
      Operation *ldsWrite2 = ldsRead2->getOperand(0).getDefiningOp();
      Operation *loadOp2 = currOp2;
      Operation *viewSlice2 = currOp2->getOperand(0).getDefiningOp();
      // moveAfter(ldsWrite, loadOp);
      // moveAfter(loadOp2, ldsWrite);
      // moveAfter(ldsWrite2, loadOp2);

      Operation *firstDotFirstGEMM3 = dotChains[i][2];
      // firstDotSecondGEMM->dump();
      Operation *currOp3 = firstDotFirstGEMM3->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOp3)) {
        currOp3 = currOp3->getOperand(0).getDefiningOp();
      }
      Operation *ldsRead3 = firstDotFirstGEMM3->getOperand(1).getDefiningOp();
      Operation *ldsWrite3 = ldsRead3->getOperand(0).getDefiningOp();
      Operation *loadOp3 = currOp3;
      Operation *viewSlice3 = currOp3->getOperand(0).getDefiningOp();

      Operation *firstDotFirstGEMM4 = dotChains[i][3];
      // firstDotSecondGEMM->dump();
      Operation *currOp4 = firstDotFirstGEMM4->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOp4)) {
        currOp4 = currOp4->getOperand(0).getDefiningOp();
      }
      Operation *ldsRead4 = firstDotFirstGEMM4->getOperand(1).getDefiningOp();
      Operation *ldsWrite4 = ldsRead4->getOperand(0).getDefiningOp();
      Operation *loadOp4 = currOp4;
      Operation *viewSlice4 = currOp4->getOperand(0).getDefiningOp();
      moveAfter(loadOp2, loadOp);
      moveAfter(ldsWrite, loadOp2);
      moveAfter(ldsRead, ldsWrite);

      moveAfter(ldsWrite2, firstDotFirstGEMM);
      moveAfter(loadOp3, ldsWrite2);
      moveAfter(ldsRead2, loadOp3);

      moveAfter(ldsWrite3, firstDotFirstGEMM2);
      moveAfter(loadOp4, ldsWrite3);
      moveAfter(ldsRead3, loadOp4);

      moveAfter(ldsWrite4, firstDotFirstGEMM3);
      moveAfter(ldsRead4, ldsWrite4);

      moveBefore(viewSlice, loadOp);
      moveBefore(viewSlice2, loadOp2);
      moveBefore(viewSlice3, loadOp3);
      moveBefore(viewSlice4, loadOp4);

    }


    for (int i = 0; i < 1; i++) {
      Operation *firstDotFirstGEMM = dotChains[i][0];
      Operation *currOpK = firstDotFirstGEMM->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpK)) {
        currOpK = currOpK->getOperand(0).getDefiningOp();
      }
      Operation *loadOpK = currOpK;
      Operation *ldsReadK = firstDotFirstGEMM->getOperand(1).getDefiningOp();
      Operation *ldsWriteK = ldsReadK->getOperand(0).getDefiningOp();
      Operation *viewSliceK = loadOpK->getOperand(0).getDefiningOp();

      Operation *currOpQ = firstDotFirstGEMM->getOperand(0).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpQ)) {
        currOpQ = currOpQ->getOperand(0).getDefiningOp();
      }
      Operation *loadOpQ = currOpQ;
      Operation *viewSliceQ = loadOpQ->getOperand(0).getDefiningOp();
      Operation *ldsReadQ = firstDotFirstGEMM->getOperand(0).getDefiningOp();
      Operation *ldsWriteQ = ldsReadQ->getOperand(0).getDefiningOp();
      Operation *truncf = ldsWriteQ->getOperand(0).getDefiningOp();
      Operation *mulf = truncf->getOperand(0).getDefiningOp();
      Operation *extf = mulf->getOperand(0).getDefiningOp();


      Operation *firstDotFirstGEMM2 = dotChains[i][1];
      Operation *currOpK2 = firstDotFirstGEMM2->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpK2)) {
        currOpK2 = currOpK2->getOperand(0).getDefiningOp();
      }
      Operation *loadOpK2 = currOpK2;
      Operation *ldsReadK2 = firstDotFirstGEMM2->getOperand(1).getDefiningOp();
      Operation *ldsWriteK2 = ldsReadK2->getOperand(0).getDefiningOp();
      Operation *viewSliceK2 = loadOpK2->getOperand(0).getDefiningOp();

      Operation *currOpQ2 = firstDotFirstGEMM2->getOperand(0).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpQ2)) {
        currOpQ2 = currOpQ2->getOperand(0).getDefiningOp();
      }
      Operation *loadOpQ2 = currOpQ2;
      Operation *viewSliceQ2 = loadOpQ2->getOperand(0).getDefiningOp();
      Operation *ldsReadQ2 = firstDotFirstGEMM2->getOperand(0).getDefiningOp();
      Operation *ldsWriteQ2 = ldsReadQ2->getOperand(0).getDefiningOp();
      Operation *truncf2 = ldsWriteQ2->getOperand(0).getDefiningOp();
      Operation *mulf2 = truncf2->getOperand(0).getDefiningOp();
      Operation *extf2 = mulf2->getOperand(0).getDefiningOp();


      Operation *firstDotFirstGEMM3 = dotChains[i][2];
      Operation *currOpK3 = firstDotFirstGEMM3->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpK3)) {
        currOpK3 = currOpK3->getOperand(0).getDefiningOp();
      }
      Operation *loadOpK3 = currOpK3;
      Operation *ldsReadK3 = firstDotFirstGEMM3->getOperand(1).getDefiningOp();
      Operation *ldsWriteK3 = ldsReadK3->getOperand(0).getDefiningOp();
      Operation *viewSliceK3 = loadOpK3->getOperand(0).getDefiningOp();

      Operation *currOpQ3 = firstDotFirstGEMM3->getOperand(0).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpQ3)) {
        currOpQ3 = currOpQ3->getOperand(0).getDefiningOp();
      }
      Operation *loadOpQ3 = currOpQ3;
      Operation *viewSliceQ3 = loadOpQ3->getOperand(0).getDefiningOp();
      Operation *ldsReadQ3 = firstDotFirstGEMM3->getOperand(0).getDefiningOp();
      Operation *ldsWriteQ3 = ldsReadQ3->getOperand(0).getDefiningOp();
      Operation *truncf3 = ldsWriteQ3->getOperand(0).getDefiningOp();
      Operation *mulf3 = truncf3->getOperand(0).getDefiningOp();
      Operation *extf3 = mulf3->getOperand(0).getDefiningOp();


      Operation *firstDotFirstGEMM4 = dotChains[i][3];
      Operation *currOpK4 = firstDotFirstGEMM4->getOperand(1).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpK4)) {
        currOpK4 = currOpK4->getOperand(0).getDefiningOp();
      }
      Operation *loadOpK4 = currOpK4;
      Operation *ldsReadK4 = firstDotFirstGEMM4->getOperand(1).getDefiningOp();
      Operation *ldsWriteK4 = ldsReadK4->getOperand(0).getDefiningOp();
      Operation *viewSliceK4 = loadOpK4->getOperand(0).getDefiningOp();

      Operation *currOpQ4 = firstDotFirstGEMM4->getOperand(0).getDefiningOp();
      while (!isa<tt::LoadOp>(currOpQ4)) {
        currOpQ4 = currOpQ4->getOperand(0).getDefiningOp();
      }
      Operation *loadOpQ4 = currOpQ4;
      Operation *viewSliceQ4 = loadOpQ4->getOperand(0).getDefiningOp();
      Operation *ldsReadQ4 = firstDotFirstGEMM4->getOperand(0).getDefiningOp();
      Operation *ldsWriteQ4 = ldsReadQ4->getOperand(0).getDefiningOp();
      Operation *truncf4 = ldsWriteQ4->getOperand(0).getDefiningOp();
      Operation *mulf4 = truncf4->getOperand(0).getDefiningOp();
      Operation *extf4 = mulf4->getOperand(0).getDefiningOp();


      moveAfter(loadOpQ, loadOpK);
      moveAfter(extf, loadOpQ);
      moveAfter(mulf, extf);
      moveAfter(truncf, mulf);

      moveAfter(loadOpQ2, loadOpK2);
      moveAfter(extf2, loadOpQ2);
      moveAfter(mulf2, extf2);
      moveAfter(truncf2, mulf2);

      moveAfter(loadOpQ3, loadOpK3);
      moveAfter(extf3, loadOpQ3);
      moveAfter(mulf3, extf3);
      moveAfter(truncf3, mulf3);

      moveAfter(loadOpQ4, loadOpK4);
      moveAfter(extf4, loadOpQ4);
      moveAfter(mulf4, extf4);
      moveAfter(truncf4, mulf4);

      moveAfter(ldsWriteQ, ldsWriteK);
      moveAfter(ldsReadQ, ldsReadK);

      moveAfter(ldsWriteQ2, ldsWriteK2);
      moveAfter(ldsReadQ2, ldsReadK2);

      moveAfter(ldsWriteQ3, ldsWriteK3);
      moveAfter(ldsReadQ3, ldsReadK3);

      moveAfter(ldsWriteQ4, ldsWriteK4);
      moveAfter(ldsReadQ4, ldsReadK4);
    }
  }

  void runOnOperation() override {
    SmallVector<Operation *> movedOperations;
    ModuleOp m = getOperation();
    if(num == 0){
      moveQTensorOutOfTheLoop(m);
    }
    int stages = 2;
    bool sinkLDSRd = true;
    bool sinkLDSWr = true;
    scheduleSlicedDot(m, stages, sinkLDSRd, sinkLDSWr);
    m.dump();
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
