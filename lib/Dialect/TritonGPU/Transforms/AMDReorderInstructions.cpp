#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void sortOperandsByDominance(OperandRange operands, SmallVector<Value> operandsSorted) {
    bool swapped;
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    for(auto operand: operands){
        operandsSorted.push_back(operand);
    }
    
    if(operandsSorted.size() == 1){
        return;
    }
    do {
      swapped = false;
      for (size_t i = 0; i < operandsSorted.size() - 1; i++) {
        // Check if operands[i] is more dominant than operands[i + 1]. If so,
        // swap them.
        if (dom.dominates(operandsSorted[i].getDefiningOp(),
                          operandsSorted[i + 1].getDefiningOp())) {
          // Swap operands[i] and operands[i + 1] to ensure least dominant comes
          // first
          auto tmp = operandsSorted[i];
          operandsSorted[i] = operandsSorted[i+1];
          operandsSorted[i+1] = tmp;
          swapped = true;
        }
      }
    } while (swapped);
  }

  void moveAfter(Operation *lhs, Operation *rhs) {
    auto lhsId = getWSRoleId(lhs);
    auto rhsId = getWSRoleId(rhs);
    if (lhsId == rhsId)
      lhs->moveAfter(rhs);
  }

  void moveImmediatelyAfterOperands(Operation *op,
                                    SmallVector<Operation *> movedOperations) {

    if (std::find(movedOperations.begin(), movedOperations.end(), op) !=
        movedOperations.end()) {
      return;
    }
    auto operands = op->getOperands();
    if(operands.empty()){
        return;
    }
    SmallVector<Value> operandsSorted;
    sortOperandsByDominance(operands, operandsSorted);

    for (auto operandVal : operandsSorted) {
      Operation *argOp = operandVal.getDefiningOp();
      moveImmediatelyAfterOperands(argOp, movedOperations);
    }
    moveAfter(op, operandsSorted[operandsSorted.size() - 1].getDefiningOp());
    movedOperations.push_back(op);
  }

  void runOnOperation() override {
    SmallVector<Operation *> movedOperations;
    ModuleOp m = getOperation();

    m.walk([&](Operation *op) {
        moveImmediatelyAfterOperands(op, movedOperations);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
