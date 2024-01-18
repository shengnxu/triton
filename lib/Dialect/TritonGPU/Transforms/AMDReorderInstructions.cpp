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
#include <algorithm>
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
                if (a.getDefiningOp() && b.getDefiningOp()) {
                  return dom.dominates(a.getDefiningOp(), b.getDefiningOp());
                }
                if (!a.getDefiningOp() && b.getDefiningOp())
                  return dom.dominates(a, b.getDefiningOp());
                if (!b.getDefiningOp() && a.getDefiningOp())
                  return !dom.dominates(b, a.getDefiningOp());
                return true;
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

  bool isElementwiseOp(Operation *op) {
    if (llvm::isa<
            arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CeilDivSIOp,
            arith::CeilDivUIOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp,
            arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FloorDivSIOp,
            arith::FPToSIOp, arith::FPToUIOp, arith::MaximumFOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp,
            arith::MulFOp, arith::MulIOp, arith::NegFOp, arith::OrIOp,
            arith::RemFOp, arith::RemSIOp, arith::RemUIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::SIToFPOp, arith::SubFOp,
            arith::SubIOp, arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp,
            arith::XOrIOp>(op))
      return true;
    if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                  math::CeilOp, math::CopySignOp, math::CosOp, math::SinOp,
                  math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                  math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                  math::ExpM1Op, math::FloorOp, math::FmaOp, math::LogOp,
                  math::Log10Op, math::Log1pOp, math::Log2Op, math::PowFOp,
                  math::RsqrtOp, math::SqrtOp, math::TanhOp>(op))
      return true;
    if (llvm::isa<tt::IntToPtrOp, tt::PtrToIntOp, tt::BitcastOp, tt::FpToFpOp,
                  tt::AddPtrOp>(op))
      return true;
    if (auto externElementwiseOp = dyn_cast<tt::ExternElementwiseOp>(op))
      return externElementwiseOp.getPure();
    if (llvm::isa<arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op))
      return true;
    if (isa<ttg::ConvertLayoutOp, tt::LoadOp, tt::DotOp, tt::SplatOp, ttg::ViewSliceOp>(
            op)) {
      return true;
    }

    // if (auto vsop = dyn_cast<ttg::ViewSliceOp>(op)) {
    //   auto mulfOp = vsop.getOperand(0).getDefiningOp();
    //   return isa<tt::SplatOp>(mulfOp);
    // }

    return false;
  }

  void moveImmediatelyAfterOperands(Operation *op,
                                    SmallVector<Operation *> &movedOperations) {

    if (!isElementwiseOp(op)) {
      movedOperations.push_back(op);
    }

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

    SmallVector<Value> operandsSorted;
    sortOperandsByDominance(operands, operandsSorted);

    for (auto operandVal : operandsSorted) {
      Operation *argOp = operandVal.getDefiningOp();
      if (argOp) {
        moveImmediatelyAfterOperands(argOp, movedOperations);
      }
    }

    if (!operandsSorted.empty() &&
        operandsSorted[operandsSorted.size() - 1].getDefiningOp()) {
      // op->dump();
      // operandsSorted[operandsSorted.size()-1].getDefiningOp()->dump();

      moveAfter(op, operandsSorted[operandsSorted.size() - 1].getDefiningOp());
      if (failed(mlir::verify(m))) {
        op->dump();
        operandsSorted[operandsSorted.size() - 1].getDefiningOp()->dump();
        m.dump();
        assert(false);
      }
    }

    movedOperations.push_back(op);
  }

  void scheduleDot(ModuleOp &m) {
    m.walk([&](tt::DotOp dotOp) {
      auto *operandB = dotOp.getOperand(1).getDefiningOp();

      Operation *currOp = operandB;
      Operation *moveBeforeOp = dotOp;
      while (moveBeforeOp && currOp && !isa<ttg::ViewSliceOp>(currOp)) {
        moveBefore(currOp, moveBeforeOp);
        moveBeforeOp = currOp;
        currOp = currOp->getOperand(0).getDefiningOp();
      }
      if(currOp && moveBeforeOp)
        moveBefore(currOp, moveBeforeOp);

    });

    m.walk([&](tt::DotOp dotOp) {
      auto *operandA = dotOp.getOperand(0).getDefiningOp();
      auto convert = dyn_cast<ttg::ConvertLayoutOp>(operandA);
      auto srcTy = convert.getSrc().getType().cast<RankedTensorType>();
      Attribute srcLayout = srcTy.getEncoding();

      if (isa<ttg::MfmaEncodingAttr>(srcLayout)) {
        Operation *currOp = operandA;
        Operation *moveBeforeOp = dotOp;
        while (!isa<ttg::ViewSliceOp>(currOp)) {
          moveBefore(currOp, moveBeforeOp);
          moveBeforeOp = currOp;
          currOp = currOp->getOperand(0).getDefiningOp();
        }
        moveBefore(currOp, moveBeforeOp);
      }
    });
  }

  void scheduleViewSlice(ModuleOp &m) {
    llvm::DenseMap<Operation *, llvm::SmallVector<ttg::ViewSliceOp>> operMap;
    m.walk([&](ttg::ViewSliceOp viewSLiceOp) {
      auto *rootOperation = viewSLiceOp.getOperand(0).getDefiningOp();
      operMap[rootOperation].push_back(viewSLiceOp);
    });

    for (auto &element : operMap) {
      Operation *rootOperation = element.first;
      SmallVector<ttg::ViewSliceOp> viewSliceVec = element.second;

      std::sort(viewSliceVec.begin(), viewSliceVec.end(),
                [&](ttg::ViewSliceOp &a, ttg::ViewSliceOp &b) {
                  return a.getStaticOffsets()[1] < b.getStaticOffsets()[1];
                });

      auto moveAfterOp = rootOperation;
      for (auto vsop : viewSliceVec) {
        moveAfter(vsop, moveAfterOp);
        moveAfterOp = (Operation *)vsop;
      }
    }
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

  // A lot of assumptions about IR structure made here :)
  void hideLoadLatency(ModuleOp m, int load_lat_stages) {
    SmallVector<SmallVector<Operation *> > dotChains;

    m.walk([&](tt::DotOp dotOp) {
      if(!containsInAnyChain(dotChains, dotOp)){
        SmallVector<Operation *> newChain;
        Operation *currOp = dotOp;
        newChain.push_back(currOp);
        auto user = *currOp->getUsers().begin();

        while (isa<tt::DotOp>(user)) {
          newChain.push_back(user);
          user = *user->getUsers().begin();
        }
        dotChains.push_back(newChain);
      }
    });

    for (auto chain : dotChains) {
        Operation *currLoad = nullptr;
      for (int i = 0; i < chain.size(); i++) {
        auto currDot = chain[i];
        auto operandB = currDot->getOperand(1).getDefiningOp();
        Operation *currOp = operandB;
        int dotIdx = i > 0 ? i - 1 : 0;
        Operation *moveBeforeOp =
            i < 4 ? chain[dotIdx] : chain[i];
        while (currOp && !isa<ttg::ViewSliceOp>(currOp)) {
          moveBefore(currOp, moveBeforeOp);
          moveBeforeOp = currOp;
          currOp = currOp->getOperand(0).getDefiningOp();
        }
        // if(i == 0){
        //   currLoad = *currOp->getUsers().begin();
        // }
        if(currOp && i > 0 && i != 2){
          auto currUser = *currOp->getUsers().begin();
          moveAfter(currOp, currLoad);
          moveAfter(currUser, currOp);
        }
        currLoad = *currOp->getUsers().begin();
      }
    }
  }

  void runOnOperation() override {
    SmallVector<Operation *> movedOperations;
    ModuleOp m = getOperation();

    m.dump();
    // scheduleViewSlice(m);
    m.walk([&](Operation *op) {
      moveImmediatelyAfterOperands(op, movedOperations);
    });

    int load_lat_stages = 2;
    m.walk([&](tt::DotOp dotOp) {
      auto *operandA = dotOp.getOperand(0).getDefiningOp();
      auto convert = dyn_cast<ttg::ConvertLayoutOp>(operandA);
      auto srcTy = convert.getSrc().getType().cast<RankedTensorType>();
      Attribute srcLayout = srcTy.getEncoding();

      if (isa<ttg::MfmaEncodingAttr>(srcLayout)) {
        Operation *currOp = operandA;
        Operation *moveBeforeOp = dotOp;
        while (!isa<ttg::ViewSliceOp>(currOp)) {
          moveBefore(currOp, moveBeforeOp);
          moveBeforeOp = currOp;
          currOp = currOp->getOperand(0).getDefiningOp();
        }
        moveBefore(currOp, moveBeforeOp);
      }
    });

    hideLoadLatency(m, load_lat_stages);
    // hideLDSRdLatency(lds_rd_lat_stages);
    // hideLDSWtLatency(lds_wr_lat_stages);








    // mlir::DominanceInfo dom(m);
    // scheduleDot(m);
    // scheduleViewSlice(m);

    // m.walk([&](tt::LoadOp loadOp) {
    //   auto *viewSlice = loadOp.getOperand(0).getDefiningOp();
    //   moveAfter(loadOp, viewSlice);
    // });
    // m.walk([&](ttg::ConvertLayoutOp cvtLayoutOp) {
    //   auto *viewSlice = cvtLayoutOp->getOperand(0).getDefiningOp();
    //   moveAfter(cvtLayoutOp, viewSlice);
    // });

    // m.dump();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
