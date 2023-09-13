#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

LogicalResult fixupLoops(ModuleOp mod);

// TODO: Interface
LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                             Attribute &ret);

bool isExpensiveLoadOrStore(Operation *op, Attribute &targetEncoding);

std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding);
std::optional<Attribute> inferDstEncoding(Operation *op, Attribute encoding);
bool isExpensiveLoadOrStore(Operation *op);

bool isExpensiveToRemat(Operation *op, Attribute &targetEncoding);

// skipInit is True when we only consider the operands of the initOp but
// not the initOp itself.
int simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    Attribute targetEncoding);

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping);

void rematerializeConversionChain(
    const llvm::MapVector<Value, Attribute> &toConvert,
    mlir::PatternRewriter &rewriter, SetVector<Operation *> &processed,
    IRMapping &mapping);

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding);

LogicalResult canMoveOutOfLoop(BlockArgument arg,
                               SmallVector<Operation *> &cvts);

LogicalResult
getConvertBackwardSlice(Value root, SetVector<Value> &slice,
                        Attribute rootEncoding,
                        DenseMap<Value, Attribute> &layout,
                        std::function<bool(Operation *)> stopPropagation);

void populateForOpDeadArgumentElimination(RewritePatternSet &patterns);

} // namespace mlir


#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
