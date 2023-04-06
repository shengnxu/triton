#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Analysis/Utility.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!(isSharedEncoding(result) || isLDSEncoding(result)))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreMfmaEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!isMfmaEncoding(result))
      return op->emitOpError() << "requires all results to be Mfma encoding";

  return success();
};

mlir::LogicalResult
mlir::OpTrait::impl::verifySameOperandsAndResultNumElements(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() != 1)
    return op->emitOpError() << "requires exact 1 operand and 1 result";
  auto src = op->getOperand(0);
  auto dst = op->getResult(0);

  auto tensorTy = src.getType().dyn_cast<RankedTensorType>();
  auto memrefTy = dst.getType().dyn_cast<MemRefType>();

  if (!tensorTy || !memrefTy)
    return op->emitOpError() << "requires tensor operand and memref result";

  auto srcShape = tensorTy.getShape();
  auto dstShape = memrefTy.getShape();

  long srcNumElements = product<long>(srcShape);
  long dstNumElements = product<long>(dstShape);
  if (srcNumElements != dstNumElements)
    return op->emitOpError()
           << "requires operand and result have same number of elements";

  return success();
};
