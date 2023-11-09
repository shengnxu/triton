#include "triton/Conversion/TritonToTritonGPUROCM/TritonToTritonGPUPass.h"

#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/AnalysisROCM/Utility.h"
#include "triton/Dialect/TritonROCM/IR/Dialect.h"
#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/TritonGPUConversion.h"
#include "triton/Target/PTXROCM/TmaMetadata.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton_rocm;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToTritonGPUROCM/Passes.h.inc"

namespace {

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithShLIPattern : public OpConversionPattern<arith::ShLIOp> {
public:
  using OpConversionPattern<mlir::arith::ShLIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    mlir::Type elementType = mlir::getElementTypeOrSelf(lhs.getType());
    unsigned typeWidth = elementType.getIntOrFloatBitWidth();
    auto constValue = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, typeWidth, elementType);
    auto zeroConst =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, elementType);

    if (lhs.getType().isIntOrIndex()) {
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, constValue);
      auto shiftValue =
          rewriter.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, zeroConst);
      rewriter.replaceOp(op, {selectOp.getResult()});
    } else {
      auto splatValue = rewriter.create<mlir::tensor::SplatOp>(
          loc, lhs.getType(), constValue);
      auto zeroValue = rewriter.create<mlir::tensor::SplatOp>(
          loc, lhs.getType(), zeroConst);
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, splatValue);
      auto shiftValue =
          rewriter.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, zeroValue);
      rewriter.replaceOp(op, {selectOp.getResult()});
    }
    return success();
  }
};

class ArithShRUIOpattern : public OpConversionPattern<arith::ShRUIOp> {
public:
  using OpConversionPattern<mlir::arith::ShRUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::arith::ShRUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    mlir::Type elementType = mlir::getElementTypeOrSelf(lhs.getType());
    unsigned typeWidth = elementType.getIntOrFloatBitWidth();
    auto constValue = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, typeWidth, elementType);
    auto zeroConst =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, elementType);

    if (lhs.getType().isIntOrIndex()) {
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, constValue);
      auto shiftValue =
          rewriter.create<mlir::arith::ShRUIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, zeroConst);
      rewriter.replaceOp(op, {selectOp.getResult()});
    } else {
      auto splatValue = rewriter.create<mlir::tensor::SplatOp>(
          loc, lhs.getType(), constValue);
      auto zeroValue = rewriter.create<mlir::tensor::SplatOp>(
          loc, lhs.getType(), zeroConst);
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, splatValue);
      auto shiftValue =
          rewriter.create<mlir::arith::ShRUIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, zeroValue);
      rewriter.replaceOp(op, {selectOp.getResult()});
    }
    return success();
  }
};

class ArithShRSIOpPattern : public OpConversionPattern<arith::ShRSIOp> {
public:
  using OpConversionPattern<mlir::arith::ShRSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::arith::ShRSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    mlir::Type elementType = getElementTypeOrSelf(lhs.getType());
    unsigned typeWidth = elementType.getIntOrFloatBitWidth();
    auto constValue =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, typeWidth, elementType);
    auto zeroConst =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, elementType);
    uint64_t ones_val = 0xFFFFFFFFFFFFFFFF;
    auto onesConst =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, ones_val, elementType);

    if (lhs.getType().isIntOrIndex()) {
      auto negativeCmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, lhs, zeroConst);
      auto otherValue = mlir::Value(rewriter.create<mlir::arith::SelectOp>(
          loc, negativeCmpValue, onesConst, zeroConst));
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, constValue);
      auto shiftValue =
          rewriter.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, otherValue);
      rewriter.replaceOp(op, {selectOp.getResult()});
    } else {
      auto splatValue = rewriter.create<mlir::tensor::SplatOp>(
          loc, lhs.getType(), constValue);
      auto zeroValue =
          rewriter.create<mlir::tensor::SplatOp>(loc, lhs.getType(), zeroConst);
      auto onesValue =
          rewriter.create<mlir::tensor::SplatOp>(loc, lhs.getType(), onesConst);
      auto negativeCmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, lhs, zeroValue);
      auto otherValue = mlir::Value(rewriter.create<mlir::arith::SelectOp>(
          loc, negativeCmpValue, onesValue, zeroValue));
      auto cmpValue = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, rhs, splatValue);
      auto shiftValue = rewriter.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
      auto selectOp = rewriter.create<mlir::arith::SelectOp>(
          loc, cmpValue, shiftValue, otherValue);
      rewriter.replaceOp(op, {selectOp.getResult()});
    }
    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = retType.cast<ShapedType>();
    auto value = adaptor.getValue().dyn_cast<DenseElementsAttr>();
    if (dyn_cast<RankedTensorType>(retShapedType)) {
      assert(value);
      if (value.getElementType().isInteger(1) && value.isSplat())
        // Workaround until https://reviews.llvm.org/D133743 is included.
        value =
            DenseElementsAttr::get(retShapedType, value.getSplatValue<bool>());
      else
        // This is a hack. We just want to add encoding
        value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUROCMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUROCMConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      // Shift ops
      ArithShLIPattern,
      ArithShRUIOpattern,
      ArithShRSIOpPattern, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxSIOp>,
      GenericOpPattern<arith::MaxUIOp>, GenericOpPattern<arith::MinimumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonGPUROCMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUROCMConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::CosOp>,
               GenericOpPattern<math::SinOp>, GenericOpPattern<math::LogOp>,
               GenericOpPattern<math::AbsFOp>, GenericOpPattern<math::AbsIOp>,
               GenericOpPattern<math::SqrtOp>>(typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton_rocm::ExpandDimsOp> {
  using OpConversionPattern<triton_rocm::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        adaptor.getSrc().getType().cast<RankedTensorType>();
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = _argEncoding.cast<triton_rocm::gpu_rocm::BlockedEncodingAttr>();
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    // return encoding
    auto retSizePerThread = argEncoding.getSizePerThread().vec();
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = argEncoding.getThreadsPerWarp().vec();
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = argEncoding.getWarpsPerCTA().vec();
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto argCTALayout = argEncoding.getCTALayout();
    auto retCTAsPerCGA = insertOne(argCTALayout.getCTAsPerCGA(), op.getAxis());
    auto retCTASplitNum =
        insertOne(argCTALayout.getCTASplitNum(), op.getAxis());
    auto retCTAOrder = insertOrder(argCTALayout.getCTAOrder(), op.getAxis());
    auto retCTALayout = triton_rocm::gpu_rocm::CTALayoutAttr::get(
        getContext(), retCTAsPerCGA, retCTASplitNum, retCTAOrder);

    triton_rocm::gpu_rocm::BlockedEncodingAttr retEncoding =
        triton_rocm::gpu_rocm::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCTALayout);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton_rocm::gpu_rocm::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton_rocm::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton_rocm::DotOp> {
  using OpConversionPattern<triton_rocm::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType().cast<RankedTensorType>();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUROCMTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();

    SmallVector<unsigned> retSizePerThread = {1, 1};
    if (origShape[0] * origShape[1] / (numWarps * threadsPerWarp) >= 4)
      retSizePerThread = {2, 2};
    if (origShape[0] * origShape[1] / (numWarps * threadsPerWarp) >= 16)
      retSizePerThread = {4, 4};
    SmallVector<unsigned> retOrder = {1, 0};
    Attribute dEncoding = triton_rocm::gpu_rocm::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType =
        RankedTensorType::get(origShape, origType.getElementType(), dEncoding);
    // a & b must be of smem layout
    auto aType = adaptor.getA().getType().cast<RankedTensorType>();
    auto bType = adaptor.getB().getType().cast<RankedTensorType>();
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!aEncoding.isa<triton_rocm::gpu_rocm::DotOperandEncodingAttr>()) {
      Attribute encoding = triton_rocm::gpu_rocm::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType =
          RankedTensorType::get(aType.getShape(), aEltType, encoding);
      a = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!bEncoding.isa<triton_rocm::gpu_rocm::DotOperandEncodingAttr>()) {
      Attribute encoding = triton_rocm::gpu_rocm::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType =
          RankedTensorType::get(bType.getShape(), bEltType, encoding);
      b = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    c = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(c.getLoc(), retType, c);

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton_rocm::DotOp>(
                      op, retType, a, b, c, adaptor.getAllowTF32(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton_rocm::CatOp> {

  using OpConversionPattern<triton_rocm::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The cat op satisfy two conditions:
    // 1. output.numel = lhs.numel + rhs.numel
    // 2. output.total_elems_per_thread =
    // next_power_of_2(lhs.total_elems_per_thread + rhs.total_elems_per_thread)
    // For now, this behaves like generic, but this
    // will evolve when we add support for `can_reorder=False`.
    auto retType = this->getTypeConverter()
                       ->convertType(op.getType())
                       .cast<RankedTensorType>();
    auto retEncoding =
        retType.getEncoding().cast<triton_rocm::gpu_rocm::BlockedEncodingAttr>();
    auto lhsType = adaptor.getLhs().getType().cast<RankedTensorType>();
    auto rhsType = adaptor.getRhs().getType().cast<RankedTensorType>();
    auto lhsTotalElemsPerThread = triton_rocm::gpu_rocm::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton_rocm::gpu_rocm::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton_rocm::gpu_rocm::getTotalElemsPerThread(retType);
    auto retShape = retType.getShape();
    auto retOrder = retEncoding.getOrder();
    auto retSizePerThread = retEncoding.getSizePerThread();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    // Get new retSizePerThread if ret elems per thread is not enough.
    // We have to round it up to the next power of 2 due to triton's tensor size
    // constraint.
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = retSizePerThread.vec();
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton_rocm::gpu_rocm::BlockedEncodingAttr newRetEncoding =
        triton_rocm::gpu_rocm::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
    auto newRetType = RankedTensorType::get(retShape, retType.getElementType(),
                                            newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton_rocm::CatOp>(
                      op, newRetType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<triton_rocm::TransOp> {

  using OpConversionPattern<triton_rocm::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcType = src.getType().cast<RankedTensorType>();
    Attribute srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    if (!srcEncoding.isa<triton_rocm::gpu_rocm::SharedEncodingAttr>()) {
      // TODO: end-to-end correctness is broken if
      // the input is blocked and the output is shared
      // with different order. Maybe a backend issue in BlockedToShared?
      SmallVector<unsigned> order = {1, 0};
      if (auto srcBlockedEncoding =
              srcEncoding.dyn_cast<triton_rocm::gpu_rocm::BlockedEncodingAttr>())
        llvm::copy(srcBlockedEncoding.getOrder(), order.begin());
      // TODO(Qingyi): need to check whether the CTALayout of srcEncoding should
      // be used here. For tests where numCTAs = 1, this is not a problem since
      // all CTALayouts are the same.
      auto CTALayout = triton_rocm::gpu_rocm::getCTALayout(srcEncoding);
      srcEncoding = triton_rocm::gpu_rocm::SharedEncodingAttr::get(getContext(), 1, 1, 1,
                                                         order, CTALayout);
      srcType = RankedTensorType::get(srcType.getShape(),
                                      srcType.getElementType(), srcEncoding);
      src = rewriter.create<triton_rocm::gpu_rocm::ConvertLayoutOp>(src.getLoc(), srcType,
                                                          src);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton_rocm::TransOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton_rocm::BroadcastOp> {
  using OpConversionPattern<triton_rocm::BroadcastOp>::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getSrc().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    auto opType = op.getType().cast<RankedTensorType>();
    Type retType = RankedTensorType::get(opType.getShape(),
                                         opType.getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton_rocm::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton_rocm::ReduceOp> {
  using OpConversionPattern<triton_rocm::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = rewriter.create<triton_rocm::ReduceOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton_rocm::ScanOp> {
  using OpConversionPattern<triton_rocm::ScanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan = rewriter.create<triton_rocm::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton_rocm::FuncOp> {
public:
  using OpConversionPattern<triton_rocm::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<triton_rocm::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton_rocm::CallOp> {
public:
  using OpConversionPattern<triton_rocm::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton_rocm::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton_rocm::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(TritonGPUROCMTypeConverter &typeConverter,
                            RewritePatternSet &patterns, unsigned numCTAs) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      GenericOpPattern<triton_rocm::AdvanceOp>,
      GenericOpPattern<triton_rocm::MakeTensorPtrOp>,
      GenericOpPattern<triton_rocm::ViewOp>, GenericOpPattern<triton_rocm::BitcastOp>,
      GenericOpPattern<triton_rocm::FpToFpOp>, GenericOpPattern<triton_rocm::IntToPtrOp>,
      GenericOpPattern<triton_rocm::PtrToIntOp>, GenericOpPattern<triton_rocm::SplatOp>,
      TritonBroadcastPattern, GenericOpPattern<triton_rocm::AddPtrOp>,
      TritonCatPattern, GenericOpPattern<triton_rocm::ElementwiseInlineAsmOp>,
      TritonReducePattern, GenericOpPattern<triton_rocm::ReduceReturnOp>,
      TritonScanPattern, GenericOpPattern<triton_rocm::ScanReturnOp>,
      GenericOpPattern<triton_rocm::MakeRangeOp>, TritonExpandDimsPattern,
      TritonTransPattern, TritonDotPattern, GenericOpPattern<triton_rocm::LoadOp>,
      GenericOpPattern<triton_rocm::StoreOp>,
      GenericOpPattern<triton_rocm::ExternElementwiseOp>,
      GenericOpPattern<triton_rocm::PrintOp>, GenericOpPattern<triton_rocm::AssertOp>,
      GenericOpPattern<triton_rocm::AtomicCASOp>,
      GenericOpPattern<triton_rocm::AtomicRMWOp>, GenericOpPattern<ReturnOp>,
      GenericOpPattern<triton_rocm::CallOp>, TritonFuncOpPattern>(typeConverter,
                                                             context);
}

//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern<scf::ConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonGPUROCMTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonGPUROCMTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}
//

class ConvertTritonToTritonGPUROCM
    : public ConvertTritonToTritonGPUROCMBase<ConvertTritonToTritonGPUROCM> {
public:
  ConvertTritonToTritonGPUROCM() = default;
  // constructor with some parameters set explicitly.
  ConvertTritonToTritonGPUROCM(int numWarps, int threadsPerWarp, int numCTAs,
                           int computeCapability) {
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUROCMTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs);
    TritonGPUROCMConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    auto inti = llvm::APSInt(32, false);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);

    mod->setAttr(
        AttrNumWarpsName,
        IntegerAttr::get(i32_ty, llvm::APInt(32, numWarps.getValue())));
    mod->setAttr(
        AttrNumThreadsPerWarp,
        IntegerAttr::get(i32_ty, llvm::APInt(32, threadsPerWarp.getValue())));

    mod->setAttr(AttrNumCTAsName,
                 IntegerAttr::get(i32_ty, llvm::APInt(32, numCTAs.getValue())));

    mod->setAttr(AttrComputeCapabilityName,
                 IntegerAttr::get(
                     i32_ty, llvm::APInt(32, computeCapability.getValue())));

    // update layouts
    //  broadcast src => multicast, dst => broadcasted
    // if (failed(target.refineLayouts(mod, numWarps)))
    //   return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton_rocm::createConvertTritonToTritonGPUROCMPass(int numWarps,
                                                 int threadsPerWarp,
                                                 int numCTAs,
                                                 int computeCapability) {
  return std::make_unique<::ConvertTritonToTritonGPUROCM>(
      numWarps, threadsPerWarp, numCTAs, computeCapability);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton_rocm::createConvertTritonToTritonGPUROCMPass() {
  return std::make_unique<::ConvertTritonToTritonGPUROCM>();
}
