#ifndef TRITONROCM_IR_TRAITS_H_
#define TRITONROCM_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace impl {
// The rationale for this trait is to prevent users from creating programs
// that would have catastrophic register pressure and cause the compiler to
// hang.
// Since H100 has 256KB registers, we should allow users to create tensors
// of size up to 256K elements. It will spill for datatypes wider than 1B,
// but we probably should limit number of elements (rather than bytes) to
// keep specs simple
int constexpr maxTensorNumElementsROCM = 1048576;

LogicalResult verifyTensorSize(Operation *op);

LogicalResult verifySameOperandsEncodingROCM(Operation *op,
                                         bool allowTensorPointerType = false);

LogicalResult
verifySameOperandsAndResultEncodingROCM(Operation *op,
                                    bool allowTensorPointerType = false);

LogicalResult verifySameLoadStoreOperandsShapeROCM(Operation *op);

LogicalResult verifySameLoadStoreOperandsAndResultShapeROCM(Operation *op);

bool verifyLoadStorePointerAndValueType(Type valueType, Type ptrType);

} // namespace impl

template <class ConcreteType>
class TensorSizeTraitROCM : public TraitBase<ConcreteType, TensorSizeTraitROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorSize(op);
  }
};

template <typename ConcreteType>
class SameOperandsAndResultEncodingROCM
    : public TraitBase<ConcreteType, SameOperandsAndResultEncodingROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncodingROCM(op);
  }
};

template <typename ConcreteType>
class SameOperandsEncodingROCM
    : public TraitBase<ConcreteType, SameOperandsEncodingROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncodingROCM(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsShapeROCM
    : public TraitBase<ConcreteType, SameLoadStoreOperandsShapeROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsShapeROCM(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultShapeROCM
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultShapeROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsAndResultShapeROCM(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsEncodingROCM
    : public TraitBase<ConcreteType, SameLoadStoreOperandsEncodingROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncodingROCM(op,
                                            /*allowTensorPointerType=*/true);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultEncodingROCM
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultEncodingROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncodingROCM(
        op, /*allowTensorPointerType=*/true);
  }
};

} // namespace OpTrait
} // namespace mlir

#endif
