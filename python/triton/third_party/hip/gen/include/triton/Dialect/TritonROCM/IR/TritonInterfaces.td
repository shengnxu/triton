#ifndef TRITONROCM_INTERFACES
#define TRITONROCM_INTERFACES

include "mlir/IR/OpBase.td"

def TensorSizeTraitROCM : NativeOpTrait<"TensorSizeTraitROCM">;
def SameOperandsEncodingROCM : NativeOpTrait<"SameOperandsEncodingROCM">;
def SameOperandsAndResultEncodingROCM : NativeOpTrait<"SameOperandsAndResultEncodingROCM">;
def SameLoadStoreOperandsShapeROCM : NativeOpTrait<"SameLoadStoreOperandsShapeROCM">;
def SameLoadStoreOperandsAndResultShapeROCM : NativeOpTrait<"SameLoadStoreOperandsAndResultShapeROCM">;
def SameLoadStoreOperandsEncodingROCM : NativeOpTrait<"SameLoadStoreOperandsEncodingROCM">;
def SameLoadStoreOperandsAndResultEncodingROCM : NativeOpTrait<"SameLoadStoreOperandsAndResultEncodingROCM">;

#endif // TRITON_INTERFACES
