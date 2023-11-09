#ifndef TRITONROCM_TYPES
#define TRITONROCM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonROCM/IR/TritonDialect.td"

//
// Types
//
class TritonROCMTypeDef<string name, string _mnemonic>
    : TypeDef<TritonROCM_Dialect, name> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Floating-point Type
def TTROCM_Float : AnyTypeOf<[F8E4M3FNUZ, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def TTROCM_FloatTensor : TensorOf<[TTROCM_Float]>;
def TTROCM_FloatLike : AnyTypeOf<[TTROCM_Float, TTROCM_FloatTensor]>;

// Boolean Type
// TT_Bool -> I1
def TTROCM_BoolTensor : TensorOf<[I1]>;
def TTROCM_BoolLike : AnyTypeOf<[I1, TTROCM_BoolTensor]>;

// Integer Type
def TTROCM_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def TTROCM_IntTensor : TensorOf<[TTROCM_Int]>;
def TTROCM_IntLike : AnyTypeOf<[TTROCM_Int, TTROCM_IntTensor]>;

// I32 Type
// TT_I32 -> I32
// TT_I32Tensor -> I32Tensor
def TTROCM_I32Like : AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// TT_I64 -> I64
// TT_I64Tensor -> I64Tensor
def TTROCM_I64Like : AnyTypeOf<[I64, I64Tensor]>;

// Pointer Type in TableGen
class TTROCM_PtrOf<list<Type> pointeeTypes> :
    DialectType<TritonROCM_Dialect,
                And<[CPred<"$_self.isa<::mlir::triton_rocm::PointerType>()">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }($_self.cast<::mlir::triton_rocm::PointerType>().getPointeeType())">]>,
                "ptr", "::mlir::triton_rocm::PointerType">;

// Pointer Type in C++ (corresponding to `TT_PtrOf`)
def TTROCM_PtrType : TritonROCMTypeDef<"Pointer", "ptr"> {
    let summary = "Pointer type (`::mlir::triton_rocm::PointerType`) in Triton IR type system";

    let description = [{
        Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

// Scalar Pointer Type: `ptr<>`
def TTROCM_Ptr : TTROCM_PtrOf<[AnyType]>;

// Tensor of Pointer Type: `tensor<ptr<>>`
def TTROCM_PtrTensor : TensorOf<[TTROCM_Ptr]>;

// Tensor of Pointer Type or Pointer type: `tensor<ptr<>>` or `ptr<>`
def TTROCM_PtrLike : AnyTypeOf<[TTROCM_Ptr, TTROCM_PtrTensor]>;

// Tensor Type
def TTROCM_FpIntTensor : AnyTypeOf<[TTROCM_FloatTensor, TTROCM_IntTensor]>;
def TTROCM_Tensor : AnyTypeOf<[TTROCM_FpIntTensor, TTROCM_PtrTensor]>;

// Pointer Type to Tensor Type: `ptr<tensor<>>`
def TTROCM_TensorPtr : TTROCM_PtrOf<[TTROCM_Tensor]>;

// Any Type in Triton IR
def TTROCM_Type : AnyTypeOf<[TTROCM_FloatLike, TTROCM_IntLike, TTROCM_PtrLike, TTROCM_TensorPtr]>;

#endif
