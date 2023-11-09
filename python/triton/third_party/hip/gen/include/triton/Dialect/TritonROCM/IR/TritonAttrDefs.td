#ifndef TRITONROCM_ATTR_DEFS
#define TRITONROCM_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

// Attributes for LoadOp and StoreOp
def TTROCM_CacheModifierAttr : I32EnumAttr<
    "CacheModifier", "",
    [
        I32EnumAttrCase<"NONE", 1, "none">,
        I32EnumAttrCase<"CA", 2, "ca">,
        I32EnumAttrCase<"CG", 3, "cg">,
        I32EnumAttrCase<"WB", 4, "wb">,
        I32EnumAttrCase<"CS", 5, "cs">,
        I32EnumAttrCase<"WT", 6, "wt">,
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

def TTROCM_MemSemanticAttr : I32EnumAttr<
    "MemSemantic", "",
    [
      I32EnumAttrCase<"RELAXED", 1, "relaxed">,
      I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
      I32EnumAttrCase<"RELEASE", 3, "release">,
      I32EnumAttrCase<"ACQUIRE_RELEASE", 4, "acq_rel">,
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

def TTROCM_EvictionPolicyAttr : I32EnumAttr<
    "EvictionPolicy", "",
    [
        I32EnumAttrCase<"NORMAL", 1, "evict_normal">,
        I32EnumAttrCase<"EVICT_FIRST", 2, "evict_first">,
        I32EnumAttrCase<"EVICT_LAST", 3, "evict_last">
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

def TTROCM_PaddingOptionAttr : I32EnumAttr<
    "PaddingOption", "",
    [
        I32EnumAttrCase<"PAD_ZERO", 1, "zero">,
        // We can not set the string value to "NAN" because it is a keyword in C++
        I32EnumAttrCase<"PAD_NAN", 2, "nan">
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

// atomic
def TTROCM_AtomicRMWAttr : I32EnumAttr<
    "RMWOp", "",
    [
        I32EnumAttrCase<"AND", 1, "and">,
        I32EnumAttrCase<"OR", 2, "or">,
        I32EnumAttrCase<"XOR", 3, "xor">,
        I32EnumAttrCase<"ADD", 4, "add">,
        I32EnumAttrCase<"FADD", 5, "fadd">,
        I32EnumAttrCase<"MAX", 6, "max">,
        I32EnumAttrCase<"MIN", 7, "min">,
        I32EnumAttrCase<"UMAX", 8, "umax">,
        I32EnumAttrCase<"UMIN", 9, "umin">,
        I32EnumAttrCase<"XCHG", 10, "exch">
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

def TTROCM_MemSyncScopeAttr : I32EnumAttr<
    "MemSyncScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"CTA", 2, "cta">,
      I32EnumAttrCase<"SYSTEM", 3, "sys">,
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}

// Program ID dimensions.
def TTROCM_ProgramDim : I32EnumAttr<
    "ProgramIDDim", "",
    [
        I32EnumAttrCase<"X", 0, "x">,
        I32EnumAttrCase<"Y", 1, "y">,
        I32EnumAttrCase<"Z", 2, "z">,
    ]> {
    let cppNamespace = "::mlir::triton_rocm";
}


#endif
