load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

cc_library(
    name = "TritonAMDGPUToLLVM",
    srcs = glob([
        "third_party/amd/lib/TritonAMDGPUToLLVM/*.cpp",
        "third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM/*.cpp",
        "third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/*.cpp",
    ]),
    hdrs = glob([
        "third_party/amd/include/TritonAMDGPUToLLVM/*.h",
        "third_party/amd/include/*.h",
        "third_party/amd/lib/*.h",
        "third_party/amd/lib/TritonAMDGPUToLLVM/*.h",
    ]),
    copts = _no_unused_variable,
    includes = ["include",
		"third_party/amd/include"],
    deps = [
        ":TritonAnalysis",
        ":TritonDialects",
        ":TritonGPUToLLVM",
        ":triton_conversion_amdgpu_to_llvm_passes_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "triton_conversion_amdgpu_to_llvm_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonAMDGPUToLLVM",
            ],
            "third_party/amd/include/TritonAMDGPUToLLVM/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/amd/include/TritonAMDGPUToLLVM/Passes.td",
    deps = ["td_files"],
)
