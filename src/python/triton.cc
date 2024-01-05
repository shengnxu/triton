#include <pybind11/pybind11.h>
#include "mlir/Pass/PassManager.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUToLLVM/Passes.h"

namespace py = pybind11;

      
void init_tritonamd(py::module &_m) {
  py::module m = _m.def_submodule("triton_amd");

  py::class_<mlir::PassManager>(m, "pass_manager", py::module_local())
    .def(py::init<mlir::MLIRContext *>())
    .def("run",
           [](mlir::PassManager &self, mlir::ModuleOp &mod) {
             // TODO: maybe dump module to file and print error for better
             // diagnostics
             if (mlir::failed(self.run(mod.getOperation())))
               throw std::runtime_error("PassManager::run failed");
           })

    .def("add_triton_gpu_to_llvm",
          [](mlir::PassManager &self) {
            self.addPass(mlir::triton::createConvertTritonAMDGPUToLLVMPass());
          })
    ;

}
