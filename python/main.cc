#include <pybind11/pybind11.h>

void init_tritonamd(pybind11::module &m);

PYBIND11_MODULE(libtriton_amd, m) {
  m.doc() = "Triton AMD backend API";
  init_tritonamd(m);
}
