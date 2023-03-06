configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCe_DIR}/lit.cfg.py
)

set(TRITON_TEST_DEPENDS
  triton-opt
  FileCheck
)

add_lit_testsuite(check-rocMLIR-lit-tests "Running the dot-rocMLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRITON_TEST_DEPENDS}
)

set_target_properties(check-rocMLIR-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(DOT-ROCMLIR-LIT-TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_TEST_DEPENDS})
