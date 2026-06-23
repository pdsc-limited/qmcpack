# Deterministic test for QMC optimizer

if(MIXED_PRECISION)
  message(VERBOSE "Skipping optimizer deterministic He tests in mixed precision (QMC_MIXED_PRECISION=1)")
else()
  simple_run_and_check(
    deterministic-He_ae-opt
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt.xml
    1
    1
    check_opt.py
    -s
    1
    -p
    det_He_opt
    -r
    reference.det_He_opt.s001.opt.xml)

  simple_run_and_check(
    deterministic-He_ae-opt
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_mwalkers.xml
    1
    16
    check_opt.py
    -s
    1
    -p
    det_He_opt_mwalkers
    -r
    reference.det_He_opt_mwalkers_1-16.s001.opt.xml)

  simple_run_and_check(
    deterministic-He_ae-opt
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_mwalkers.xml
    2
    3
    check_opt.py
    -s
    1
    -p
    det_He_opt_mwalkers
    -r
    reference.det_He_opt_mwalkers_2-3.s001.opt.xml)

  list(APPEND He_VMC_SCALARS "totenergy" "-3.25291030 0.000001")

  list(APPEND He_VMC_MWALKERS_1-16_SCALAR "totenergy" "-2.84983901 0.000001")

  list(APPEND He_VMC_MWALKERS_2-3_SCALAR "totenergy" "-2.81743526 0.000001")

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt
    det_He_opt.xml
    1
    1
    TRUE
    2
    He_VMC_SCALARS # VMC
  )

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_mwalkers
    det_He_opt_mwalkers.xml
    1
    16
    TRUE
    2
    He_VMC_MWALKERS_1-16_SCALAR # VMC
  )

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_mwalkers
    det_He_opt_mwalkers.xml
    2
    3
    TRUE
    2
    He_VMC_MWALKERS_2-3_SCALAR # VMC
  )

  # Test batched version of optimizer

  list(APPEND He_batch_VMC_SCALARS "totenergy" "-3.25291030 0.000001")

  list(APPEND He_batch_VMC_MWALKERS_1-16_SCALARS "totenergy" "-2.82592127 0.000001")

  list(APPEND He_batch_VMC_MWALKERS_2-3_SCALARS "totenergy" "-2.92537421 0.000001")

  simple_run_and_check(
    deterministic-He_ae-opt-batch
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch.xml
    1
    1
    check_opt.py
    -s
    1
    -p
    det_He_opt_batch
    -r
    reference.det_He_opt.s001.opt.xml)

  simple_run_and_check(
    deterministic-He_ae-opt-batch-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch_mwalkers.xml
    1
    16
    check_opt.py
    -s
    1
    -p
    det_He_opt_batch_mwalkers
    -r
    reference.det_He_opt_batch_mwalkers_1-16.s001.opt.xml)

  simple_run_and_check(
    deterministic-He_ae-opt-batch-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch_mwalkers.xml
    2
    3
    check_opt.py
    -s
    1
    -p
    det_He_opt_batch_mwalkers
    -r
    reference.det_He_opt_batch_mwalkers_2-3.s001.opt.xml)

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc-batch
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch
    det_He_opt_batch.xml
    1
    1
    TRUE
    2
    He_batch_VMC_SCALARS # VMC
  )

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc-batch-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch_mwalkers
    det_He_opt_batch_mwalkers.xml
    1
    16
    TRUE
    2
    He_batch_VMC_MWALKERS_1-16_SCALARS # VMC
  )

  qmc_run_and_check(
    deterministic-He_ae-opt_vmc-batch-mwalkers
    "${qmcpack_SOURCE_DIR}/tests/molecules/He_ae"
    det_He_opt_batch_mwalkers
    det_He_opt_batch_mwalkers.xml
    2
    3
    TRUE
    2
    He_batch_VMC_MWALKERS_2-3_SCALARS # VMC
  )

endif()

if(ENABLE_DEEPQMC_INFERENCE AND DEEPQMC_RESOLVED_HE_CHECKPOINT AND DEEPQMC_PYTHON_SITE_PACKAGES)
  set(DEEPQMC_HE_VMC_BATCH_TEST deterministic-He_ae-deepqmc_vmc_batch)
  set(DEEPQMC_HE_VMC_BATCH_WORKDIR "${CMAKE_CURRENT_BINARY_DIR}/${DEEPQMC_HE_VMC_BATCH_TEST}-r1-t1")
  file(MAKE_DIRECTORY "${DEEPQMC_HE_VMC_BATCH_WORKDIR}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/deepqmc_he_vmc_batch.xml.in"
    "${DEEPQMC_HE_VMC_BATCH_WORKDIR}/deepqmc_he_vmc_batch.xml"
    @ONLY)

  if(TARGET deepqmc_he_checkpoint)
    add_dependencies(qmcpack deepqmc_he_checkpoint)
  endif()

  run_qmc_app_no_copy(
    ${DEEPQMC_HE_VMC_BATCH_TEST}-r1-t1
    "${DEEPQMC_HE_VMC_BATCH_WORKDIR}"
    1
    1
    DEEPQMC_HE_VMC_BATCH_TEST_ADDED
    DEEPQMC_HE_VMC_BATCH_TEST_LABELS
    deepqmc_he_vmc_batch.xml)

  if(DEEPQMC_HE_VMC_BATCH_TEST_ADDED)
    set(DEEPQMC_HE_VMC_BATCH_PYTHONPATH "${DEEPQMC_PYTHON_SITE_PACKAGES}")
    if(DEEPQMC_PYTHONPATH)
      set(DEEPQMC_HE_VMC_BATCH_PYTHONPATH "${DEEPQMC_PYTHONPATH}")
    endif()
    set_property(
      TEST ${DEEPQMC_HE_VMC_BATCH_TEST}-r1-t1
      APPEND
      PROPERTY ENVIRONMENT "JAX_PLATFORMS=cpu" "DEEPQMC_HE_CHECKPOINT=${DEEPQMC_RESOLVED_HE_CHECKPOINT}"
               "DEEPQMC_PYTHON_SITE_PACKAGES=${DEEPQMC_PYTHON_SITE_PACKAGES}"
               "PYTHONPATH=${DEEPQMC_HE_VMC_BATCH_PYTHONPATH}")
  endif()
elseif(ENABLE_DEEPQMC_INFERENCE)
  message(STATUS "DeepQMC He VMC batch system test will skip. It requires a resolved He checkpoint and DEEPQMC_PYTHON_SITE_PACKAGES.")
endif()
