include_guard(GLOBAL)

include(CMakeParseArguments)

function(universal_add_cross_build_benchmark_aggregation)
  set(options)
  set(oneValueArgs
    FAMILY_LABEL
    SUMMARY_DEBUG_TARGET
    SUMMARY_RELEASE_TARGET
    COMPARE_BUILDS_TARGET
    BENCHMARK_TARGET
    COMPARE_TARGET
    BENCHMARK_RELATIVE_PATH
    SUMMARY_SCHEMA_VERSION
    REPORT_OUTPUT_PATH
    TARGET_FOLDER)
  cmake_parse_arguments(UCBBA "${options}" "${oneValueArgs}" "" ${ARGN})

  foreach(required_arg
      FAMILY_LABEL
      SUMMARY_DEBUG_TARGET
      SUMMARY_RELEASE_TARGET
      COMPARE_BUILDS_TARGET
      BENCHMARK_TARGET
      COMPARE_TARGET
      BENCHMARK_RELATIVE_PATH
      SUMMARY_SCHEMA_VERSION
      REPORT_OUTPUT_PATH)
    if(NOT UCBBA_${required_arg})
      message(FATAL_ERROR
        "universal_add_cross_build_benchmark_aggregation missing required argument: ${required_arg}")
    endif()
  endforeach()

  set(_orchestrate_script
    "${PROJECT_SOURCE_DIR}/tools/cmake/CrossBuildBenchmarkAggregationOrchestrate.cmake")

  add_custom_target(${UCBBA_SUMMARY_DEBUG_TARGET}
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DBENCHMARK_FAMILY_LABEL=${UCBBA_FAMILY_LABEL}
            -DWANTED_CONFIG=Debug
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${UCBBA_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${UCBBA_SUMMARY_SCHEMA_VERSION}
            -P ${_orchestrate_script}
    DEPENDS ${UCBBA_BENCHMARK_TARGET}
    USES_TERMINAL
    VERBATIM)

  add_custom_target(${UCBBA_SUMMARY_RELEASE_TARGET}
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DBENCHMARK_FAMILY_LABEL=${UCBBA_FAMILY_LABEL}
            -DWANTED_CONFIG=Release
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${UCBBA_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${UCBBA_SUMMARY_SCHEMA_VERSION}
            -P ${_orchestrate_script}
    DEPENDS ${UCBBA_BENCHMARK_TARGET}
    USES_TERMINAL
    VERBATIM)

  add_custom_target(${UCBBA_COMPARE_BUILDS_TARGET}
    COMMAND ${CMAKE_COMMAND}
            -DACTION=compare-builds
            -DBENCHMARK_FAMILY_LABEL=${UCBBA_FAMILY_LABEL}
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${UCBBA_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${UCBBA_SUMMARY_SCHEMA_VERSION}
            -DCOMPARE_BINARY=$<TARGET_FILE:${UCBBA_COMPARE_TARGET}>
            -DREPORT_OUTPUT_PATH=${UCBBA_REPORT_OUTPUT_PATH}
            -P ${_orchestrate_script}
    DEPENDS ${UCBBA_BENCHMARK_TARGET}
            ${UCBBA_COMPARE_TARGET}
    USES_TERMINAL
    VERBATIM)

  if(UCBBA_TARGET_FOLDER)
    set_target_properties(
      ${UCBBA_SUMMARY_DEBUG_TARGET}
      ${UCBBA_SUMMARY_RELEASE_TARGET}
      ${UCBBA_COMPARE_BUILDS_TARGET}
      PROPERTIES FOLDER "${UCBBA_TARGET_FOLDER}")
  endif()
endfunction()
