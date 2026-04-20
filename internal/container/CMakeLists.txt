file (GLOB SOURCES "./*.cpp")
file (GLOB PERFORMANCE_SOURCES "./performance/*.cpp")

compile_all("true" "container" "Internal/container" "${SOURCES}")
compile_all("true" "container_perf" "Internal/container/performance" "${PERFORMANCE_SOURCES}")

set(SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/sso_vector_performance_orchestrate.cmake")
set(SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH
    "internal/container/container_perf_sso_vector_performance${CMAKE_EXECUTABLE_SUFFIX}")
set(SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION 3)

add_custom_target(container_perf_sso_vector_summary_debug
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DWANTED_CONFIG=Debug
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION}
            -P ${SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT}
    USES_TERMINAL
    VERBATIM)

add_custom_target(container_perf_sso_vector_summary_release
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DWANTED_CONFIG=Release
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION}
            -P ${SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT}
    USES_TERMINAL
    VERBATIM)

add_custom_target(container_perf_sso_vector_compare_builds
    COMMAND ${CMAKE_COMMAND}
            -DACTION=compare-builds
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION}
            -DCOMPARE_BINARY=$<TARGET_FILE:container_perf_sso_vector_performance_compare>
            -P ${SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT}
    DEPENDS container_perf_sso_vector_performance
            container_perf_sso_vector_performance_compare
    USES_TERMINAL
    VERBATIM)
