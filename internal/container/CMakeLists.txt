file (GLOB SOURCES "./*.cpp")
file (GLOB PERFORMANCE_SOURCES "./performance/*.cpp")
list(FILTER PERFORMANCE_SOURCES EXCLUDE REGEX "sso_vector_performance_compare\\.cpp$")

compile_all("true" "container" "Internal/container" "${SOURCES}")
compile_all("true" "container_perf" "Internal/container/performance" "${PERFORMANCE_SOURCES}")

# This helper compares persisted benchmark summaries and requires explicit
# summary-file arguments, so keep it buildable but out of direct CTest runs.
add_executable(container_perf_sso_vector_performance_compare
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/sso_vector_performance_compare.cpp")
set_target_properties(container_perf_sso_vector_performance_compare PROPERTIES
    FOLDER "Internal/container/performance")

set(SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/sso_vector_performance_orchestrate.cmake")
set(SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH
    "internal/container/container_perf_sso_vector_performance${CMAKE_EXECUTABLE_SUFFIX}")
set(SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION 3)
set(SSO_VECTOR_PERF_REPORT_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/benchmark-results/sso_vector_debug_vs_release_report.txt")

add_custom_target(container_perf_sso_vector_summary_debug
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DWANTED_CONFIG=Debug
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION}
            -P ${SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT}
    DEPENDS container_perf_sso_vector_performance
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
    DEPENDS container_perf_sso_vector_performance
    USES_TERMINAL
    VERBATIM)

add_custom_target(container_perf_sso_vector_compare_builds
    COMMAND ${CMAKE_COMMAND}
            -DACTION=compare-builds
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${SSO_VECTOR_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${SSO_VECTOR_PERF_SUMMARY_SCHEMA_VERSION}
            -DCOMPARE_BINARY=$<TARGET_FILE:container_perf_sso_vector_performance_compare>
            -DREPORT_OUTPUT_PATH=${SSO_VECTOR_PERF_REPORT_PATH}
            -P ${SSO_VECTOR_PERF_ORCHESTRATE_SCRIPT}
    DEPENDS container_perf_sso_vector_performance
            container_perf_sso_vector_performance_compare
    USES_TERMINAL
    VERBATIM)
