file (GLOB SOURCES "./*.cpp")
file (GLOB PERFORMANCE_SOURCES "./performance/*.cpp")
list(FILTER PERFORMANCE_SOURCES EXCLUDE REGEX "sso_vector_performance_compare\\.cpp$")

compile_all("true" "container" "Internal/container" "${SOURCES}")
compile_all("true" "container_perf" "Internal/container/performance" "${PERFORMANCE_SOURCES}")

# This helper compares provenance-aware benchmark summaries for cross-build
# aggregation and requires explicit summary-file arguments.
add_executable(container_perf_sso_vector_performance_compare
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/sso_vector_performance_compare.cpp")
set_target_properties(container_perf_sso_vector_performance_compare PROPERTIES
    FOLDER "Internal/container/performance")

if(UNIVERSAL_ENABLE_CROSS_BUILD_BENCHMARK_AGGREGATION)
    include("${PROJECT_SOURCE_DIR}/tools/cmake/CrossBuildBenchmarkProvenance.cmake")
    include("${PROJECT_SOURCE_DIR}/tools/cmake/CrossBuildBenchmarkAggregation.cmake")

    universal_enable_cross_build_benchmark_provenance(container_perf_sso_vector_performance)
    universal_add_cross_build_benchmark_aggregation(
        FAMILY_LABEL "sso_vector"
        SUMMARY_DEBUG_TARGET container_perf_sso_vector_summary_debug
        SUMMARY_RELEASE_TARGET container_perf_sso_vector_summary_release
        COMPARE_BUILDS_TARGET container_perf_sso_vector_compare_builds
        BENCHMARK_TARGET container_perf_sso_vector_performance
        COMPARE_TARGET container_perf_sso_vector_performance_compare
        BENCHMARK_RELATIVE_PATH "internal/container/container_perf_sso_vector_performance${CMAKE_EXECUTABLE_SUFFIX}"
        SUMMARY_SCHEMA_VERSION 3
        REPORT_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/benchmark-results/sso_vector_debug_vs_release_report.txt"
        TARGET_FOLDER "Internal/container/performance")
endif()
