file (GLOB API_SRC         "api/*.cpp")
file (GLOB LOGIC_SRC       "logic/*.cpp")
file (GLOB CONVERSION_SRC  "conversion/*.cpp")
file (GLOB ARITHMETIC_SRC  "arithmetic/*.cpp")
file (GLOB MATH_SRC        "./math/*.cpp")
file (GLOB PERFORMANCE_SRC "./performance/*.cpp")
# This helper compares provenance-aware benchmark summaries for cross-build
# aggregation and requires explicit summary-file arguments.
list(FILTER PERFORMANCE_SRC EXCLUDE REGEX "performance_compare\\.cpp$")

compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/api"         "${API_SRC}")
compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/logic"       "${LOGIC_SRC}")
compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/conversion"  "${CONVERSION_SRC}")
compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/arithmetic"  "${ARITHMETIC_SRC}")
compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/math"        "${MATH_SRC}")
compile_all("true" "eint" "Number Systems/elastic/integer/binary/einteger/performance" "${PERFORMANCE_SRC}")

add_executable(eint_performance_compare_tool "${CMAKE_CURRENT_SOURCE_DIR}/performance/performance_compare.cpp")
set_target_properties(eint_performance_compare_tool PROPERTIES
    FOLDER "Number Systems/elastic/integer/binary/einteger/performance")

if(UNIVERSAL_ENABLE_CROSS_BUILD_BENCHMARK_AGGREGATION)
    include("${PROJECT_SOURCE_DIR}/tools/cmake/CrossBuildBenchmarkProvenance.cmake")
    include("${PROJECT_SOURCE_DIR}/tools/cmake/CrossBuildBenchmarkAggregation.cmake")

    universal_enable_cross_build_benchmark_provenance(eint_performance)
    universal_add_cross_build_benchmark_aggregation(
        FAMILY_LABEL "einteger"
        SUMMARY_DEBUG_TARGET eint_performance_summary_debug
        SUMMARY_RELEASE_TARGET eint_performance_summary_release
        COMPARE_BUILDS_TARGET eint_performance_compare_builds
        BENCHMARK_TARGET eint_performance
        COMPARE_TARGET eint_performance_compare_tool
        BENCHMARK_RELATIVE_PATH "elastic/einteger/eint_performance${CMAKE_EXECUTABLE_SUFFIX}"
        SUMMARY_SCHEMA_VERSION 1
        REPORT_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/benchmark-results/einteger_debug_vs_release_report.txt"
        TARGET_FOLDER "Number Systems/elastic/integer/binary/einteger/performance")
endif()
