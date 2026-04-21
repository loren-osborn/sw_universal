file (GLOB API_SRC         "api/*.cpp")
file (GLOB LOGIC_SRC       "logic/*.cpp")
file (GLOB CONVERSION_SRC  "conversion/*.cpp")
file (GLOB ARITHMETIC_SRC  "arithmetic/*.cpp")
file (GLOB MATH_SRC        "./math/*.cpp")
file (GLOB PERFORMANCE_SRC "./performance/*.cpp")
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

set(EINTEGER_PERF_ORCHESTRATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/einteger_performance_orchestrate.cmake")
set(EINTEGER_PERF_BENCHMARK_RELATIVE_PATH
    "elastic/einteger/eint_performance${CMAKE_EXECUTABLE_SUFFIX}")
set(EINTEGER_PERF_SUMMARY_SCHEMA_VERSION 1)

add_custom_target(eint_performance_summary_debug
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DWANTED_CONFIG=Debug
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${EINTEGER_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${EINTEGER_PERF_SUMMARY_SCHEMA_VERSION}
            -P ${EINTEGER_PERF_ORCHESTRATE_SCRIPT}
    USES_TERMINAL
    VERBATIM)

add_custom_target(eint_performance_summary_release
    COMMAND ${CMAKE_COMMAND}
            -DACTION=refresh-config
            -DWANTED_CONFIG=Release
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${EINTEGER_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${EINTEGER_PERF_SUMMARY_SCHEMA_VERSION}
            -P ${EINTEGER_PERF_ORCHESTRATE_SCRIPT}
    USES_TERMINAL
    VERBATIM)

add_custom_target(eint_performance_compare_builds
    COMMAND ${CMAKE_COMMAND}
            -DACTION=compare-builds
            -DCURRENT_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DBENCHMARK_RELATIVE_PATH=${EINTEGER_PERF_BENCHMARK_RELATIVE_PATH}
            -DSUMMARY_SCHEMA_VERSION=${EINTEGER_PERF_SUMMARY_SCHEMA_VERSION}
            -DCOMPARE_BINARY=$<TARGET_FILE:eint_performance_compare_tool>
            -P ${EINTEGER_PERF_ORCHESTRATE_SCRIPT}
    DEPENDS eint_performance
            eint_performance_compare_tool
    USES_TERMINAL
    VERBATIM)
