file (GLOB SOURCES "./*.cpp")
file (GLOB PERFORMANCE_SOURCES "./performance/*.cpp")

compile_all("true" "container" "Internal/container" "${SOURCES}")
compile_all("true" "container_perf" "Internal/container/performance" "${PERFORMANCE_SOURCES}")
