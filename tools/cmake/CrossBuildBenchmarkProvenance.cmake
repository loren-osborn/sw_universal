include_guard(GLOBAL)

# Cross-build benchmark aggregation provenance is opt-in. Normal benchmark
# builds do not need this generated header or its metadata.
function(universal_prepare_cross_build_benchmark_provenance)
  if(TARGET universal_cross_build_benchmark_provenance)
    return()
  endif()

  set(BENCH_BUILD_CONFIG "${CMAKE_BUILD_TYPE}")
  if("${BENCH_BUILD_CONFIG}" STREQUAL "")
    set(BENCH_BUILD_CONFIG "Unknown")
  endif()

  set(BENCH_PROVENANCE_STATUS "unknown")
  set(BENCH_PROVENANCE_REASON "git state unavailable")
  set(BENCH_PROVENANCE_BASE_COMMIT_HASH "")
  set(BENCH_PROVENANCE_DIRTY_FINGERPRINT "")
  set(BENCH_PROVENANCE_PUBLISHABLE "0")

  set(BENCH_GIT_HEAD_RESULT 1)
  set(BENCH_GIT_STATUS_RESULT 1)
  if(GIT_EXECUTABLE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE BENCH_GIT_HEAD_RESULT
      OUTPUT_VARIABLE BENCH_GIT_HEAD_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND ${GIT_EXECUTABLE} status --porcelain=v1 --untracked-files=all
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE BENCH_GIT_STATUS_RESULT
      OUTPUT_VARIABLE BENCH_GIT_STATUS_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  else()
    set(BENCH_PROVENANCE_REASON "git executable unavailable")
  endif()

  if(GIT_EXECUTABLE AND BENCH_GIT_HEAD_RESULT EQUAL 0 AND BENCH_GIT_STATUS_RESULT EQUAL 0)
    set(BENCH_PROVENANCE_BASE_COMMIT_HASH "${BENCH_GIT_HEAD_HASH}")
    set(BENCH_PROVENANCE_STATUS "clean")
    set(BENCH_PROVENANCE_REASON "clean HEAD")
    set(BENCH_PROVENANCE_PUBLISHABLE "1")
    set(BENCH_RELEVANT_UNTRACKED_FILES "")
    string(REPLACE "\n" ";" BENCH_GIT_STATUS_LINES "${BENCH_GIT_STATUS_OUTPUT}")
    foreach(BENCH_GIT_STATUS_LINE IN LISTS BENCH_GIT_STATUS_LINES)
      if("${BENCH_GIT_STATUS_LINE}" STREQUAL "")
        continue()
      endif()

      if(BENCH_GIT_STATUS_LINE MATCHES "^\\?\\? ")
        string(SUBSTRING "${BENCH_GIT_STATUS_LINE}" 3 -1 BENCH_UNTRACKED_PATH)
        if(BENCH_UNTRACKED_PATH MATCHES "(^|/)CMakeLists\\.txt$" OR
           BENCH_UNTRACKED_PATH MATCHES "\\.(cpp|cc|c|hpp|hh|h|cmake)$")
          list(APPEND BENCH_RELEVANT_UNTRACKED_FILES "${BENCH_UNTRACKED_PATH}")
        endif()
      elseif(NOT BENCH_GIT_STATUS_LINE MATCHES "^!! ")
        set(BENCH_PROVENANCE_STATUS "dirty_matchable")
        set(BENCH_PROVENANCE_REASON "tracked changes present")
        set(BENCH_PROVENANCE_PUBLISHABLE "0")
      endif()
    endforeach()

    if(BENCH_RELEVANT_UNTRACKED_FILES)
      list(SORT BENCH_RELEVANT_UNTRACKED_FILES)
      set(BENCH_PROVENANCE_STATUS "dirty_matchable")
      set(BENCH_PROVENANCE_REASON "tracked and/or relevant untracked changes present")
      set(BENCH_PROVENANCE_PUBLISHABLE "0")
    endif()

    if(BENCH_PROVENANCE_STATUS STREQUAL "dirty_matchable")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --no-ext-diff --binary --cached HEAD --
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE BENCH_GIT_STAGED_DIFF_RESULT
        OUTPUT_VARIABLE BENCH_GIT_STAGED_DIFF_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --no-ext-diff --binary --
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE BENCH_GIT_UNSTAGED_DIFF_RESULT
        OUTPUT_VARIABLE BENCH_GIT_UNSTAGED_DIFF_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(BENCH_GIT_STAGED_DIFF_RESULT EQUAL 0 AND BENCH_GIT_UNSTAGED_DIFF_RESULT EQUAL 0)
        set(BENCH_DIRTY_CANONICAL_TEXT "base_commit=${BENCH_PROVENANCE_BASE_COMMIT_HASH}\n")
        string(APPEND BENCH_DIRTY_CANONICAL_TEXT "staged_diff_begin\n${BENCH_GIT_STAGED_DIFF_OUTPUT}\nstaged_diff_end\n")
        string(APPEND BENCH_DIRTY_CANONICAL_TEXT "unstaged_diff_begin\n${BENCH_GIT_UNSTAGED_DIFF_OUTPUT}\nunstaged_diff_end\n")
        foreach(BENCH_UNTRACKED_PATH IN LISTS BENCH_RELEVANT_UNTRACKED_FILES)
          set(BENCH_UNTRACKED_ABS_PATH "${CMAKE_SOURCE_DIR}/${BENCH_UNTRACKED_PATH}")
          if(EXISTS "${BENCH_UNTRACKED_ABS_PATH}")
            file(SHA256 "${BENCH_UNTRACKED_ABS_PATH}" BENCH_UNTRACKED_SHA256)
            string(APPEND BENCH_DIRTY_CANONICAL_TEXT
                   "untracked_file=${BENCH_UNTRACKED_PATH}|sha256=${BENCH_UNTRACKED_SHA256}\n")
          else()
            string(APPEND BENCH_DIRTY_CANONICAL_TEXT
                   "untracked_file=${BENCH_UNTRACKED_PATH}|missing\n")
          endif()
        endforeach()
        string(SHA256 BENCH_PROVENANCE_DIRTY_FINGERPRINT "${BENCH_DIRTY_CANONICAL_TEXT}")
      else()
        set(BENCH_PROVENANCE_STATUS "unknown")
        set(BENCH_PROVENANCE_REASON "dirty tree fingerprint unavailable")
        set(BENCH_PROVENANCE_BASE_COMMIT_HASH "")
        set(BENCH_PROVENANCE_DIRTY_FINGERPRINT "")
        set(BENCH_PROVENANCE_PUBLISHABLE "0")
      endif()
    endif()
  endif()

  configure_file(
    "${PROJECT_SOURCE_DIR}/tools/cmake/Templates/BenchmarkProvenance.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/BenchmarkProvenance.hpp"
    @ONLY
  )

  add_library(universal_cross_build_benchmark_provenance INTERFACE)
  target_include_directories(universal_cross_build_benchmark_provenance
    INTERFACE
      "${CMAKE_BINARY_DIR}/generated")
  target_compile_definitions(universal_cross_build_benchmark_provenance
    INTERFACE
      UNIVERSAL_HAS_CROSS_BUILD_BENCHMARK_PROVENANCE_HEADER=1)
endfunction()

function(universal_enable_cross_build_benchmark_provenance target_name)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR
      "Cross-build benchmark aggregation provenance expected an existing target: ${target_name}")
  endif()

  universal_prepare_cross_build_benchmark_provenance()
  target_link_libraries("${target_name}" PRIVATE universal_cross_build_benchmark_provenance)
endfunction()
