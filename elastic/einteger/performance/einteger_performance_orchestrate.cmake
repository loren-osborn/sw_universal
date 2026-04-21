cmake_minimum_required(VERSION 3.22)

# Cross-platform orchestration for the einteger benchmark family.
# Responsibilities:
# - benchmark executable: emit metadata and write summaries
# - compare executable: compare persisted summaries only
# - this script: locate binaries, refresh stale summaries, invoke compare

function(parse_key_value_text text prefix)
  string(REPLACE "\r\n" "\n" _normalized "${text}")
  string(REPLACE "\r" "\n" _normalized "${_normalized}")
  string(REPLACE "\n" ";" _lines "${_normalized}")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "^([^=]+)=(.*)$")
      set("${prefix}_${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

function(query_benchmark_metadata benchmark_binary prefix)
  if(NOT EXISTS "${benchmark_binary}")
    message(FATAL_ERROR "Benchmark binary not found: ${benchmark_binary}")
  endif()

  execute_process(
    COMMAND "${benchmark_binary}" --build-metadata
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Failed to query benchmark metadata from ${benchmark_binary}: ${_error}")
  endif()

  parse_key_value_text("${_output}" "${prefix}")
  set("${prefix}_build_config" "${${prefix}_build_config}" PARENT_SCOPE)
  set("${prefix}_provenance_status" "${${prefix}_provenance_status}" PARENT_SCOPE)
  set("${prefix}_provenance_reason" "${${prefix}_provenance_reason}" PARENT_SCOPE)
  set("${prefix}_provenance_publishable" "${${prefix}_provenance_publishable}" PARENT_SCOPE)
  set("${prefix}_base_commit_hash" "${${prefix}_base_commit_hash}" PARENT_SCOPE)
  set("${prefix}_dirty_fingerprint" "${${prefix}_dirty_fingerprint}" PARENT_SCOPE)
  set("${prefix}_summary_schema" "${${prefix}_summary_schema}" PARENT_SCOPE)
  set("${prefix}_summary_path" "${${prefix}_summary_path}" PARENT_SCOPE)
  set("${prefix}_benchmark_binary" "${benchmark_binary}" PARENT_SCOPE)
endfunction()

function(find_benchmark_binary_for_config wanted_config out_var)
  set(_candidates "")
  list(APPEND _candidates "${CURRENT_BUILD_DIR}/${BENCHMARK_RELATIVE_PATH}")

  get_filename_component(_build_parent "${CURRENT_BUILD_DIR}" DIRECTORY)
  if(EXISTS "${_build_parent}")
    file(GLOB _sibling_dirs LIST_DIRECTORIES true "${_build_parent}/*")
    foreach(_dir IN LISTS _sibling_dirs)
      if(IS_DIRECTORY "${_dir}" AND NOT "${_dir}" STREQUAL "${CURRENT_BUILD_DIR}")
        list(APPEND _candidates "${_dir}/${BENCHMARK_RELATIVE_PATH}")
      endif()
    endforeach()
  endif()

  foreach(_candidate IN LISTS _candidates)
    if(NOT EXISTS "${_candidate}")
      continue()
    endif()
    query_benchmark_metadata("${_candidate}" "candidate")
    if("${candidate_build_config}" STREQUAL "${wanted_config}")
      set("${out_var}" "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set("${out_var}" "" PARENT_SCOPE)
endfunction()

function(parse_summary_file summary_path prefix out_ok)
  if(NOT EXISTS "${summary_path}")
    set("${out_ok}" FALSE PARENT_SCOPE)
    return()
  endif()

  file(STRINGS "${summary_path}" _lines)
  set(_row_count 0)
  set(_schema_version "")
  set(_build_config "")
  set(_provenance_status "")
  set(_provenance_publishable "")
  set(_base_commit_hash "")
  set(_dirty_fingerprint "")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "^row=")
      math(EXPR _row_count "${_row_count} + 1")
    endif()
    if(_line MATCHES "^([^=]+)=(.*)$")
      set("_${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}")
    endif()
  endforeach()

  set("${prefix}_schema_version" "${_schema_version}" PARENT_SCOPE)
  set("${prefix}_build_config" "${_build_config}" PARENT_SCOPE)
  set("${prefix}_provenance_status" "${_provenance_status}" PARENT_SCOPE)
  set("${prefix}_provenance_publishable" "${_provenance_publishable}" PARENT_SCOPE)
  set("${prefix}_base_commit_hash" "${_base_commit_hash}" PARENT_SCOPE)
  set("${prefix}_dirty_fingerprint" "${_dirty_fingerprint}" PARENT_SCOPE)
  set("${prefix}_row_count" "${_row_count}" PARENT_SCOPE)
  if(_row_count GREATER 0 AND NOT "${_schema_version}" STREQUAL "" AND NOT "${_build_config}" STREQUAL "")
    set("${out_ok}" TRUE PARENT_SCOPE)
  else()
    set("${out_ok}" FALSE PARENT_SCOPE)
  endif()
endfunction()

function(summary_is_stale benchmark_binary prefix out_stale out_reason)
  set(_summary_path "${${prefix}_summary_path}")
  if("${_summary_path}" STREQUAL "")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "summary path missing from benchmark metadata" PARENT_SCOPE)
    return()
  endif()

  parse_summary_file("${_summary_path}" "summary" _summary_ok)
  if(NOT _summary_ok)
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "summary missing or malformed" PARENT_SCOPE)
    return()
  endif()

  if(NOT "${summary_schema_version}" STREQUAL "${SUMMARY_SCHEMA_VERSION}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "schema mismatch" PARENT_SCOPE)
    return()
  endif()
  if(NOT "${summary_build_config}" STREQUAL "${${prefix}_build_config}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "build configuration mismatch" PARENT_SCOPE)
    return()
  endif()
  if(NOT "${summary_provenance_status}" STREQUAL "${${prefix}_provenance_status}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "provenance status mismatch" PARENT_SCOPE)
    return()
  endif()
  if(NOT "${summary_provenance_publishable}" STREQUAL "${${prefix}_provenance_publishable}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "provenance publishable mismatch" PARENT_SCOPE)
    return()
  endif()
  if(NOT "${summary_base_commit_hash}" STREQUAL "${${prefix}_base_commit_hash}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "base commit mismatch" PARENT_SCOPE)
    return()
  endif()
  if(NOT "${summary_dirty_fingerprint}" STREQUAL "${${prefix}_dirty_fingerprint}")
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "dirty fingerprint mismatch" PARENT_SCOPE)
    return()
  endif()

  file(TIMESTAMP "${_summary_path}" _summary_timestamp UTC "%Y-%m-%dT%H:%M:%SZ")
  file(TIMESTAMP "${benchmark_binary}" _binary_timestamp UTC "%Y-%m-%dT%H:%M:%SZ")
  if(_summary_timestamp STRLESS _binary_timestamp)
    set("${out_stale}" TRUE PARENT_SCOPE)
    set("${out_reason}" "summary is older than benchmark binary" PARENT_SCOPE)
    return()
  endif()

  set("${out_stale}" FALSE PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(ensure_current_summary benchmark_binary out_summary_path)
  query_benchmark_metadata("${benchmark_binary}" "meta")

  summary_is_stale("${benchmark_binary}" "meta" _is_stale _stale_reason)
  if(_is_stale)
    message(STATUS "Refreshing ${meta_build_config} einteger benchmark summary: ${_stale_reason}")
    execute_process(
      COMMAND "${benchmark_binary}" --write-summary-only
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _error
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "Failed to write summary for ${benchmark_binary}: ${_error}${_output}")
    endif()

    parse_summary_file("${meta_summary_path}" "summary" _summary_ok)
    if(NOT _summary_ok)
      message(FATAL_ERROR "Summary remained unreadable after refresh: ${meta_summary_path}")
    endif()
  else()
    message(STATUS "${meta_build_config} einteger benchmark summary is up to date")
  endif()

  set("${out_summary_path}" "${meta_summary_path}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED ACTION)
  message(FATAL_ERROR "ACTION is required")
endif()
if(NOT DEFINED CURRENT_BUILD_DIR)
  message(FATAL_ERROR "CURRENT_BUILD_DIR is required")
endif()
if(NOT DEFINED BENCHMARK_RELATIVE_PATH)
  message(FATAL_ERROR "BENCHMARK_RELATIVE_PATH is required")
endif()
if(NOT DEFINED SUMMARY_SCHEMA_VERSION)
  message(FATAL_ERROR "SUMMARY_SCHEMA_VERSION is required")
endif()

if(ACTION STREQUAL "refresh-config")
  if(NOT DEFINED WANTED_CONFIG)
    message(FATAL_ERROR "WANTED_CONFIG is required for refresh-config")
  endif()

  find_benchmark_binary_for_config("${WANTED_CONFIG}" _benchmark_binary)
  if("${_benchmark_binary}" STREQUAL "")
    message(FATAL_ERROR "${WANTED_CONFIG} benchmark binary not found")
  endif()

  ensure_current_summary("${_benchmark_binary}" _summary_path)
  message(STATUS "${WANTED_CONFIG} summary ready: ${_summary_path}")
  return()
endif()

if(ACTION STREQUAL "compare-builds")
  if(NOT DEFINED COMPARE_BINARY)
    message(FATAL_ERROR "COMPARE_BINARY is required for compare-builds")
  endif()

  find_benchmark_binary_for_config("Debug" _debug_binary)
  find_benchmark_binary_for_config("Release" _release_binary)
  if("${_debug_binary}" STREQUAL "")
    message(FATAL_ERROR "Debug benchmark binary not found")
  endif()
  if("${_release_binary}" STREQUAL "")
    message(FATAL_ERROR "Release benchmark binary not found")
  endif()

  ensure_current_summary("${_debug_binary}" _debug_summary)
  ensure_current_summary("${_release_binary}" _release_summary)

  execute_process(
    COMMAND "${COMPARE_BINARY}" --debug-summary "${_debug_summary}" --release-summary "${_release_summary}"
    RESULT_VARIABLE _compare_result
    OUTPUT_VARIABLE _compare_output
    ERROR_VARIABLE _compare_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_compare_output)
    message("${_compare_output}")
  endif()
  if(NOT _compare_result EQUAL 0)
    message(FATAL_ERROR "Benchmark comparison failed: ${_compare_error}${_compare_output}")
  endif()
  return()
endif()

message(FATAL_ERROR "Unknown ACTION: ${ACTION}")
