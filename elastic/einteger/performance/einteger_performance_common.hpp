#pragma once

#include <universal/benchmark/cross_build_benchmark_helpers.hpp>

namespace sw { namespace universal { namespace einteger_perf_detail {

namespace cross = sw::universal::benchmark::cross_build;

inline constexpr int summary_schema_version = 1;

using benchmark_metadata = cross::benchmark_metadata<summary_schema_version>;
using summary_row = cross::summary_row;
using scenario_summary_row = cross::scenario_summary_row;
using scenario_summary = cross::scenario_summary;
using persisted_summary = cross::persisted_summary<summary_schema_version>;
using common_benchmark_cli_action = cross::common_benchmark_cli_action;

inline std::filesystem::path benchmark_summary_dir(const std::filesystem::path& binary_path) {
	return cross::benchmark_summary_dir(binary_path);
}

inline std::filesystem::path benchmark_summary_path(const std::filesystem::path& binary_path,
                                                    std::string_view build_config) {
	return cross::benchmark_summary_path(binary_path, build_config, "einteger_performance");
}

inline std::int64_t current_epoch_seconds() {
	return cross::current_epoch_seconds();
}

inline void write_persisted_summary(const std::filesystem::path& path, const persisted_summary& summary) {
	cross::write_persisted_summary(path, summary);
}

inline bool read_persisted_summary(const std::filesystem::path& path, persisted_summary& summary) {
	return cross::read_persisted_summary(path, summary);
}

inline const summary_row* find_summary_row(const persisted_summary& summary, std::string_view label) {
	return cross::find_summary_row(summary, label);
}

inline const scenario_summary* find_scenario_summary(const persisted_summary& summary, std::string_view label) {
	return cross::find_scenario_summary(summary, label);
}

inline const scenario_summary_row* find_scenario_summary_row(const scenario_summary& summary, std::string_view label) {
	return cross::find_scenario_summary_row(summary, label);
}

inline void print_metadata(std::ostream& out, const benchmark_metadata& metadata) {
	cross::print_metadata(out, metadata);
}

inline benchmark_metadata current_benchmark_metadata(const std::filesystem::path& binary_path) {
	return cross::current_benchmark_metadata<summary_schema_version>(
		binary_path,
		[](const std::filesystem::path& path, std::string_view build_config) {
			return benchmark_summary_path(path, build_config);
		});
}

inline void print_provenance_banner(const benchmark_metadata& metadata) {
	cross::print_provenance_banner(metadata);
}

inline void print_usage(const char* argv0) {
	cross::print_benchmark_usage(argv0);
}

inline common_benchmark_cli_action handle_common_benchmark_argument(
	std::string_view arg,
	const benchmark_metadata& metadata,
	bool& write_summary_only,
	const char* argv0) {
	return cross::handle_common_benchmark_argument(arg, metadata, write_summary_only, argv0);
}

}}} // namespace sw::universal::einteger_perf_detail
