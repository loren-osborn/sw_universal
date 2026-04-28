#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(UNIVERSAL_HAS_CROSS_BUILD_BENCHMARK_PROVENANCE_HEADER)
#include <BenchmarkProvenance.hpp>
#else
// Normal benchmark builds do not require cross-build benchmark aggregation
// provenance. These fallback values are only used when that interface target
// is not linked into this executable.
#define UNIVERSAL_BENCH_BUILD_CONFIG "Unknown"
#define UNIVERSAL_BENCH_PROVENANCE_STATUS "unknown"
#define UNIVERSAL_BENCH_PROVENANCE_REASON "cross-build benchmark aggregation disabled"
#define UNIVERSAL_BENCH_PROVENANCE_BASE_COMMIT_HASH ""
#define UNIVERSAL_BENCH_PROVENANCE_DIRTY_FINGERPRINT ""
#define UNIVERSAL_BENCH_PROVENANCE_PUBLISHABLE "0"
#endif

namespace sw { namespace universal { namespace benchmark { namespace cross_build {

template<int SchemaVersion>
struct benchmark_metadata {
	std::string build_config;
	std::string provenance_status;
	std::string provenance_reason;
	std::string base_commit_hash;
	std::string dirty_fingerprint;
	bool provenance_publishable = false;
	int summary_schema = SchemaVersion;
	std::filesystem::path binary_path;
	std::filesystem::path summary_path;

	bool clean_publishable() const noexcept {
		return provenance_status == "clean" && provenance_publishable && !base_commit_hash.empty();
	}

	bool dirty_matchable() const noexcept {
		return provenance_status == "dirty_matchable" && !base_commit_hash.empty() && !dirty_fingerprint.empty();
	}
};

struct summary_row {
	std::string label;
	double overall_seconds = 0.0;
	double arithmetic_mean_ratio = 1.0;
	double geometric_mean_ratio = 1.0;
};

struct scenario_summary_row {
	std::string label;
	double seconds = 0.0;
	double relative_ratio = 1.0;
};

struct scenario_summary {
	std::string label;
	std::vector<scenario_summary_row> rows;
};

template<int SchemaVersion>
struct persisted_summary {
	int schema_version = SchemaVersion;
	std::string build_config;
	std::string provenance_status;
	std::string base_commit_hash;
	std::string dirty_fingerprint;
	bool provenance_publishable = false;
	std::int64_t timestamp_epoch = 0;
	std::string payload_name;
	std::vector<summary_row> rows;
	std::vector<scenario_summary> scenarios;
};

inline std::filesystem::path benchmark_summary_dir(const std::filesystem::path& binary_path) {
	return binary_path.parent_path() / "benchmark-results";
}

inline std::filesystem::path benchmark_summary_path(const std::filesystem::path& binary_path,
                                                    std::string_view build_config,
                                                    std::string_view summary_file_stem) {
	const std::string suffix = (build_config == "Release") ? "release" : "debug";
	return benchmark_summary_dir(binary_path) /
	       (std::string(summary_file_stem) + "_" + suffix + ".txt");
}

inline std::int64_t current_epoch_seconds() {
	using namespace std::chrono;
	return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

template<int SchemaVersion>
inline void write_persisted_summary(const std::filesystem::path& path,
                                    const persisted_summary<SchemaVersion>& summary) {
	std::filesystem::create_directories(path.parent_path());
	std::ofstream out(path);
	out << "schema_version=" << summary.schema_version << '\n';
	out << "build_config=" << summary.build_config << '\n';
	out << "provenance_status=" << summary.provenance_status << '\n';
	out << "provenance_publishable=" << (summary.provenance_publishable ? "true" : "false") << '\n';
	out << "base_commit_hash=" << summary.base_commit_hash << '\n';
	out << "dirty_fingerprint=" << summary.dirty_fingerprint << '\n';
	out << "timestamp_epoch=" << summary.timestamp_epoch << '\n';
	out << "payload=" << summary.payload_name << '\n';
	for (const auto& row : summary.rows) {
		out << "row="
		    << row.label << '|'
		    << row.overall_seconds << '|'
		    << row.arithmetic_mean_ratio << '|'
		    << row.geometric_mean_ratio << '\n';
	}
	for (const auto& scenario : summary.scenarios) {
		out << "scenario=" << scenario.label << '\n';
		for (const auto& row : scenario.rows) {
			out << "scenario_row="
			    << row.label << '|'
			    << row.seconds << '|'
			    << row.relative_ratio << '\n';
		}
	}
}

template<int SchemaVersion>
inline bool read_persisted_summary(const std::filesystem::path& path,
                                   persisted_summary<SchemaVersion>& summary) {
	std::ifstream in(path);
	if (!in) return false;

	summary = {};
	std::string line;
	scenario_summary* current_scenario = nullptr;
	while (std::getline(in, line)) {
		if (line.rfind("schema_version=", 0) == 0) {
			summary.schema_version = std::stoi(line.substr(std::string("schema_version=").size()));
			continue;
		}
		if (line.rfind("build_config=", 0) == 0) {
			summary.build_config = line.substr(std::string("build_config=").size());
			continue;
		}
		if (line.rfind("provenance_status=", 0) == 0) {
			summary.provenance_status = line.substr(std::string("provenance_status=").size());
			continue;
		}
		if (line.rfind("provenance_publishable=", 0) == 0) {
			summary.provenance_publishable =
				line.substr(std::string("provenance_publishable=").size()) == "true";
			continue;
		}
		if (line.rfind("base_commit_hash=", 0) == 0) {
			summary.base_commit_hash = line.substr(std::string("base_commit_hash=").size());
			continue;
		}
		if (line.rfind("dirty_fingerprint=", 0) == 0) {
			summary.dirty_fingerprint = line.substr(std::string("dirty_fingerprint=").size());
			continue;
		}
		if (line.rfind("timestamp_epoch=", 0) == 0) {
			summary.timestamp_epoch = std::stoll(line.substr(std::string("timestamp_epoch=").size()));
			continue;
		}
		if (line.rfind("payload=", 0) == 0) {
			summary.payload_name = line.substr(std::string("payload=").size());
			continue;
		}
		if (line.rfind("row=", 0) == 0) {
			std::stringstream row_stream(line.substr(std::string("row=").size()));
			std::string label;
			std::string seconds_text;
			std::string arithmetic_text;
			std::string geometric_text;
			if (!std::getline(row_stream, label, '|')) return false;
			if (!std::getline(row_stream, seconds_text, '|')) return false;
			if (!std::getline(row_stream, arithmetic_text, '|')) return false;
			if (!std::getline(row_stream, geometric_text, '|')) return false;
			summary.rows.push_back(summary_row{
				label,
				std::stod(seconds_text),
				std::stod(arithmetic_text),
				std::stod(geometric_text)
			});
			continue;
		}
		if (line.rfind("scenario=", 0) == 0) {
			summary.scenarios.push_back(scenario_summary{line.substr(std::string("scenario=").size()), {}});
			current_scenario = &summary.scenarios.back();
			continue;
		}
		if (line.rfind("scenario_row=", 0) == 0) {
			if (!current_scenario) return false;
			std::stringstream row_stream(line.substr(std::string("scenario_row=").size()));
			std::string label;
			std::string seconds_text;
			std::string relative_text;
			if (!std::getline(row_stream, label, '|')) return false;
			if (!std::getline(row_stream, seconds_text, '|')) return false;
			if (!std::getline(row_stream, relative_text, '|')) return false;
			current_scenario->rows.push_back(scenario_summary_row{
				label,
				std::stod(seconds_text),
				std::stod(relative_text)
			});
		}
	}
	return summary.schema_version == SchemaVersion &&
	       !summary.build_config.empty() &&
	       !summary.rows.empty() &&
	       !summary.scenarios.empty();
}

template<int SchemaVersion>
inline const summary_row* find_summary_row(const persisted_summary<SchemaVersion>& summary,
                                           std::string_view label) {
	for (const auto& row : summary.rows) {
		if (row.label == label) return &row;
	}
	return nullptr;
}

template<int SchemaVersion>
inline const scenario_summary* find_scenario_summary(const persisted_summary<SchemaVersion>& summary,
                                                     std::string_view label) {
	for (const auto& scenario : summary.scenarios) {
		if (scenario.label == label) return &scenario;
	}
	return nullptr;
}

inline const scenario_summary_row* find_scenario_summary_row(const scenario_summary& summary,
                                                             std::string_view label) {
	for (const auto& row : summary.rows) {
		if (row.label == label) return &row;
	}
	return nullptr;
}

template<int SchemaVersion>
inline void print_metadata(std::ostream& out, const benchmark_metadata<SchemaVersion>& metadata) {
	out << "build_config=" << metadata.build_config << '\n';
	out << "provenance_status=" << metadata.provenance_status << '\n';
	out << "provenance_reason=" << metadata.provenance_reason << '\n';
	out << "provenance_publishable=" << (metadata.provenance_publishable ? "true" : "false") << '\n';
	out << "base_commit_hash=" << metadata.base_commit_hash << '\n';
	out << "commit_hash=" << metadata.base_commit_hash << '\n';
	out << "dirty_fingerprint=" << metadata.dirty_fingerprint << '\n';
	out << "summary_schema=" << metadata.summary_schema << '\n';
	out << "benchmark_binary=" << metadata.binary_path.string() << '\n';
	out << "summary_path=" << metadata.summary_path.string() << '\n';
}

template<int SchemaVersion, typename SummaryPathFn>
inline benchmark_metadata<SchemaVersion> current_benchmark_metadata(
	const std::filesystem::path& binary_path,
	SummaryPathFn&& summary_path_fn) {
	benchmark_metadata<SchemaVersion> metadata;
	metadata.build_config = UNIVERSAL_BENCH_BUILD_CONFIG;
	metadata.provenance_status = UNIVERSAL_BENCH_PROVENANCE_STATUS;
	metadata.provenance_reason = UNIVERSAL_BENCH_PROVENANCE_REASON;
	metadata.base_commit_hash = UNIVERSAL_BENCH_PROVENANCE_BASE_COMMIT_HASH;
	metadata.dirty_fingerprint = UNIVERSAL_BENCH_PROVENANCE_DIRTY_FINGERPRINT;
	metadata.provenance_publishable = std::string_view{UNIVERSAL_BENCH_PROVENANCE_PUBLISHABLE} == "1";
	metadata.binary_path = binary_path;
	metadata.summary_path = summary_path_fn(binary_path, metadata.build_config);
	return metadata;
}

template<int SchemaVersion>
inline void print_provenance_banner(const benchmark_metadata<SchemaVersion>& metadata) {
	std::cout << "Build configuration: " << metadata.build_config << '\n';
	if (metadata.clean_publishable()) {
		std::cout << "Build provenance   : clean commit " << metadata.base_commit_hash << '\n';
		return;
	}

	if (metadata.dirty_matchable()) {
		std::cout << "Build provenance   : DIRTY BUT MATCHABLE\n";
		std::cout << "Base commit        : " << metadata.base_commit_hash << '\n';
		std::cout << "Dirty fingerprint  : " << metadata.dirty_fingerprint << '\n';
		std::cout << "Comparison policy  : unpublished/internal comparison only\n";
		return;
	}

	std::cout << "Build provenance   : " << metadata.provenance_status;
	if (!metadata.provenance_reason.empty()) {
		std::cout << " (" << metadata.provenance_reason << ')';
	}
	std::cout << '\n';
}

inline void print_benchmark_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " [--build-metadata] [--commit-hash] [--write-summary-only]\n";
}

template<int SchemaVersion>
inline void print_commit_hash_or_status(const benchmark_metadata<SchemaVersion>& metadata) {
	if (!metadata.base_commit_hash.empty()) {
		std::cout << metadata.base_commit_hash << '\n';
		return;
	}

	std::cout << metadata.provenance_status;
	if (!metadata.provenance_reason.empty()) {
		std::cout << ": " << metadata.provenance_reason;
	}
	std::cout << '\n';
}

enum class common_benchmark_cli_action {
	continue_run,
	exit_success,
	exit_failure,
};

template<int SchemaVersion>
inline common_benchmark_cli_action handle_common_benchmark_argument(
	std::string_view arg,
	const benchmark_metadata<SchemaVersion>& metadata,
	bool& write_summary_only,
	const char* argv0) {
	if (arg == "--build-metadata") {
		print_metadata(std::cout, metadata);
		return common_benchmark_cli_action::exit_success;
	}
	if (arg == "--commit-hash") {
		print_commit_hash_or_status(metadata);
		return common_benchmark_cli_action::exit_success;
	}
	if (arg == "--write-summary-only") {
		write_summary_only = true;
		return common_benchmark_cli_action::continue_run;
	}
	if (arg == "--help" || arg == "-h") {
		print_benchmark_usage(argv0);
		return common_benchmark_cli_action::exit_success;
	}

	std::cerr << "Unknown argument: " << arg << '\n';
	print_benchmark_usage(argv0);
	return common_benchmark_cli_action::exit_failure;
}

}}}} // namespace sw::universal::benchmark::cross_build
