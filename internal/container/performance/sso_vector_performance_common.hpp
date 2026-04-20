#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sw { namespace universal { namespace internal { namespace sso_vector_perf_detail {

inline constexpr int summary_schema_version = 2;

struct benchmark_metadata {
	std::string build_config;
	std::string provenance_status;
	std::string provenance_reason;
	std::string base_commit_hash;
	std::string dirty_fingerprint;
	bool provenance_publishable = false;
	int summary_schema = summary_schema_version;
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

struct persisted_summary {
	int schema_version = summary_schema_version;
	std::string build_config;
	std::string provenance_status;
	std::string base_commit_hash;
	std::string dirty_fingerprint;
	bool provenance_publishable = false;
	std::int64_t timestamp_epoch = 0;
	std::string payload_name;
	std::vector<summary_row> rows;
};

inline std::filesystem::path benchmark_summary_dir(const std::filesystem::path& binary_path) {
	return binary_path.parent_path() / "benchmark-results";
}

inline std::filesystem::path benchmark_summary_path(const std::filesystem::path& binary_path, std::string_view build_config) {
	const std::string suffix = (build_config == "Release") ? "release" : "debug";
	return benchmark_summary_dir(binary_path) / ("sso_vector_performance_" + suffix + ".txt");
}

inline std::int64_t current_epoch_seconds() {
	using namespace std::chrono;
	return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

inline void write_persisted_summary(const std::filesystem::path& path, const persisted_summary& summary) {
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
}

inline bool read_persisted_summary(const std::filesystem::path& path, persisted_summary& summary) {
	std::ifstream in(path);
	if (!in) return false;

	summary = {};
	std::string line;
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
			summary.provenance_publishable = line.substr(std::string("provenance_publishable=").size()) == "true";
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
		}
	}
	return summary.schema_version == summary_schema_version && !summary.build_config.empty() && !summary.rows.empty();
}

inline const summary_row* find_summary_row(const persisted_summary& summary, std::string_view label) {
	for (const auto& row : summary.rows) {
		if (row.label == label) return &row;
	}
	return nullptr;
}

inline void print_metadata(std::ostream& out, const benchmark_metadata& metadata) {
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

}}}} // namespace sw::universal::internal::sso_vector_perf_detail
