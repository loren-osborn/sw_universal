// sso_vector_performance_compare.cpp : aggregate Debug vs Release benchmark summaries
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <universal/utility/directives.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "sso_vector_performance_common.hpp"

namespace {

namespace perf = sw::universal::internal::sso_vector_perf_detail;

enum class comparison_mode {
	none,
	clean_match,
	dirty_match,
};

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

std::string quote_arg(const std::filesystem::path& path) {
	return "\"" + path.string() + "\"";
}

bool capture_command_output(const std::string& command, std::string& output, int& exit_code) {
	output.clear();
	FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		exit_code = -1;
		return false;
	}

	std::array<char, 512> buffer{};
	while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
		output += buffer.data();
	}
	exit_code = pclose(pipe);
	return true;
}

bool parse_metadata_output(const std::string& text, perf::benchmark_metadata& metadata) {
	metadata = {};
	std::stringstream ss(text);
	std::string line;
	while (std::getline(ss, line)) {
		if (line.rfind("build_config=", 0) == 0) {
			metadata.build_config = line.substr(std::string("build_config=").size());
		} else if (line.rfind("provenance_status=", 0) == 0) {
			metadata.provenance_status = line.substr(std::string("provenance_status=").size());
		} else if (line.rfind("provenance_reason=", 0) == 0) {
			metadata.provenance_reason = line.substr(std::string("provenance_reason=").size());
		} else if (line.rfind("provenance_publishable=", 0) == 0) {
			metadata.provenance_publishable = line.substr(std::string("provenance_publishable=").size()) == "true";
		} else if (line.rfind("base_commit_hash=", 0) == 0) {
			metadata.base_commit_hash = line.substr(std::string("base_commit_hash=").size());
		} else if (line.rfind("commit_hash=", 0) == 0) {
			if (metadata.base_commit_hash.empty()) {
				metadata.base_commit_hash = line.substr(std::string("commit_hash=").size());
			}
		} else if (line.rfind("dirty_fingerprint=", 0) == 0) {
			metadata.dirty_fingerprint = line.substr(std::string("dirty_fingerprint=").size());
		} else if (line.rfind("summary_schema=", 0) == 0) {
			metadata.summary_schema = std::stoi(line.substr(std::string("summary_schema=").size()));
		} else if (line.rfind("benchmark_binary=", 0) == 0) {
			metadata.binary_path = line.substr(std::string("benchmark_binary=").size());
		} else if (line.rfind("summary_path=", 0) == 0) {
			metadata.summary_path = line.substr(std::string("summary_path=").size());
		}
	}
	return !metadata.build_config.empty() && !metadata.binary_path.empty() && !metadata.summary_path.empty();
}

bool query_benchmark_metadata(const std::filesystem::path& binary_path,
                              perf::benchmark_metadata& metadata,
                              std::string& error) {
	if (!std::filesystem::exists(binary_path)) {
		error = "benchmark binary not found: " + binary_path.string();
		return false;
	}

	std::string output;
	int exit_code = 0;
	if (!capture_command_output(quote_arg(binary_path) + " --build-metadata", output, exit_code) || exit_code != 0) {
		error = "failed to query benchmark metadata from " + binary_path.string();
		return false;
	}
	if (!parse_metadata_output(output, metadata)) {
		error = "malformed benchmark metadata from " + binary_path.string();
		return false;
	}
	return true;
}

std::filesystem::path compare_build_tree_root(const std::filesystem::path& compare_binary) {
	return compare_binary.parent_path().parent_path().parent_path();
}

std::filesystem::path find_benchmark_binary_for_config(const std::filesystem::path& compare_binary,
                                                       std::string_view wanted_config) {
	const auto current_build_tree = compare_build_tree_root(compare_binary);
	const auto build_parent = current_build_tree.parent_path();

	auto try_candidate = [&](const std::filesystem::path& candidate) -> std::filesystem::path {
		perf::benchmark_metadata metadata;
		std::string error;
		if (!query_benchmark_metadata(candidate, metadata, error)) return {};
		return metadata.build_config == wanted_config ? candidate : std::filesystem::path{};
	};

	const auto local_candidate = current_build_tree / "internal/container/container_perf_sso_vector_performance";
	if (const auto found = try_candidate(local_candidate); !found.empty()) return found;

	if (!std::filesystem::exists(build_parent)) return {};
	for (const auto& entry : std::filesystem::directory_iterator(build_parent)) {
		if (!entry.is_directory()) continue;
		if (entry.path() == current_build_tree) continue;
		const auto candidate = entry.path() / "internal/container/container_perf_sso_vector_performance";
		if (const auto found = try_candidate(candidate); !found.empty()) return found;
	}
	return {};
}

bool summary_is_stale(const perf::persisted_summary& summary,
                      const perf::benchmark_metadata& metadata,
                      std::string& reason) {
	if (summary.schema_version != perf::summary_schema_version) {
		reason = "schema mismatch";
		return true;
	}
	if (summary.build_config != metadata.build_config) {
		reason = "build configuration mismatch";
		return true;
	}
	if (summary.provenance_status != metadata.provenance_status) {
		reason = "provenance status mismatch";
		return true;
	}
	if (summary.provenance_publishable != metadata.provenance_publishable) {
		reason = "provenance publishable flag mismatch";
		return true;
	}
	if (summary.base_commit_hash != metadata.base_commit_hash) {
		reason = "base commit hash mismatch";
		return true;
	}
	if (summary.dirty_fingerprint != metadata.dirty_fingerprint) {
		reason = "dirty fingerprint mismatch";
		return true;
	}
	if (summary.schema_version != metadata.summary_schema) {
		reason = "benchmark metadata schema mismatch";
		return true;
	}
	if (!std::filesystem::exists(metadata.summary_path)) {
		reason = "summary file missing";
		return true;
	}
	if (std::filesystem::last_write_time(metadata.summary_path) < std::filesystem::last_write_time(metadata.binary_path)) {
		reason = "summary is older than benchmark binary";
		return true;
	}
	return false;
}

bool refresh_summary(const perf::benchmark_metadata& metadata, std::string& error) {
	std::string output;
	int exit_code = 0;
	if (!capture_command_output(quote_arg(metadata.binary_path) + " --write-summary-only", output, exit_code) || exit_code != 0) {
		error = "failed to regenerate summary for " + metadata.binary_path.string();
		if (!output.empty()) {
			error += "\n" + output;
		}
		return false;
	}
	return true;
}

bool ensure_current_summary(const perf::benchmark_metadata& metadata,
                            perf::persisted_summary& summary,
                            std::string& error) {
	if (!perf::read_persisted_summary(metadata.summary_path, summary)) {
		if (!refresh_summary(metadata, error)) return false;
		if (!perf::read_persisted_summary(metadata.summary_path, summary)) {
			error = "summary remained unreadable after regeneration: " + metadata.summary_path.string();
			return false;
		}
		return true;
	}

	std::string stale_reason;
	if (!summary_is_stale(summary, metadata, stale_reason)) return true;

	if (!refresh_summary(metadata, error)) return false;
	if (!perf::read_persisted_summary(metadata.summary_path, summary)) {
		error = "summary remained unreadable after regeneration: " + metadata.summary_path.string();
		return false;
	}
	if (summary_is_stale(summary, metadata, stale_reason)) {
		error = "summary remained stale after regeneration: " + stale_reason;
		return false;
	}
	return true;
}

bool determine_comparison_mode(const perf::benchmark_metadata& debug_meta,
                               const perf::benchmark_metadata& release_meta,
                               comparison_mode& mode,
                               std::string& reason) {
	mode = comparison_mode::none;
	reason.clear();

	const bool debug_clean = debug_meta.clean_publishable();
	const bool release_clean = release_meta.clean_publishable();
	const bool debug_dirty = debug_meta.dirty_matchable();
	const bool release_dirty = release_meta.dirty_matchable();

	if (debug_meta.provenance_status == "dirty_matchable" && !debug_dirty) {
		reason = "Cannot compare: Debug dirty build has no fingerprint";
		return false;
	}
	if (release_meta.provenance_status == "dirty_matchable" && !release_dirty) {
		reason = "Cannot compare: Release dirty build has no fingerprint";
		return false;
	}

	if (debug_clean && release_clean) {
		if (debug_meta.base_commit_hash != release_meta.base_commit_hash) {
			reason = "Debug and Release benchmark binaries were built from different commits";
			return false;
		}
		mode = comparison_mode::clean_match;
		return true;
	}

	if (debug_dirty && release_dirty) {
		if (debug_meta.base_commit_hash != release_meta.base_commit_hash) {
			reason = "Cannot compare: base commits differ";
			return false;
		}
		if (debug_meta.dirty_fingerprint != release_meta.dirty_fingerprint) {
			reason = "Cannot compare: dirty fingerprints differ";
			return false;
		}
		mode = comparison_mode::dirty_match;
		return true;
	}

	if ((!debug_clean && !debug_dirty) || (!release_clean && !release_dirty)) {
		reason = "Cannot compare: provenance unavailable";
		return false;
	}

	if (debug_clean != release_clean || debug_dirty != release_dirty) {
		reason = "Cannot compare: one build is clean and the other is dirty";
		return false;
	}

	reason = "Cannot compare: provenance unavailable";
	return false;
}

void print_combined_report(const perf::benchmark_metadata& debug_meta,
                           const perf::persisted_summary& debug_summary,
                           const perf::benchmark_metadata& release_meta,
                           const perf::persisted_summary& release_summary,
                           comparison_mode mode) {
	std::cout << "sso_vector Debug vs Release benchmark comparison\n";
	if (mode == comparison_mode::clean_match) {
		std::cout << "Comparison mode: CLEAN MATCH\n";
		std::cout << "Commit hash     : " << debug_meta.base_commit_hash << '\n';
	} else {
		std::cout << "Comparison mode: DIRTY MATCH (unpublished/internal only)\n";
		std::cout << "Base commit     : " << debug_meta.base_commit_hash << '\n';
		std::cout << "Fingerprint     : " << debug_meta.dirty_fingerprint << '\n';
		std::cout << "Provenance note : same base commit + same dirty working-tree fingerprint\n";
	}
	std::cout << "Debug summary   : " << debug_meta.summary_path.string() << '\n';
	std::cout << "Release summary : " << release_meta.summary_path.string() << '\n';
	std::cout << '\n';
	std::cout << std::left << std::setw(32) << "Container"
	          << std::right << std::setw(14) << "Debug Time"
	          << std::setw(14) << "Release Time"
	          << std::setw(14) << "Debug Arith"
	          << std::setw(14) << "Release Arith"
	          << std::setw(14) << "Debug Geom"
	          << std::setw(14) << "Release Geom"
	          << '\n';
	std::cout << std::string(116, '-') << '\n';
	for (const auto& debug_row : debug_summary.rows) {
		const auto* release_row = perf::find_summary_row(release_summary, debug_row.label);
		if (!release_row) continue;
		std::cout << std::left << std::setw(32) << debug_row.label
		          << std::right << std::setw(14) << std::fixed << std::setprecision(6) << debug_row.overall_seconds
		          << std::setw(14) << release_row->overall_seconds
		          << std::setw(14) << std::setprecision(2) << debug_row.arithmetic_mean_ratio << 'x'
		          << std::setw(14) << release_row->arithmetic_mean_ratio << 'x'
		          << std::setw(14) << debug_row.geometric_mean_ratio << 'x'
		          << std::setw(14) << release_row->geometric_mean_ratio << 'x'
		          << '\n';
	}
}

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " [--debug-binary PATH] [--release-binary PATH]\n";
}

} // namespace

int main(int argc, char** argv)
try {
	std::filesystem::path debug_binary;
	std::filesystem::path release_binary;

	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--debug-binary" && i + 1 < argc) {
			debug_binary = argv[++i];
			continue;
		}
		if (arg == "--release-binary" && i + 1 < argc) {
			release_binary = argv[++i];
			continue;
		}
		if (arg == "--help" || arg == "-h") {
			print_usage(argv[0]);
			return EXIT_SUCCESS;
		}
		std::cerr << "Unknown argument: " << arg << '\n';
		print_usage(argv[0]);
		return EXIT_FAILURE;
	}

	const std::filesystem::path compare_binary =
		(argc > 0 && argv[0]) ? std::filesystem::absolute(argv[0]) : std::filesystem::current_path();

	if (debug_binary.empty()) {
		debug_binary = find_benchmark_binary_for_config(compare_binary, "Debug");
	}
	if (release_binary.empty()) {
		release_binary = find_benchmark_binary_for_config(compare_binary, "Release");
	}

	if (debug_binary.empty()) {
		std::cerr << "Debug benchmark binary not found\n";
		return EXIT_FAILURE;
	}
	if (release_binary.empty()) {
		std::cerr << "Release benchmark binary not found\n";
		return EXIT_FAILURE;
	}

	perf::benchmark_metadata debug_meta;
	perf::benchmark_metadata release_meta;
	std::string error;
	if (!query_benchmark_metadata(debug_binary, debug_meta, error)) {
		std::cerr << error << '\n';
		return EXIT_FAILURE;
	}
	if (!query_benchmark_metadata(release_binary, release_meta, error)) {
		std::cerr << error << '\n';
		return EXIT_FAILURE;
	}

	comparison_mode mode = comparison_mode::none;
	if (!determine_comparison_mode(debug_meta, release_meta, mode, error)) {
		if (error == "Cannot compare: provenance unavailable") {
			if (!debug_meta.clean_publishable() && !debug_meta.dirty_matchable()) {
				std::cerr << "Debug benchmark binary reports " << debug_meta.provenance_status;
				if (!debug_meta.provenance_reason.empty()) {
					std::cerr << " (" << debug_meta.provenance_reason << ')';
				}
				std::cerr << '\n';
			}
			if (!release_meta.clean_publishable() && !release_meta.dirty_matchable()) {
				std::cerr << "Release benchmark binary reports " << release_meta.provenance_status;
				if (!release_meta.provenance_reason.empty()) {
					std::cerr << " (" << release_meta.provenance_reason << ')';
				}
				std::cerr << '\n';
			}
		}
		std::cerr << error << '\n';
		return EXIT_FAILURE;
	}

	perf::persisted_summary debug_summary;
	perf::persisted_summary release_summary;
	if (!ensure_current_summary(debug_meta, debug_summary, error)) {
		std::cerr << error << '\n';
		return EXIT_FAILURE;
	}
	if (!ensure_current_summary(release_meta, release_summary, error)) {
		std::cerr << error << '\n';
		return EXIT_FAILURE;
	}
	if (debug_summary.payload_name != release_summary.payload_name) {
		std::cerr << "Debug and Release benchmark summaries use different payload labels\n";
		return EXIT_FAILURE;
	}

	print_combined_report(debug_meta, debug_summary, release_meta, release_summary, mode);
	return EXIT_SUCCESS;
}
catch (const std::exception& ex) {
	std::cerr << "Exception: " << ex.what() << '\n';
	return EXIT_FAILURE;
}
catch (...) {
	std::cerr << "Unknown exception\n";
	return EXIT_FAILURE;
}
