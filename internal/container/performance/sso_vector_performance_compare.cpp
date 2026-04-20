// sso_vector_performance_compare.cpp : compare persisted Debug vs Release benchmark summaries
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <universal/utility/directives.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "sso_vector_performance_common.hpp"

namespace {

namespace perf = sw::universal::internal::sso_vector_perf_detail;

enum class comparison_mode {
	none,
	clean_match,
	dirty_match,
};

perf::benchmark_metadata metadata_from_summary(const perf::persisted_summary& summary,
                                               const std::filesystem::path& summary_path) {
	perf::benchmark_metadata metadata;
	metadata.build_config = summary.build_config;
	metadata.provenance_status = summary.provenance_status;
	metadata.base_commit_hash = summary.base_commit_hash;
	metadata.dirty_fingerprint = summary.dirty_fingerprint;
	metadata.provenance_publishable = summary.provenance_publishable;
	metadata.summary_path = summary_path;
	return metadata;
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
		reason = "Cannot compare: Debug dirty summary has no fingerprint";
		return false;
	}
	if (release_meta.provenance_status == "dirty_matchable" && !release_dirty) {
		reason = "Cannot compare: Release dirty summary has no fingerprint";
		return false;
	}

	if (debug_clean && release_clean) {
		if (debug_meta.base_commit_hash != release_meta.base_commit_hash) {
			reason = "Debug and Release benchmark summaries were produced from different commits";
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

	for (const auto& debug_scenario : debug_summary.scenarios) {
		const perf::scenario_summary* release_scenario = nullptr;
		for (const auto& candidate : release_summary.scenarios) {
			if (candidate.label == debug_scenario.label) {
				release_scenario = &candidate;
				break;
			}
		}
		if (!release_scenario) continue;

		std::cout << "\nScenario: " << debug_scenario.label << '\n';
		std::cout << std::left << std::setw(32) << "Container"
		          << std::right << std::setw(14) << "Debug Time"
		          << std::setw(14) << "Release Time"
		          << std::setw(14) << "Debug Rel"
		          << std::setw(14) << "Release Rel"
		          << '\n';
		std::cout << std::string(88, '-') << '\n';
		for (const auto& debug_row : debug_scenario.rows) {
			const perf::scenario_summary_row* release_row = nullptr;
			for (const auto& candidate : release_scenario->rows) {
				if (candidate.label == debug_row.label) {
					release_row = &candidate;
					break;
				}
			}
			if (!release_row) continue;
			std::cout << std::left << std::setw(32) << debug_row.label
			          << std::right << std::setw(14) << std::fixed << std::setprecision(6) << debug_row.seconds
			          << std::setw(14) << release_row->seconds
			          << std::setw(14) << std::setprecision(2) << debug_row.relative_ratio << 'x'
			          << std::setw(14) << release_row->relative_ratio << 'x'
			          << '\n';
		}
	}
}

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " --debug-summary PATH --release-summary PATH\n";
}

} // namespace

int main(int argc, char** argv)
try {
	std::filesystem::path debug_summary_path;
	std::filesystem::path release_summary_path;

	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--debug-summary" && i + 1 < argc) {
			debug_summary_path = argv[++i];
			continue;
		}
		if (arg == "--release-summary" && i + 1 < argc) {
			release_summary_path = argv[++i];
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

	if (debug_summary_path.empty() || release_summary_path.empty()) {
		print_usage(argv[0]);
		return EXIT_FAILURE;
	}

	perf::persisted_summary debug_summary;
	perf::persisted_summary release_summary;
	if (!perf::read_persisted_summary(debug_summary_path, debug_summary)) {
		std::cerr << "Debug benchmark summary unreadable or malformed: "
		          << debug_summary_path.string() << '\n';
		return EXIT_FAILURE;
	}
	if (!perf::read_persisted_summary(release_summary_path, release_summary)) {
		std::cerr << "Release benchmark summary unreadable or malformed: "
		          << release_summary_path.string() << '\n';
		return EXIT_FAILURE;
	}
	if (debug_summary.build_config != "Debug") {
		std::cerr << "Debug summary does not report build_config=Debug\n";
		return EXIT_FAILURE;
	}
	if (release_summary.build_config != "Release") {
		std::cerr << "Release summary does not report build_config=Release\n";
		return EXIT_FAILURE;
	}
	if (debug_summary.payload_name != release_summary.payload_name) {
		std::cerr << "Debug and Release benchmark summaries use different payload labels\n";
		return EXIT_FAILURE;
	}

	const auto debug_meta = metadata_from_summary(debug_summary, debug_summary_path);
	const auto release_meta = metadata_from_summary(release_summary, release_summary_path);

	comparison_mode mode = comparison_mode::none;
	std::string error;
	if (!determine_comparison_mode(debug_meta, release_meta, mode, error)) {
		std::cerr << error << '\n';
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
