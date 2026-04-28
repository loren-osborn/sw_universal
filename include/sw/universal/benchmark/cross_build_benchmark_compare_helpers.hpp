#pragma once

#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

namespace sw { namespace universal { namespace benchmark { namespace cross_build {

enum class comparison_mode {
	none,
	clean_match,
	dirty_match,
};

struct compare_cli_options {
	std::filesystem::path debug_summary_path;
	std::filesystem::path release_summary_path;
};

enum class compare_cli_parse_result {
	run,
	exit_success,
	exit_failure,
};

inline void print_compare_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " --debug-summary PATH --release-summary PATH\n";
}

inline compare_cli_parse_result parse_compare_cli(int argc, char** argv, compare_cli_options& options) {
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--debug-summary" && i + 1 < argc) {
			options.debug_summary_path = argv[++i];
			continue;
		}
		if (arg == "--release-summary" && i + 1 < argc) {
			options.release_summary_path = argv[++i];
			continue;
		}
		if (arg == "--help" || arg == "-h") {
			print_compare_usage(argv[0]);
			return compare_cli_parse_result::exit_success;
		}
		std::cerr << "Unknown argument: " << arg << '\n';
		print_compare_usage(argv[0]);
		return compare_cli_parse_result::exit_failure;
	}

	if (options.debug_summary_path.empty() || options.release_summary_path.empty()) {
		print_compare_usage(argv[0]);
		return compare_cli_parse_result::exit_failure;
	}

	return compare_cli_parse_result::run;
}

template<typename BenchmarkMetadata, typename PersistedSummary>
inline BenchmarkMetadata metadata_from_summary(const PersistedSummary& summary,
                                               const std::filesystem::path& summary_path) {
	BenchmarkMetadata metadata;
	metadata.build_config = summary.build_config;
	metadata.provenance_status = summary.provenance_status;
	metadata.base_commit_hash = summary.base_commit_hash;
	metadata.dirty_fingerprint = summary.dirty_fingerprint;
	metadata.provenance_publishable = summary.provenance_publishable;
	metadata.summary_path = summary_path;
	return metadata;
}

template<typename BenchmarkMetadata>
inline bool determine_comparison_mode(const BenchmarkMetadata& debug_meta,
                                      const BenchmarkMetadata& release_meta,
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

template<typename PersistedSummary, typename ReadSummaryFn>
inline bool load_and_validate_compare_summaries(std::string_view family_label,
                                                const compare_cli_options& options,
                                                ReadSummaryFn&& read_summary,
                                                PersistedSummary& debug_summary,
                                                PersistedSummary& release_summary) {
	if (!read_summary(options.debug_summary_path, debug_summary)) {
		std::cerr << family_label << ": Debug benchmark summary unreadable or malformed: "
		          << options.debug_summary_path.string() << '\n';
		return false;
	}
	if (!read_summary(options.release_summary_path, release_summary)) {
		std::cerr << family_label << ": Release benchmark summary unreadable or malformed: "
		          << options.release_summary_path.string() << '\n';
		return false;
	}
	if (debug_summary.build_config != "Debug") {
		std::cerr << family_label << ": Debug summary does not report build_config=Debug\n";
		return false;
	}
	if (release_summary.build_config != "Release") {
		std::cerr << family_label << ": Release summary does not report build_config=Release\n";
		return false;
	}
	if (debug_summary.payload_name != release_summary.payload_name) {
		std::cerr << family_label << ": Debug and Release benchmark summaries use different payload labels\n";
		return false;
	}
	return true;
}

template<typename BenchmarkMetadata>
inline void print_comparison_preamble(std::ostream& out,
                                      std::string_view title,
                                      const BenchmarkMetadata& debug_meta,
                                      const BenchmarkMetadata& release_meta,
                                      comparison_mode mode) {
	out << title << '\n';
	if (mode == comparison_mode::clean_match) {
		out << "Comparison mode: CLEAN MATCH\n";
		out << "Commit hash     : " << debug_meta.base_commit_hash << '\n';
	} else {
		out << "Comparison mode: DIRTY MATCH (unpublished/internal only)\n";
		out << "Base commit     : " << debug_meta.base_commit_hash << '\n';
		out << "Fingerprint     : " << debug_meta.dirty_fingerprint << '\n';
		out << "Provenance note : same base commit + same dirty working-tree fingerprint\n";
	}
	out << "Debug summary   : " << debug_meta.summary_path.string() << '\n';
	out << "Release summary : " << release_meta.summary_path.string() << '\n';
	out << '\n';
}

}}}} // namespace sw::universal::benchmark::cross_build
