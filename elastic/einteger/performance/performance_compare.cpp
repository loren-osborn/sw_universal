// performance_compare.cpp : compare persisted Debug vs Release einteger benchmark summaries
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

#include <universal/benchmark/cross_build_benchmark_compare_helpers.hpp>

#include "einteger_performance_common.hpp"

namespace {

namespace perf = sw::universal::einteger_perf_detail;
namespace cross = sw::universal::benchmark::cross_build;

void print_combined_report(const perf::benchmark_metadata& debug_meta,
                           const perf::persisted_summary& debug_summary,
                           const perf::benchmark_metadata& release_meta,
                           const perf::persisted_summary& release_summary,
                           cross::comparison_mode mode) {
	constexpr int label_width = 40;
	constexpr int metric_width = 14;

	cross::print_comparison_preamble(
		std::cout,
		"einteger BigInt Debug vs Release benchmark comparison",
		debug_meta,
		release_meta,
		mode);

	std::cout << std::left << std::setw(label_width) << "Backend"
	          << std::right << std::setw(metric_width) << "Debug Time"
	          << std::setw(metric_width) << "Release Time"
	          << std::setw(metric_width) << "Debug Arith"
	          << std::setw(metric_width) << "Release Arith"
	          << std::setw(metric_width) << "Debug Geom"
	          << std::setw(metric_width) << "Release Geom"
	          << '\n';
	std::cout << std::string(label_width + 6 * metric_width, '-') << '\n';
	for (const auto& debug_row : debug_summary.rows) {
		const auto* release_row = perf::find_summary_row(release_summary, debug_row.label);
		if (!release_row) continue;
		std::cout << std::left << std::setw(label_width) << debug_row.label
		          << std::right << std::setw(metric_width) << std::fixed << std::setprecision(6) << debug_row.overall_seconds
		          << std::setw(metric_width) << release_row->overall_seconds
		          << std::setw(metric_width) << std::setprecision(2) << debug_row.arithmetic_mean_ratio << 'x'
		          << std::setw(metric_width) << release_row->arithmetic_mean_ratio << 'x'
		          << std::setw(metric_width) << debug_row.geometric_mean_ratio << 'x'
		          << std::setw(metric_width) << release_row->geometric_mean_ratio << 'x'
		          << '\n';
	}

	for (const auto& debug_scenario : debug_summary.scenarios) {
		const auto* release_scenario = perf::find_scenario_summary(release_summary, debug_scenario.label);
		if (!release_scenario) continue;

		std::cout << "\nWorkload: " << debug_scenario.label << '\n';
		std::cout << std::left << std::setw(label_width) << "Backend"
		          << std::right << std::setw(metric_width) << "Debug Time"
		          << std::setw(metric_width) << "Release Time"
		          << std::setw(metric_width) << "Debug Rel"
		          << std::setw(metric_width) << "Release Rel"
		          << '\n';
		std::cout << std::string(label_width + 4 * metric_width, '-') << '\n';
		for (const auto& debug_row : debug_scenario.rows) {
			const auto* release_row = perf::find_scenario_summary_row(*release_scenario, debug_row.label);
			if (!release_row) continue;
			std::cout << std::left << std::setw(label_width) << debug_row.label
			          << std::right << std::setw(metric_width) << std::fixed << std::setprecision(6) << debug_row.seconds
			          << std::setw(metric_width) << release_row->seconds
			          << std::setw(metric_width) << std::setprecision(2) << debug_row.relative_ratio << 'x'
			          << std::setw(metric_width) << release_row->relative_ratio << 'x'
			          << '\n';
		}
	}
}

} // namespace

int main(int argc, char** argv)
try {
	cross::compare_cli_options options;
	switch (cross::parse_compare_cli(argc, argv, options)) {
	case cross::compare_cli_parse_result::run:
		break;
	case cross::compare_cli_parse_result::exit_success:
		return EXIT_SUCCESS;
	case cross::compare_cli_parse_result::exit_failure:
		return EXIT_FAILURE;
	}

	perf::persisted_summary debug_summary;
	perf::persisted_summary release_summary;
	if (!cross::load_and_validate_compare_summaries(
			"einteger",
			options,
			[](const auto& path, auto& summary) { return perf::read_persisted_summary(path, summary); },
			debug_summary,
			release_summary)) {
		return EXIT_FAILURE;
	}

	const auto debug_meta = cross::metadata_from_summary<perf::benchmark_metadata>(debug_summary, options.debug_summary_path);
	const auto release_meta = cross::metadata_from_summary<perf::benchmark_metadata>(release_summary, options.release_summary_path);

	cross::comparison_mode mode = cross::comparison_mode::none;
	std::string error;
	if (!cross::determine_comparison_mode(debug_meta, release_meta, mode, error)) {
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
