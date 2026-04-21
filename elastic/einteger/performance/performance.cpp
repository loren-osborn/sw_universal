// performance.cpp: benchmark adaptive-precision integer storage backends
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <universal/utility/directives.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <BenchmarkProvenance.hpp>
#include <universal/internal/container/sso_vector.hpp>
#include <universal/number/einteger/einteger.hpp>

#include "einteger_performance_common.hpp"

namespace {

namespace perf = sw::universal::einteger_perf_detail;
using sw::universal::einteger;
using sw::universal::internal::sso_vector_default;
using sw::universal::internal::sso_vector_detail::default_inline_elems;

using limb_type = std::uint32_t;
using std_block_container = std::vector<limb_type>;
using sso_block_container = sso_vector_default<limb_type>;
using std_bigint = einteger<limb_type, std_block_container>;
using sso_bigint = einteger<limb_type, sso_block_container>;

static_assert(sw::universal::is_einteger<std_bigint>);
static_assert(sw::universal::is_einteger<sso_bigint>);

struct measurement {
	double seconds = 0.0;
	double ops_per_sec = 0.0;
};

struct scenario_spec {
	std::string_view label;
	std::size_t iteration_divisor;
};

struct scenario_results {
	std::string_view label;
	std::size_t iterations = 0;
	measurement std_vector_backed;
	measurement sso_vector_backed;
};

volatile std::uint64_t g_sink = 0;

inline std::size_t small_limb_count(std::size_t iteration) noexcept {
	switch (iteration % 10u) {
	case 0u: return 0u;
	case 1u:
	case 2u:
	case 3u: return 1u;
	case 4u:
	case 5u: return 2u;
	case 6u:
	case 7u: return 3u;
	case 8u: return 4u;
	default: return 5u;
	}
}

inline std::size_t medium_limb_count(std::size_t iteration) noexcept {
	return 4u + (iteration % 5u);
}

template<typename Integer>
Integer make_pattern_integer(std::uint32_t seed, std::size_t limbs) {
	Integer value;
	if (limbs == 0) return value;

	for (std::size_t i = 0; i < limbs; ++i) {
		std::uint64_t mixed =
			static_cast<std::uint64_t>(seed) * 0x9e3779b1ull +
			static_cast<std::uint64_t>(i + 1u) * 0x85ebca6bull +
			static_cast<std::uint64_t>(limbs) * 0xc2b2ae35ull;
		mixed ^= (mixed >> 16);
		auto block = static_cast<typename Integer::bt>(mixed);
		if (block == 0) {
			block = static_cast<typename Integer::bt>((seed | 1u) + static_cast<std::uint32_t>(i));
		}
		if (i + 1u == limbs && block == 0) {
			block = static_cast<typename Integer::bt>(1u);
		}
		value.setblock(static_cast<unsigned>(i), block);
	}
	return value;
}

template<typename Integer>
std::uint64_t reduce_hash(const Integer& value) {
	std::uint64_t acc = value.isneg() ? 0xd6e8feb86659fd93ull : 0x9e3779b97f4a7c15ull;
	acc ^= static_cast<std::uint64_t>(value.limbs()) * 0x94d049bb133111ebull;
	for (unsigned i = 0; i < value.limbs(); ++i) {
		acc ^= static_cast<std::uint64_t>(value.block(i)) + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_small_construction(std::size_t iterations) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		const auto value = make_pattern_integer<Integer>(1001u + static_cast<std::uint32_t>(i), small_limb_count(i));
		acc ^= reduce_hash(value);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_copy_read_mostly_small(std::size_t iterations) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		const auto src =
			make_pattern_integer<Integer>(3001u + static_cast<std::uint32_t>(i), (std::max)(std::size_t{1}, small_limb_count(i * 3u + 1u)));
		Integer copy = src;
		acc ^= reduce_hash(src);
		acc ^= reduce_hash(copy);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_add_sub_small_medium(std::size_t iterations) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		const auto a = make_pattern_integer<Integer>(5001u + static_cast<std::uint32_t>(i), 1u + (i % 3u));
		const auto b = make_pattern_integer<Integer>(7001u + static_cast<std::uint32_t>(i), medium_limb_count(i));
		const Integer sum = a + b;
		const Integer diff = sum - a;
		acc ^= reduce_hash(sum);
		acc ^= reduce_hash(diff);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_multiply_small_medium(std::size_t iterations) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		const auto a = make_pattern_integer<Integer>(11001u + static_cast<std::uint32_t>(i), 1u + (i % 3u));
		const auto b = make_pattern_integer<Integer>(13001u + static_cast<std::uint32_t>(i), 2u + (i % 3u));
		const Integer product = a * b;
		acc ^= reduce_hash(product);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_growth_shrink(std::size_t iterations) {
	constexpr std::array<int, 4> shifts{7, 19, 11, 23};
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Integer value = make_pattern_integer<Integer>(17001u + static_cast<std::uint32_t>(i), 1u + (i % 2u));
		for (std::size_t step = 0; step < shifts.size(); ++step) {
			value <<= shifts[step];
			value += make_pattern_integer<Integer>(18001u + static_cast<std::uint32_t>(i + step), 1u + ((i + step) % 2u));
			value >>= shifts[step] / 2;
		}
		acc ^= reduce_hash(value);
	}
	return acc;
}

template<typename Integer>
std::uint64_t workload_medium_copy_read_mostly(std::size_t iterations) {
	constexpr std::size_t medium_limbs = 24u;
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		const auto src = make_pattern_integer<Integer>(21001u + static_cast<std::uint32_t>(i), medium_limbs);
		Integer copy_a = src;
		Integer copy_b = src;
		acc ^= reduce_hash(src);
		acc ^= reduce_hash(copy_a);
		acc ^= reduce_hash(copy_b);
	}
	return acc;
}

template<typename Fn>
double measure_seconds(Fn&& fn) {
	const auto begin = std::chrono::steady_clock::now();
	fn();
	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(end - begin).count();
}

measurement benchmark_case(std::size_t iterations, std::uint64_t (*workload)(std::size_t)) {
	std::uint64_t result = 0;
	const double seconds = measure_seconds([&] {
		result = workload(iterations);
	});
	g_sink ^= result;
	return measurement{seconds, static_cast<double>(iterations) / seconds};
}

double ratio_vs_std_vector(const measurement& result, const measurement& baseline) {
	return result.seconds / baseline.seconds;
}

template<typename Member>
double arithmetic_mean_ratio(const std::vector<scenario_results>& results, Member member) {
	double total = 0.0;
	for (const auto& result : results) {
		total += ratio_vs_std_vector(result.*member, result.std_vector_backed);
	}
	return total / static_cast<double>(results.size());
}

template<typename Member>
double geometric_mean_ratio(const std::vector<scenario_results>& results, Member member) {
	double log_total = 0.0;
	for (const auto& result : results) {
		log_total += std::log(ratio_vs_std_vector(result.*member, result.std_vector_backed));
	}
	return std::exp(log_total / static_cast<double>(results.size()));
}

perf::benchmark_metadata current_benchmark_metadata(const std::filesystem::path& binary_path) {
	perf::benchmark_metadata metadata;
	metadata.build_config = UNIVERSAL_BENCH_BUILD_CONFIG;
	metadata.provenance_status = UNIVERSAL_BENCH_PROVENANCE_STATUS;
	metadata.provenance_reason = UNIVERSAL_BENCH_PROVENANCE_REASON;
	metadata.base_commit_hash = UNIVERSAL_BENCH_PROVENANCE_BASE_COMMIT_HASH;
	metadata.dirty_fingerprint = UNIVERSAL_BENCH_PROVENANCE_DIRTY_FINGERPRINT;
	metadata.provenance_publishable = std::string_view{UNIVERSAL_BENCH_PROVENANCE_PUBLISHABLE} == "1";
	metadata.summary_schema = perf::summary_schema_version;
	metadata.binary_path = binary_path;
	metadata.summary_path = perf::benchmark_summary_path(binary_path, metadata.build_config);
	return metadata;
}

void print_provenance_banner(const perf::benchmark_metadata& metadata) {
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

perf::persisted_summary run_bigint_benchmark(std::size_t iterations, bool emit_report) {
	constexpr std::array<scenario_spec, 6> scenarios{{
		{"small construction churn", 1u},
		{"small copy/read-mostly", 1u},
		{"add/subtract small+medium", 2u},
		{"multiply small+medium", 8u},
		{"growth/shrink shifts", 2u},
		{"medium copy/read-mostly", 12u},
	}};

	constexpr auto sso_inline_elems = default_inline_elems<limb_type, std::allocator<limb_type>>();
	constexpr std::string_view std_label = "einteger std::vector";
	constexpr std::string_view sso_label = "einteger sso_vector";

	constexpr int label_width = 40;
	constexpr int time_width = 16;
	constexpr int ops_width = 16;
	constexpr int ratio_width = 20;
	constexpr int summary_ratio_width = 18;

	const auto print_scenario_header = [&] {
		std::cout << std::left << std::setw(label_width) << "Backend"
		          << std::right << std::setw(time_width) << "Time(s)"
		          << std::setw(ops_width) << "Ops/sec"
		          << std::setw(ratio_width) << "Rel vs std::vector"
		          << '\n';
	};
	const auto print_scenario_row = [&](std::string_view label, const measurement& result, const measurement& baseline) {
		std::cout << std::left << std::setw(label_width) << label
		          << std::right << std::setw(time_width) << std::fixed << std::setprecision(6) << result.seconds
		          << std::setw(ops_width) << std::setprecision(0) << result.ops_per_sec
		          << std::setw(ratio_width) << std::setprecision(2) << ratio_vs_std_vector(result, baseline) << 'x'
		          << '\n';
	};

	std::vector<scenario_results> results(scenarios.size());
	for (std::size_t i = 0; i < results.size(); ++i) {
		results[i].label = scenarios[i].label;
		results[i].iterations = (std::max)(std::size_t{1}, iterations / scenarios[i].iteration_divisor);
	}

	results[0].std_vector_backed = benchmark_case(results[0].iterations, workload_small_construction<std_bigint>);
	results[0].sso_vector_backed = benchmark_case(results[0].iterations, workload_small_construction<sso_bigint>);

	results[1].std_vector_backed = benchmark_case(results[1].iterations, workload_copy_read_mostly_small<std_bigint>);
	results[1].sso_vector_backed = benchmark_case(results[1].iterations, workload_copy_read_mostly_small<sso_bigint>);

	results[2].std_vector_backed = benchmark_case(results[2].iterations, workload_add_sub_small_medium<std_bigint>);
	results[2].sso_vector_backed = benchmark_case(results[2].iterations, workload_add_sub_small_medium<sso_bigint>);

	results[3].std_vector_backed = benchmark_case(results[3].iterations, workload_multiply_small_medium<std_bigint>);
	results[3].sso_vector_backed = benchmark_case(results[3].iterations, workload_multiply_small_medium<sso_bigint>);

	results[4].std_vector_backed = benchmark_case(results[4].iterations, workload_growth_shrink<std_bigint>);
	results[4].sso_vector_backed = benchmark_case(results[4].iterations, workload_growth_shrink<sso_bigint>);

	results[5].std_vector_backed = benchmark_case(results[5].iterations, workload_medium_copy_read_mostly<std_bigint>);
	results[5].sso_vector_backed = benchmark_case(results[5].iterations, workload_medium_copy_read_mostly<sso_bigint>);

	if (emit_report) {
		std::cout << "\nPayload              : einteger<uint32_t>\n";
		std::cout << "Limb storage         : uint32_t words\n";
		std::cout << "sso limb inline elems: " << sso_inline_elems << '\n';
		for (const auto& scenario : results) {
			std::cout << "\nScenario: " << scenario.label << '\n';
			print_scenario_header();
			std::cout << std::string(label_width + time_width + ops_width + ratio_width, '-') << '\n';
			print_scenario_row(std_label, scenario.std_vector_backed, scenario.std_vector_backed);
			print_scenario_row(sso_label, scenario.sso_vector_backed, scenario.std_vector_backed);
		}
	}

	const auto sum_seconds = [](const auto& result_set, auto member) {
		double total = 0.0;
		for (const auto& result : result_set) {
			total += (result.*member).seconds;
		}
		return total;
	};

	const double overall_std_seconds = sum_seconds(results, &scenario_results::std_vector_backed);
	const double overall_sso_seconds = sum_seconds(results, &scenario_results::sso_vector_backed);

	const std::vector<perf::summary_row> summary_rows{
		{std::string(std_label), overall_std_seconds, 1.0, 1.0},
		{std::string(sso_label), overall_sso_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_vector_backed),
		 geometric_mean_ratio(results, &scenario_results::sso_vector_backed)},
	};

	std::vector<perf::scenario_summary> scenario_summaries;
	scenario_summaries.reserve(results.size());
	for (const auto& scenario : results) {
		scenario_summaries.push_back(perf::scenario_summary{
			std::string(scenario.label),
			{
				{std::string(std_label), scenario.std_vector_backed.seconds, 1.0},
				{std::string(sso_label), scenario.sso_vector_backed.seconds,
				 ratio_vs_std_vector(scenario.sso_vector_backed, scenario.std_vector_backed)},
			}
		});
	}

	if (emit_report) {
		std::cout << "\nOverall summary\n";
		std::cout << std::left << std::setw(label_width) << "Backend"
		          << std::right << std::setw(time_width) << "Time(s)"
		          << std::setw(ops_width) << "Ops/sec"
		          << std::setw(ratio_width) << "Rel vs std::vector"
		          << std::setw(summary_ratio_width) << "Arith mean"
		          << std::setw(summary_ratio_width) << "Geom mean"
		          << '\n';
		std::cout << std::string(label_width + time_width + ops_width + ratio_width + 2 * summary_ratio_width, '-') << '\n';
		for (const auto& row : summary_rows) {
			const double ops_per_sec = static_cast<double>(iterations) / row.overall_seconds;
			std::cout << std::left << std::setw(label_width) << row.label
			          << std::right << std::setw(time_width) << std::fixed << std::setprecision(6) << row.overall_seconds
			          << std::setw(ops_width) << std::setprecision(0) << ops_per_sec
			          << std::setw(ratio_width) << std::setprecision(2) << (row.overall_seconds / overall_std_seconds) << 'x'
			          << std::setw(summary_ratio_width) << row.arithmetic_mean_ratio << 'x'
			          << std::setw(summary_ratio_width) << row.geometric_mean_ratio << 'x'
			          << '\n';
		}
	}

	perf::persisted_summary summary;
	summary.build_config = UNIVERSAL_BENCH_BUILD_CONFIG;
	summary.provenance_status = UNIVERSAL_BENCH_PROVENANCE_STATUS;
	summary.provenance_publishable = std::string_view{UNIVERSAL_BENCH_PROVENANCE_PUBLISHABLE} == "1";
	summary.base_commit_hash = UNIVERSAL_BENCH_PROVENANCE_BASE_COMMIT_HASH;
	summary.dirty_fingerprint = UNIVERSAL_BENCH_PROVENANCE_DIRTY_FINGERPRINT;
	summary.timestamp_epoch = perf::current_epoch_seconds();
	summary.payload_name = "einteger<uint32_t>";
	summary.rows = summary_rows;
	summary.scenarios = std::move(scenario_summaries);
	return summary;
}

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " [--build-metadata] [--commit-hash] [--write-summary-only]\n";
}

} // namespace

int main(int argc, char** argv)
try {
	const std::filesystem::path binary_path =
		(argc > 0 && argv[0]) ? std::filesystem::absolute(argv[0]) : std::filesystem::current_path();
	const auto metadata = current_benchmark_metadata(binary_path);

	bool write_summary_only = false;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--build-metadata") {
			perf::print_metadata(std::cout, metadata);
			return EXIT_SUCCESS;
		}
		if (arg == "--commit-hash") {
			if (!metadata.base_commit_hash.empty()) {
				std::cout << metadata.base_commit_hash << '\n';
			}
			else {
				std::cout << metadata.provenance_status;
				if (!metadata.provenance_reason.empty()) {
					std::cout << ": " << metadata.provenance_reason;
				}
				std::cout << '\n';
			}
			return EXIT_SUCCESS;
		}
		if (arg == "--write-summary-only") {
			write_summary_only = true;
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

	if (!write_summary_only) {
		std::cout << "einteger BigInt storage benchmark\n";
		print_provenance_banner(metadata);
		std::cout << "Backends: std::vector<uint32_t> vs sso_vector<uint32_t>\n";
		std::cout << "Workloads: small construction churn, small copy/read-mostly,\n";
		std::cout << "           add/subtract small+medium, multiply small+medium,\n";
		std::cout << "           growth/shrink shifts, and medium copy/read-mostly\n";
		std::cout << '\n';
	}

	auto summary = run_bigint_benchmark(120000, !write_summary_only);
	summary.build_config = metadata.build_config;
	summary.provenance_status = metadata.provenance_status;
	summary.provenance_publishable = metadata.provenance_publishable;
	summary.base_commit_hash = metadata.base_commit_hash;
	summary.dirty_fingerprint = metadata.dirty_fingerprint;
	perf::write_persisted_summary(metadata.summary_path, summary);

	std::cout << "Saved benchmark summary: " << metadata.summary_path.string() << '\n';

	if (g_sink == 0) {
		std::cout << "optimizer guard\n";
	}

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
