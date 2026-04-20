// sso_vector_performance.cpp : internal speed benchmark for small-vector workloads
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
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <BenchmarkProvenance.hpp>
#include <universal/internal/container/sso_vector.hpp>

#include "sso_vector_performance_common.hpp"

namespace {

using sw::universal::internal::sso_cow_vector;
using sw::universal::internal::sso_vector;
using sw::universal::internal::zero_inline_policy;
namespace perf = sw::universal::internal::sso_vector_perf_detail;

template<typename T>
struct payload_ops;

template<>
struct payload_ops<std::uint32_t> {
	static std::uint32_t make(std::uint32_t seed, std::size_t index) noexcept {
		return seed + static_cast<std::uint32_t>(index * 17u + 3u);
	}

	static std::uint64_t hash(std::uint32_t value) noexcept {
		return value;
	}

	static void mutate(std::uint32_t& value, std::uint32_t delta) noexcept {
		value += delta;
	}
};

template<typename Container>
void append_pattern(Container& c, std::size_t count, std::uint32_t seed) {
	using value_type = typename Container::value_type;
	for (std::size_t i = 0; i < count; ++i) {
		c.push_back(payload_ops<value_type>::make(seed, i));
	}
}

template<typename Container>
std::uint64_t reduce_hash(const Container& c) {
	using value_type = typename Container::value_type;
	std::uint64_t acc = static_cast<std::uint64_t>(c.size()) * 1315423911ull;
	for (const auto& value : c) {
		acc ^= payload_ops<value_type>::hash(value) + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2);
	}
	return acc;
}

inline std::size_t distributed_size(std::size_t iteration, std::size_t inline_capacity) noexcept {
	const std::size_t threshold = inline_capacity == 0 ? 4 : inline_capacity;
	switch (iteration % 8u) {
	case 0u: return 0u;
	case 1u: return 1u + (iteration % 3u);
	case 2u: return threshold > 1 ? threshold - 1u : 1u;
	case 3u: return threshold;
	case 4u: return threshold + 1u;
	case 5u: return threshold + 2u + (iteration % 2u);
	case 6u: return threshold / 2u + 1u;
	default: return 2u + (iteration % ((threshold > 3u) ? 4u : threshold + 1u));
	}
}

template<typename Container>
std::uint64_t workload_ephemeral(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container v;
		append_pattern(v, distributed_size(i, inline_capacity), static_cast<std::uint32_t>(101u + i));
		acc ^= reduce_hash(v);
	}
	return acc;
}

template<typename Container>
std::uint64_t workload_copy_read_mostly(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container src;
		append_pattern(src, distributed_size(i * 3u + 1u, inline_capacity), static_cast<std::uint32_t>(2001u + i));
		Container copy = src;
		acc ^= reduce_hash(src);
		acc ^= reduce_hash(copy);
	}
	return acc;
}

template<typename Container>
std::uint64_t workload_copy_then_mutate(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container src;
		append_pattern(src, (std::max)(std::size_t{1}, distributed_size(i * 5u + 2u, inline_capacity)), static_cast<std::uint32_t>(4001u + i));
		Container copy = src;
		if (!copy.empty()) {
			const std::size_t pos = (i * 7u) % copy.size();
			auto value = static_cast<typename Container::value_type>(copy[pos]);
			payload_ops<typename Container::value_type>::mutate(value, static_cast<std::uint32_t>(i + 1u));
			copy[pos] = std::move(value);
		}
		acc ^= reduce_hash(src);
		acc ^= reduce_hash(copy);
	}
	return acc;
}

template<typename Container>
std::uint64_t workload_inline_threshold(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t acc = 0;
	const std::size_t base = inline_capacity == 0 ? 4u : inline_capacity;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container v;
		for (std::size_t n : {base > 1u ? base - 1u : 1u, base, base + 1u, base + 2u}) {
			v.clear();
			append_pattern(v, n, static_cast<std::uint32_t>(7001u + i * 13u + n));
			acc ^= reduce_hash(v);
		}
	}
	return acc;
}

template<typename Container>
std::uint64_t workload_reuse_pool(std::size_t iterations, std::size_t inline_capacity) {
	std::array<Container, 16> pool{};
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container& v = pool[i % pool.size()];
		v.clear();
		append_pattern(v, distributed_size(i * 11u + 3u, inline_capacity), static_cast<std::uint32_t>(9001u + i));
		if (!v.empty() && (i % 3u) == 0u) {
			auto value = static_cast<typename Container::value_type>(v.back());
			payload_ops<typename Container::value_type>::mutate(value, static_cast<std::uint32_t>(17u + i));
			v.back() = std::move(value);
		}
		acc ^= reduce_hash(v);
	}
	return acc;
}

template<typename Container>
std::uint64_t workload_large_copy_read_mostly(std::size_t iterations, std::size_t) {
	constexpr std::size_t large_size = 1000u;
	std::uint64_t acc = 0;
	for (std::size_t i = 0; i < iterations; ++i) {
		Container src;
		src.reserve(large_size);
		append_pattern(src, large_size, static_cast<std::uint32_t>(12001u + i * 3u));
		Container copy_a = src;
		Container copy_b = src;
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

volatile std::uint64_t g_sink = 0;

struct scenario_spec {
	std::string_view label;
	std::size_t iteration_divisor;
};

struct measurement {
	double seconds = 0.0;
	double ops_per_sec = 0.0;
};

struct scenario_results {
	std::string_view label;
	std::size_t iterations = 0;
	measurement std_vector;
	measurement sso_non_cow_zero_inline;
	measurement sso_non_cow_inline;
	measurement sso_non_cow_double_inline;
	measurement sso_cow_zero_inline;
	measurement sso_cow_inline;
	measurement sso_cow_double_inline;
};

template<typename Container>
measurement benchmark_case(std::size_t iterations, std::size_t inline_capacity,
                           std::uint64_t (*workload)(std::size_t, std::size_t)) {
	std::uint64_t result = 0;
	const double seconds = measure_seconds([&] {
		result = workload(iterations, inline_capacity);
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
		total += ratio_vs_std_vector(result.*member, result.std_vector);
	}
	return total / static_cast<double>(results.size());
}

template<typename Member>
double geometric_mean_ratio(const std::vector<scenario_results>& results, Member member) {
	double log_total = 0.0;
	for (const auto& result : results) {
		log_total += std::log(ratio_vs_std_vector(result.*member, result.std_vector));
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

template<typename T, std::size_t DefaultInlineElems>
perf::persisted_summary run_payload_benchmark(std::string_view payload_name, std::size_t iterations, bool emit_report) {
	using std_vector_t = std::vector<T>;
	constexpr std::size_t default_inline = DefaultInlineElems;
	constexpr std::size_t benchmark_inline = (std::max)(std::size_t{2}, default_inline);
	constexpr std::size_t benchmark_double_inline = benchmark_inline * 2u;
	using sso_zero_t = sso_vector<T, 0, std::allocator<T>, zero_inline_policy::allow>;
	using sso_inline_t = sso_vector<T, benchmark_inline>;
	using sso_double_t = sso_vector<T, benchmark_double_inline>;
	using sso_cow_zero_t = sso_cow_vector<T, 0, std::allocator<T>, zero_inline_policy::allow>;
	using sso_cow_inline_t = sso_cow_vector<T, benchmark_inline>;
	using sso_cow_double_t = sso_cow_vector<T, benchmark_double_inline>;

	constexpr std::array<scenario_spec, 6> scenarios{{
		{"ephemeral churn", 1u},
		{"copy/read-mostly", 2u},
		{"copy-then-mutate", 2u},
		{"inline-threshold transitions", 4u},
		{"pooled reuse", 1u},
		{"large copy/read-mostly (~1000)", 12u},
	}};

	const std::string zero_inline_label = "sso_vector inline=0";
	const std::string inline_label = "sso_vector inline=" + std::to_string(benchmark_inline);
	const std::string double_inline_label = "sso_vector inline=" + std::to_string(benchmark_double_inline);
	const std::string cow_zero_inline_label = "sso_cow_vector inline=0";
	const std::string cow_inline_label = "sso_cow_vector inline=" + std::to_string(benchmark_inline);
	const std::string cow_double_inline_label = "sso_cow_vector inline=" + std::to_string(benchmark_double_inline);

	const auto print_scenario_header = [] {
		std::cout << std::left << std::setw(32) << "Container"
		          << std::right << std::setw(16) << "Time(s)"
		          << std::setw(16) << "Ops/sec"
		          << std::setw(20) << "Rel vs std::vector"
		          << '\n';
	};
	const auto print_scenario_row = [](std::string_view label, const measurement& result, const measurement& baseline) {
		std::cout << std::left << std::setw(32) << label
		          << std::right << std::setw(16) << std::fixed << std::setprecision(6) << result.seconds
		          << std::setw(16) << std::setprecision(0) << result.ops_per_sec
		          << std::setw(20) << std::setprecision(2) << ratio_vs_std_vector(result, baseline) << 'x'
		          << '\n';
	};

	std::vector<scenario_results> results(scenarios.size());
	for (std::size_t i = 0; i < results.size(); ++i) {
		results[i].label = scenarios[i].label;
		results[i].iterations = (std::max)(std::size_t{1}, iterations / scenarios[i].iteration_divisor);
	}

	results[0].std_vector = benchmark_case<std_vector_t>(results[0].iterations, benchmark_inline, workload_ephemeral<std_vector_t>);
	results[0].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[0].iterations, 0u, workload_ephemeral<sso_zero_t>);
	results[0].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[0].iterations, benchmark_inline, workload_ephemeral<sso_inline_t>);
	results[0].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[0].iterations, benchmark_double_inline, workload_ephemeral<sso_double_t>);
	results[0].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[0].iterations, 0u, workload_ephemeral<sso_cow_zero_t>);
	results[0].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[0].iterations, benchmark_inline, workload_ephemeral<sso_cow_inline_t>);
	results[0].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[0].iterations, benchmark_double_inline, workload_ephemeral<sso_cow_double_t>);

	results[1].std_vector = benchmark_case<std_vector_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<std_vector_t>);
	results[1].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[1].iterations, 0u, workload_copy_read_mostly<sso_zero_t>);
	results[1].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<sso_inline_t>);
	results[1].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[1].iterations, benchmark_double_inline, workload_copy_read_mostly<sso_double_t>);
	results[1].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[1].iterations, 0u, workload_copy_read_mostly<sso_cow_zero_t>);
	results[1].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<sso_cow_inline_t>);
	results[1].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[1].iterations, benchmark_double_inline, workload_copy_read_mostly<sso_cow_double_t>);

	results[2].std_vector = benchmark_case<std_vector_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<std_vector_t>);
	results[2].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[2].iterations, 0u, workload_copy_then_mutate<sso_zero_t>);
	results[2].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<sso_inline_t>);
	results[2].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[2].iterations, benchmark_double_inline, workload_copy_then_mutate<sso_double_t>);
	results[2].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[2].iterations, 0u, workload_copy_then_mutate<sso_cow_zero_t>);
	results[2].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<sso_cow_inline_t>);
	results[2].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[2].iterations, benchmark_double_inline, workload_copy_then_mutate<sso_cow_double_t>);

	results[3].std_vector = benchmark_case<std_vector_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<std_vector_t>);
	results[3].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[3].iterations, 0u, workload_inline_threshold<sso_zero_t>);
	results[3].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<sso_inline_t>);
	results[3].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[3].iterations, benchmark_double_inline, workload_inline_threshold<sso_double_t>);
	results[3].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[3].iterations, 0u, workload_inline_threshold<sso_cow_zero_t>);
	results[3].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<sso_cow_inline_t>);
	results[3].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[3].iterations, benchmark_double_inline, workload_inline_threshold<sso_cow_double_t>);

	results[4].std_vector = benchmark_case<std_vector_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<std_vector_t>);
	results[4].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[4].iterations, 0u, workload_reuse_pool<sso_zero_t>);
	results[4].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<sso_inline_t>);
	results[4].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[4].iterations, benchmark_double_inline, workload_reuse_pool<sso_double_t>);
	results[4].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[4].iterations, 0u, workload_reuse_pool<sso_cow_zero_t>);
	results[4].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<sso_cow_inline_t>);
	results[4].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[4].iterations, benchmark_double_inline, workload_reuse_pool<sso_cow_double_t>);

	results[5].std_vector = benchmark_case<std_vector_t>(results[5].iterations, 0u, workload_large_copy_read_mostly<std_vector_t>);
	results[5].sso_non_cow_zero_inline = benchmark_case<sso_zero_t>(results[5].iterations, 0u, workload_large_copy_read_mostly<sso_zero_t>);
	results[5].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[5].iterations, benchmark_inline, workload_large_copy_read_mostly<sso_inline_t>);
	results[5].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[5].iterations, benchmark_double_inline, workload_large_copy_read_mostly<sso_double_t>);
	results[5].sso_cow_zero_inline = benchmark_case<sso_cow_zero_t>(results[5].iterations, 0u, workload_large_copy_read_mostly<sso_cow_zero_t>);
	results[5].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[5].iterations, benchmark_inline, workload_large_copy_read_mostly<sso_cow_inline_t>);
	results[5].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[5].iterations, benchmark_double_inline, workload_large_copy_read_mostly<sso_cow_double_t>);

	if (emit_report) {
		std::cout << "\nPayload: " << payload_name << '\n';
		std::cout << "Default inline elems      : " << default_inline << '\n';
		std::cout << "Benchmark inline elems    : " << benchmark_inline << '\n';
		std::cout << "Benchmark 2x inline elems : " << benchmark_double_inline << '\n';

		for (const auto& scenario : results) {
			std::cout << "\nScenario: " << scenario.label << '\n';
			print_scenario_header();
			std::cout << std::string(84, '-') << '\n';
			print_scenario_row("std::vector", scenario.std_vector, scenario.std_vector);
			print_scenario_row(zero_inline_label, scenario.sso_non_cow_zero_inline, scenario.std_vector);
			print_scenario_row(inline_label, scenario.sso_non_cow_inline, scenario.std_vector);
			print_scenario_row(double_inline_label, scenario.sso_non_cow_double_inline, scenario.std_vector);
			print_scenario_row(cow_zero_inline_label, scenario.sso_cow_zero_inline, scenario.std_vector);
			print_scenario_row(cow_inline_label, scenario.sso_cow_inline, scenario.std_vector);
			print_scenario_row(cow_double_inline_label, scenario.sso_cow_double_inline, scenario.std_vector);
		}
	}

	const auto sum_seconds = [](const auto& result_set, auto member) {
		double total = 0.0;
		for (const auto& result : result_set) {
			total += (result.*member).seconds;
		}
		return total;
	};

	const double overall_std_seconds = sum_seconds(results, &scenario_results::std_vector);
	const double overall_zero_inline_seconds = sum_seconds(results, &scenario_results::sso_non_cow_zero_inline);
	const double overall_inline_seconds = sum_seconds(results, &scenario_results::sso_non_cow_inline);
	const double overall_double_seconds = sum_seconds(results, &scenario_results::sso_non_cow_double_inline);
	const double overall_cow_zero_inline_seconds = sum_seconds(results, &scenario_results::sso_cow_zero_inline);
	const double overall_cow_inline_seconds = sum_seconds(results, &scenario_results::sso_cow_inline);
	const double overall_cow_double_seconds = sum_seconds(results, &scenario_results::sso_cow_double_inline);

	const std::vector<perf::summary_row> summary_rows{
		{"std::vector", overall_std_seconds, 1.0, 1.0},
		{zero_inline_label, overall_zero_inline_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_non_cow_zero_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_non_cow_zero_inline)},
		{inline_label, overall_inline_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_non_cow_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_non_cow_inline)},
		{double_inline_label, overall_double_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_non_cow_double_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_non_cow_double_inline)},
		{cow_zero_inline_label, overall_cow_zero_inline_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_cow_zero_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_cow_zero_inline)},
		{cow_inline_label, overall_cow_inline_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_cow_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_cow_inline)},
		{cow_double_inline_label, overall_cow_double_seconds,
		 arithmetic_mean_ratio(results, &scenario_results::sso_cow_double_inline),
		 geometric_mean_ratio(results, &scenario_results::sso_cow_double_inline)},
	};

	const auto make_scenario_summary = [&](const scenario_results& scenario) {
		return perf::scenario_summary{
			std::string(scenario.label),
			{
				{"std::vector", scenario.std_vector.seconds, 1.0},
				{zero_inline_label, scenario.sso_non_cow_zero_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_non_cow_zero_inline, scenario.std_vector)},
				{inline_label, scenario.sso_non_cow_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_non_cow_inline, scenario.std_vector)},
				{double_inline_label, scenario.sso_non_cow_double_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_non_cow_double_inline, scenario.std_vector)},
				{cow_zero_inline_label, scenario.sso_cow_zero_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_cow_zero_inline, scenario.std_vector)},
				{cow_inline_label, scenario.sso_cow_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_cow_inline, scenario.std_vector)},
				{cow_double_inline_label, scenario.sso_cow_double_inline.seconds,
				 ratio_vs_std_vector(scenario.sso_cow_double_inline, scenario.std_vector)},
			}
		};
	};

	std::vector<perf::scenario_summary> scenario_summaries;
	scenario_summaries.reserve(results.size());
	for (const auto& scenario : results) {
		scenario_summaries.push_back(make_scenario_summary(scenario));
	}

	if (emit_report) {
		std::cout << "\nOverall summary\n";
		std::cout << std::left << std::setw(32) << "Container"
		          << std::right << std::setw(16) << "Time(s)"
		          << std::setw(16) << "Ops/sec"
		          << std::setw(20) << "Rel vs std::vector"
		          << std::setw(20) << "Arith mean"
		          << std::setw(20) << "Geom mean"
		          << '\n';
		std::cout << std::string(124, '-') << '\n';
		for (const auto& row : summary_rows) {
			const double ops_per_sec = static_cast<double>(iterations) / row.overall_seconds;
			std::cout << std::left << std::setw(32) << row.label
			          << std::right << std::setw(16) << std::fixed << std::setprecision(6) << row.overall_seconds
			          << std::setw(16) << std::setprecision(0) << ops_per_sec
			          << std::setw(20) << std::setprecision(2) << (row.overall_seconds / overall_std_seconds) << 'x'
			          << std::setw(20) << row.arithmetic_mean_ratio << 'x'
			          << std::setw(20) << row.geometric_mean_ratio << 'x'
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
	summary.payload_name = std::string(payload_name);
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
	using namespace sw::universal::internal;

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
			} else {
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

	constexpr std::size_t u32_default_inline =
		sso_vector_detail::default_inline_elems<std::uint32_t, std::allocator<std::uint32_t>>();

	// Runtime responsibility stays narrow here: measure, report local results, and write the
	// current build's cached summary. Cross-build refresh and comparison live in CMake/script tooling.
	if (!write_summary_only) {
		std::cout << "sso_vector / sso_cow_vector performance benchmark\n";
		print_provenance_banner(metadata);
		std::cout << "Workloads: ephemeral churn, copy/read-mostly, copy-then-mutate,\n";
		std::cout << "           inline-threshold transitions, reusable pooled vectors,\n";
		std::cout << "           and a larger copy/read-mostly scenario\n";
		std::cout << "Rows include same-family no-inline-buffer baselines via explicit inline=0 opt-in.\n";
		std::cout << '\n';
	}

	auto summary =
		run_payload_benchmark<std::uint32_t, u32_default_inline>("uint32_t", 120000, !write_summary_only);
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
catch (char const* msg) {
	std::cerr << "Caught exception: " << msg << '\n';
	return EXIT_FAILURE;
}
catch (const std::runtime_error& err) {
	std::cerr << "Uncaught runtime exception: " << err.what() << '\n';
	return EXIT_FAILURE;
}
catch (...) {
	std::cerr << "Caught unknown exception\n";
	return EXIT_FAILURE;
}
