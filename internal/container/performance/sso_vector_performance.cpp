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
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

#include <universal/internal/container/sso_vector.hpp>

namespace {

using sw::universal::internal::sso_vector;
using sw::universal::internal::sso_cow_vector;

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
	// Mostly small vectors, with regular spill just below/at/above the inline threshold.
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
	measurement sso_non_cow_inline;
	measurement sso_non_cow_double_inline;
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

template<typename T, std::size_t DefaultInlineElems>
void run_payload_benchmark(std::string_view payload_name, std::size_t iterations) {
	using std_vector_t = std::vector<T>;
	constexpr std::size_t default_inline = DefaultInlineElems;
	constexpr std::size_t benchmark_inline = (std::max)(std::size_t{2}, default_inline);
	constexpr std::size_t benchmark_double_inline = benchmark_inline * 2u;
	using sso_inline_t = sso_vector<T, benchmark_inline>;
	using sso_double_t = sso_vector<T, benchmark_double_inline>;
	using sso_cow_inline_t = sso_cow_vector<T, benchmark_inline>;
	using sso_cow_double_t = sso_cow_vector<T, benchmark_double_inline>;

	constexpr std::array<scenario_spec, 5> scenarios{{
		{"ephemeral churn", 1u},
		{"copy/read-mostly", 2u},
		{"copy-then-mutate", 2u},
		{"inline-threshold transitions", 4u},
		{"pooled reuse", 1u},
	}};

	std::cout << "\nPayload: " << payload_name << '\n';
	std::cout << "Default inline elems      : " << default_inline << '\n';
	std::cout << "Benchmark inline elems    : " << benchmark_inline << '\n';
	std::cout << "Benchmark 2x inline elems : " << benchmark_double_inline << '\n';

	const auto inline_label = "sso_vector inline=" + std::to_string(benchmark_inline);
	const auto double_inline_label = "sso_vector inline=" + std::to_string(benchmark_double_inline);
	const auto cow_inline_label = "sso_cow_vector inline=" + std::to_string(benchmark_inline);
	const auto cow_double_inline_label = "sso_cow_vector inline=" + std::to_string(benchmark_double_inline);

	const auto print_header = [] {
		std::cout << std::left << std::setw(32) << "Container"
		          << std::right << std::setw(16) << "Time(s)"
		          << std::setw(16) << "Ops/sec"
		          << std::setw(20) << "Rel vs std::vector"
		          << '\n';
	};
	const auto print_row = [](std::string_view label, const measurement& result, const measurement& baseline) {
		const double ratio = result.seconds / baseline.seconds;
		std::cout << std::left << std::setw(32) << label
		          << std::right << std::setw(16) << std::fixed << std::setprecision(6) << result.seconds
		          << std::setw(16) << std::setprecision(0) << result.ops_per_sec
		          << std::setw(20) << std::setprecision(2) << ratio << 'x'
		          << '\n';
	};

	std::array<scenario_results, 5> results{};

	for (std::size_t i = 0; i < results.size(); ++i) {
		results[i].label = scenarios[i].label;
		results[i].iterations = (std::max)(std::size_t{1}, iterations / scenarios[i].iteration_divisor);
	}

	results[0].std_vector = benchmark_case<std_vector_t>(results[0].iterations, benchmark_inline, workload_ephemeral<std_vector_t>);
	results[0].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[0].iterations, benchmark_inline, workload_ephemeral<sso_inline_t>);
	results[0].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[0].iterations, benchmark_double_inline, workload_ephemeral<sso_double_t>);
	results[0].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[0].iterations, benchmark_inline, workload_ephemeral<sso_cow_inline_t>);
	results[0].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[0].iterations, benchmark_double_inline, workload_ephemeral<sso_cow_double_t>);

	results[1].std_vector = benchmark_case<std_vector_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<std_vector_t>);
	results[1].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<sso_inline_t>);
	results[1].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[1].iterations, benchmark_double_inline, workload_copy_read_mostly<sso_double_t>);
	results[1].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[1].iterations, benchmark_inline, workload_copy_read_mostly<sso_cow_inline_t>);
	results[1].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[1].iterations, benchmark_double_inline, workload_copy_read_mostly<sso_cow_double_t>);

	results[2].std_vector = benchmark_case<std_vector_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<std_vector_t>);
	results[2].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<sso_inline_t>);
	results[2].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[2].iterations, benchmark_double_inline, workload_copy_then_mutate<sso_double_t>);
	results[2].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[2].iterations, benchmark_inline, workload_copy_then_mutate<sso_cow_inline_t>);
	results[2].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[2].iterations, benchmark_double_inline, workload_copy_then_mutate<sso_cow_double_t>);

	results[3].std_vector = benchmark_case<std_vector_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<std_vector_t>);
	results[3].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<sso_inline_t>);
	results[3].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[3].iterations, benchmark_double_inline, workload_inline_threshold<sso_double_t>);
	results[3].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[3].iterations, benchmark_inline, workload_inline_threshold<sso_cow_inline_t>);
	results[3].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[3].iterations, benchmark_double_inline, workload_inline_threshold<sso_cow_double_t>);

	results[4].std_vector = benchmark_case<std_vector_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<std_vector_t>);
	results[4].sso_non_cow_inline = benchmark_case<sso_inline_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<sso_inline_t>);
	results[4].sso_non_cow_double_inline = benchmark_case<sso_double_t>(results[4].iterations, benchmark_double_inline, workload_reuse_pool<sso_double_t>);
	results[4].sso_cow_inline = benchmark_case<sso_cow_inline_t>(results[4].iterations, benchmark_inline, workload_reuse_pool<sso_cow_inline_t>);
	results[4].sso_cow_double_inline = benchmark_case<sso_cow_double_t>(results[4].iterations, benchmark_double_inline, workload_reuse_pool<sso_cow_double_t>);

	for (const auto& scenario : results) {
		std::cout << "\nScenario: " << scenario.label << '\n';
		print_header();
		std::cout << std::string(84, '-') << '\n';
		print_row("std::vector", scenario.std_vector, scenario.std_vector);
		print_row(inline_label, scenario.sso_non_cow_inline, scenario.std_vector);
		print_row(double_inline_label, scenario.sso_non_cow_double_inline, scenario.std_vector);
		print_row(cow_inline_label, scenario.sso_cow_inline, scenario.std_vector);
		print_row(cow_double_inline_label, scenario.sso_cow_double_inline, scenario.std_vector);
	}

	const auto sum_seconds = [](const auto& result_set, auto member) {
		double total = 0.0;
		for (const auto& result : result_set) {
			total += (result.*member).seconds;
		}
		return total;
	};

	const measurement overall_std{
		sum_seconds(results, &scenario_results::std_vector),
		static_cast<double>(iterations) / sum_seconds(results, &scenario_results::std_vector)
	};
	const measurement overall_inline{
		sum_seconds(results, &scenario_results::sso_non_cow_inline),
		static_cast<double>(iterations) / sum_seconds(results, &scenario_results::sso_non_cow_inline)
	};
	const measurement overall_double{
		sum_seconds(results, &scenario_results::sso_non_cow_double_inline),
		static_cast<double>(iterations) / sum_seconds(results, &scenario_results::sso_non_cow_double_inline)
	};
	const measurement overall_cow_inline{
		sum_seconds(results, &scenario_results::sso_cow_inline),
		static_cast<double>(iterations) / sum_seconds(results, &scenario_results::sso_cow_inline)
	};
	const measurement overall_cow_double{
		sum_seconds(results, &scenario_results::sso_cow_double_inline),
		static_cast<double>(iterations) / sum_seconds(results, &scenario_results::sso_cow_double_inline)
	};

	std::cout << "\nOverall summary\n";
	print_header();
	std::cout << std::string(84, '-') << '\n';
	print_row("std::vector", overall_std, overall_std);
	print_row(inline_label, overall_inline, overall_std);
	print_row(double_inline_label, overall_double, overall_std);
	print_row(cow_inline_label, overall_cow_inline, overall_std);
	print_row(cow_double_inline_label, overall_cow_double, overall_std);
}

} // namespace

int main()
try {
	using namespace sw::universal::internal;

	constexpr std::size_t u32_default_inline =
		sso_vector_detail::default_inline_elems<std::uint32_t, std::allocator<std::uint32_t>>();

	std::cout << "sso_vector / sso_cow_vector small-sequence performance benchmark\n";
	std::cout << "Workloads: ephemeral churn, copy/read-mostly, copy-then-mutate,\n";
	std::cout << "           inline-threshold transitions, and reusable pooled vectors\n";
	std::cout << '\n';

	run_payload_benchmark<std::uint32_t, u32_default_inline>("uint32_t", 120000);

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
