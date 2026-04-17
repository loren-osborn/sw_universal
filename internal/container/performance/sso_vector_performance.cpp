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
using sw::universal::internal::sso_vector_default;

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

template<typename Container>
std::uint64_t run_all_workloads(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t acc = 0;
	acc ^= workload_ephemeral<Container>(iterations, inline_capacity);
	acc ^= workload_copy_read_mostly<Container>(iterations / 2u, inline_capacity);
	acc ^= workload_copy_then_mutate<Container>(iterations / 2u, inline_capacity);
	acc ^= workload_inline_threshold<Container>((std::max)(std::size_t{1}, iterations / 4u), inline_capacity);
	acc ^= workload_reuse_pool<Container>(iterations, inline_capacity);
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

template<typename Container>
double benchmark_case(std::size_t iterations, std::size_t inline_capacity) {
	std::uint64_t result = 0;
	const double seconds = measure_seconds([&] {
		result = run_all_workloads<Container>(iterations, inline_capacity);
	});
	g_sink ^= result;
	return seconds;
}

template<typename T, std::size_t DefaultInlineElems>
void run_payload_benchmark(std::string_view payload_name, std::size_t iterations) {
	using std_vector_t = std::vector<T>;
	using sso_default_t = sso_vector_default<T>;
	using sso_double_t = sso_vector<T, DefaultInlineElems * 2u>;

	constexpr std::size_t default_inline = DefaultInlineElems;
	constexpr std::size_t doubled_inline = DefaultInlineElems * 2u;

	std::cout << "\nPayload: " << payload_name << '\n';
	std::cout << "Default inline elems : " << default_inline << '\n';
	std::cout << "Doubled inline elems : " << doubled_inline << '\n';
	std::cout << std::left << std::setw(28) << "Container"
	          << std::right << std::setw(16) << "Time(s)"
	          << std::setw(16) << "Ops/sec"
	          << '\n';
	std::cout << std::string(60, '-') << '\n';

	const auto print_row = [iterations](std::string_view label, double seconds) {
		const double ops_per_sec = static_cast<double>(iterations) / seconds;
		std::cout << std::left << std::setw(28) << label
		          << std::right << std::setw(16) << std::fixed << std::setprecision(6) << seconds
		          << std::setw(16) << std::setprecision(0) << ops_per_sec
		          << '\n';
	};

	print_row("std::vector", benchmark_case<std_vector_t>(iterations, default_inline));
	print_row("sso_vector default", benchmark_case<sso_default_t>(iterations, default_inline));
	print_row("sso_vector 2x inline", benchmark_case<sso_double_t>(iterations, doubled_inline));
}

} // namespace

int main()
try {
	using namespace sw::universal::internal;

	constexpr std::size_t u32_default_inline =
		sso_vector_detail::default_inline_elems<std::uint32_t, std::allocator<std::uint32_t>>();

	std::cout << "sso_vector small-sequence performance benchmark\n";
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
