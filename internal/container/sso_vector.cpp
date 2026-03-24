// sso_vector.cpp: unit tests for sso_vector
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <array>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <universal/internal/container/sso_vector.hpp>
#include <universal/verification/test_status.hpp>

namespace {

struct TestContext {
	const char* impl = "";
	int& failures;
};

void check(const TestContext& ctx, bool condition, const char* label) {
	if (!condition) {
		std::cerr << "FAIL(" << ctx.impl << "): " << label << "\n";
		++ctx.failures;
	}
}

// Shared helper for exception-path checks in both parity and custom-only tests.
template<typename Exception, typename Fn>
bool expect_throw(const TestContext& ctx, const char* label, Fn&& fn) {
	try {
		fn();
		std::cerr << "FAIL(" << ctx.impl << "): " << label << " did not throw\n";
		++ctx.failures;
		return false;
	} catch (const Exception&) {
		return true;
	} catch (...) {
		std::cerr << "FAIL(" << ctx.impl << "): " << label << " threw unexpected exception\n";
		++ctx.failures;
		return false;
	}
}

template<typename T>
T& identity_ref(T& value) {
	return value;
}

struct LiveCountedType {
	static int live;
	int value{0};
	LiveCountedType() { ++live; }
	explicit LiveCountedType(int v) : value(v) { ++live; }
	LiveCountedType(const LiveCountedType& other) : value(other.value) { ++live; }
	LiveCountedType(LiveCountedType&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	LiveCountedType& operator=(const LiveCountedType& other) { value = other.value; return *this; }
	LiveCountedType& operator=(LiveCountedType&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~LiveCountedType() { --live; }
};

int LiveCountedType::live = 0;

struct MoveOnly {
	static int live;
	int value;
	MoveOnly() : value(123) { ++live; }
	explicit MoveOnly(int v) : value(v) { ++live; }
	MoveOnly(const MoveOnly&) = delete;
	MoveOnly& operator=(const MoveOnly&) = delete;
	MoveOnly(MoveOnly&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	MoveOnly& operator=(MoveOnly&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~MoveOnly() { --live; }
};

int MoveOnly::live = 0;

struct ThrowingType {
	static int live;
	static int default_count;
	static int copy_count;
	static int move_count;
	static int copy_assign_count;
	static int move_assign_count;
	static int throw_on_default;
	static int throw_on_copy;
	static int throw_on_move;
	static int throw_on_copy_assign;
	static int throw_on_move_assign;

	int value{0};

	static void reset() {
		live = 0;
		default_count = copy_count = move_count = copy_assign_count = move_assign_count = 0;
		throw_on_default = throw_on_copy = throw_on_move = throw_on_copy_assign = throw_on_move_assign = -1;
	}

	static void maybe_throw(int counter, int throw_on) {
		if (throw_on >= 0 && counter == throw_on) {
			throw std::runtime_error("ThrowingType configured throw");
		}
	}

	ThrowingType() {
		++default_count;
		maybe_throw(default_count, throw_on_default);
		++live;
	}

	explicit ThrowingType(int v) : value(v) { ++live; }

	ThrowingType(const ThrowingType& other) : value(other.value) {
		++copy_count;
		maybe_throw(copy_count, throw_on_copy);
		++live;
	}

	ThrowingType(ThrowingType&& other) : value(other.value) {
		other.value = 0;
		++move_count;
		maybe_throw(move_count, throw_on_move);
		++live;
	}

	ThrowingType& operator=(const ThrowingType& other) {
		++copy_assign_count;
		maybe_throw(copy_assign_count, throw_on_copy_assign);
		value = other.value;
		return *this;
	}

	ThrowingType& operator=(ThrowingType&& other) {
		++move_assign_count;
		maybe_throw(move_assign_count, throw_on_move_assign);
		value = other.value;
		other.value = 0;
		return *this;
	}

	~ThrowingType() { --live; }
};

int ThrowingType::live = 0;
int ThrowingType::default_count = 0;
int ThrowingType::copy_count = 0;
int ThrowingType::move_count = 0;
int ThrowingType::copy_assign_count = 0;
int ThrowingType::move_assign_count = 0;
int ThrowingType::throw_on_default = -1;
int ThrowingType::throw_on_copy = -1;
int ThrowingType::throw_on_move = -1;
int ThrowingType::throw_on_copy_assign = -1;
int ThrowingType::throw_on_move_assign = -1;

struct LifetimeTrackedStats {
	int default_ctor = 0;
	int value_ctor = 0;
	int copy_ctor = 0;
	int move_ctor = 0;
	int copy_assign = 0;
	int move_assign = 0;
	int dtor = 0;
	int live = 0;
	int next_serial = 0;
};

struct LifetimeTracked {
	inline static LifetimeTrackedStats stats{};

	int value = 0;
	int serial = 0;

	static void reset() {
		stats = {};
	}

	static LifetimeTrackedStats snapshot() {
		return stats;
	}

	LifetimeTracked() : value(0), serial(++stats.next_serial) {
		++stats.default_ctor;
		++stats.live;
	}

	explicit LifetimeTracked(int v) : value(v), serial(++stats.next_serial) {
		++stats.value_ctor;
		++stats.live;
	}

	LifetimeTracked(const LifetimeTracked& other) : value(other.value), serial(++stats.next_serial) {
		++stats.copy_ctor;
		++stats.live;
	}

	LifetimeTracked(LifetimeTracked&& other) noexcept : value(other.value), serial(++stats.next_serial) {
		other.value = -1;
		++stats.move_ctor;
		++stats.live;
	}

	LifetimeTracked& operator=(const LifetimeTracked& other) {
		value = other.value;
		++stats.copy_assign;
		return *this;
	}

	LifetimeTracked& operator=(LifetimeTracked&& other) noexcept {
		value = other.value;
		other.value = -1;
		++stats.move_assign;
		return *this;
	}

	~LifetimeTracked() {
		++stats.dtor;
		--stats.live;
	}
};

struct LifetimeThrowingTracked {
	inline static int live = 0;
	inline static int default_ctor_count = 0;
	inline static int copy_ctor_count = 0;
	inline static int move_ctor_count = 0;
	inline static int copy_assign_count = 0;
	inline static int move_assign_count = 0;
	inline static int transfer_count = 0;
	inline static int dtor_count = 0;
	inline static int next_serial = 0;
	inline static int throw_on_default_ctor = -1;
	inline static int throw_on_copy_ctor = -1;
	inline static int throw_on_move_ctor = -1;
	inline static int throw_on_copy_assign = -1;
	inline static int throw_on_move_assign = -1;
	inline static int throw_on_transfer = -1;

	int value = 0;
	int serial = 0;

	static void reset() {
		live = 0;
		default_ctor_count = 0;
		copy_ctor_count = 0;
		move_ctor_count = 0;
		copy_assign_count = 0;
		move_assign_count = 0;
		transfer_count = 0;
		dtor_count = 0;
		next_serial = 0;
		throw_on_default_ctor = -1;
		throw_on_copy_ctor = -1;
		throw_on_move_ctor = -1;
		throw_on_copy_assign = -1;
		throw_on_move_assign = -1;
		throw_on_transfer = -1;
	}

	LifetimeThrowingTracked() : value(0), serial(++next_serial) {
		++default_ctor_count;
		if (throw_on_default_ctor >= 0 && default_ctor_count == throw_on_default_ctor) {
			throw std::runtime_error("LifetimeThrowingTracked default construction");
		}
		++live;
	}

	explicit LifetimeThrowingTracked(int v) : value(v), serial(++next_serial) {
		++live;
	}

	LifetimeThrowingTracked(const LifetimeThrowingTracked& other) : value(other.value), serial(++next_serial) {
		++copy_ctor_count;
		++transfer_count;
		if (throw_on_copy_ctor >= 0 && copy_ctor_count == throw_on_copy_ctor) {
			throw std::runtime_error("LifetimeThrowingTracked copy construction");
		}
		if (throw_on_transfer >= 0 && transfer_count == throw_on_transfer) {
			throw std::runtime_error("LifetimeThrowingTracked transfer");
		}
		++live;
	}

	LifetimeThrowingTracked(LifetimeThrowingTracked&& other) : value(other.value), serial(++next_serial) {
		other.value = -1;
		++move_ctor_count;
		++transfer_count;
		if (throw_on_move_ctor >= 0 && move_ctor_count == throw_on_move_ctor) {
			throw std::runtime_error("LifetimeThrowingTracked move construction");
		}
		if (throw_on_transfer >= 0 && transfer_count == throw_on_transfer) {
			throw std::runtime_error("LifetimeThrowingTracked transfer");
		}
		++live;
	}

	LifetimeThrowingTracked& operator=(const LifetimeThrowingTracked& other) {
		++copy_assign_count;
		++transfer_count;
		if (throw_on_copy_assign >= 0 && copy_assign_count == throw_on_copy_assign) {
			throw std::runtime_error("LifetimeThrowingTracked copy assignment");
		}
		if (throw_on_transfer >= 0 && transfer_count == throw_on_transfer) {
			throw std::runtime_error("LifetimeThrowingTracked transfer");
		}
		value = other.value;
		return *this;
	}

	LifetimeThrowingTracked& operator=(LifetimeThrowingTracked&& other) {
		++move_assign_count;
		++transfer_count;
		if (throw_on_move_assign >= 0 && move_assign_count == throw_on_move_assign) {
			throw std::runtime_error("LifetimeThrowingTracked move assignment");
		}
		if (throw_on_transfer >= 0 && transfer_count == throw_on_transfer) {
			throw std::runtime_error("LifetimeThrowingTracked transfer");
		}
		value = other.value;
		other.value = -1;
		return *this;
	}

	~LifetimeThrowingTracked() {
		++dtor_count;
		--live;
	}
};

template<class V>
constexpr bool has_using_sso = requires(const V& v) { v.using_sso(); };

static_assert(!has_using_sso<sw::universal::internal::sso_vector_default<int>>);

template<class T, class Allocator>
using sso_vector_auto = sw::universal::internal::sso_vector_default<T, Allocator>;

struct AllocState {
	int alloc_calls = 0;
	int dealloc_calls = 0;
	int throw_on_alloc = -1;
	std::size_t last_alloc_size = 0;
};

template<typename T>
struct CountingAllocator {
	using value_type = T;
	AllocState* state = nullptr;

	CountingAllocator() noexcept = default;
	explicit CountingAllocator(AllocState* s) noexcept : state(s) {}

	template<typename U>
	CountingAllocator(const CountingAllocator<U>& other) noexcept : state(other.state) {}

	T* allocate(std::size_t n) {
		if (state) {
			++state->alloc_calls;
			state->last_alloc_size = n;
			if (state->throw_on_alloc >= 0 && state->alloc_calls == state->throw_on_alloc) {
				throw std::bad_alloc();
			}
		}
		return std::allocator<T>{}.allocate(n);
	}

	void deallocate(T* p, std::size_t n) noexcept {
		if (state) {
			++state->dealloc_calls;
			state->last_alloc_size = n;
		}
		std::allocator<T>{}.deallocate(p, n);
	}

	template<typename U>
	bool operator==(const CountingAllocator<U>& other) const noexcept {
		return state == other.state;
	}

	template<typename U>
	bool operator!=(const CountingAllocator<U>& other) const noexcept {
		return state != other.state;
	}
};

class InputIterator {
public:
	using iterator_category = std::input_iterator_tag;
	using value_type = int;
	using difference_type = std::ptrdiff_t;
	using pointer = const int*;
	using reference = const int&;

	InputIterator() = default;
	InputIterator(const int* data, std::size_t index, std::size_t count)
		: data_(data), index_(index), count_(count) {}

	reference operator*() const { return data_[index_]; }
	InputIterator& operator++() { ++index_; return *this; }
	InputIterator operator++(int) { InputIterator tmp = *this; ++(*this); return tmp; }

	friend bool operator==(const InputIterator& lhs, const InputIterator& rhs) {
		return lhs.data_ == rhs.data_ && lhs.index_ == rhs.index_ && lhs.count_ == rhs.count_;
	}

	friend bool operator!=(const InputIterator& lhs, const InputIterator& rhs) {
		return !(lhs == rhs);
	}

private:
	const int* data_ = nullptr;
	std::size_t index_ = 0;
	std::size_t count_ = 0;
};

template<typename Vec>
void check_invariants(Vec& v, const TestContext& ctx, const char* label) {
	check(ctx, v.size() <= v.capacity(), label);
	if (v.size() > 0) {
		check(ctx, v.data() != nullptr, "data non-null for non-empty");
		if constexpr (requires(Vec& x) { &(x[0]); }) {
			for (std::size_t i = 0; i < v.size(); ++i) {
				check(ctx, &v[i] == v.data() + i, "contiguous storage");
			}
		}
	}
}

template<typename Vec>
std::vector<int> lifetime_payloads(const Vec& v) {
	std::vector<int> out;
	out.reserve(v.size());
	for (const auto& element : v) {
		out.push_back(element.value);
	}
	return out;
}

template<typename Vec>
std::vector<int> lifetime_serials(const Vec& v) {
	std::vector<int> out;
	out.reserve(v.size());
	for (const auto& element : v) {
		out.push_back(element.serial);
	}
	return out;
}

void check_values(const TestContext& ctx, const std::vector<int>& actual, std::initializer_list<int> expected, const char* label) {
	check(ctx, actual == std::vector<int>(expected), label);
}

template<typename Vec>
void check_values(const TestContext& ctx, const Vec& v, std::initializer_list<int> expected, const char* label) {
	check_values(ctx, lifetime_payloads(v), expected, label);
}

template<typename Vec>
void check_no_leak_after_scope(const TestContext& ctx, const char* label) {
	check(ctx, LifetimeTracked::stats.live == 0, label);
	check(ctx, LifetimeTracked::stats.dtor ==
		LifetimeTracked::stats.default_ctor + LifetimeTracked::stats.value_ctor + LifetimeTracked::stats.copy_ctor + LifetimeTracked::stats.move_ctor,
		"destructor count matches total LifetimeTracked constructions");
}

template<typename Vec>
void check_prefix(const TestContext& ctx, const Vec& v, std::initializer_list<int> expected, const char* label) {
	const auto values = lifetime_payloads(v);
	bool ok = values.size() >= expected.size();
	std::size_t i = 0;
	for (int expected_value : expected) {
		if (!ok || values[i] != expected_value) {
			ok = false;
			break;
		}
		++i;
	}
	check(ctx, ok, label);
}

template<typename Vec>
void check_serials_differ(const TestContext& ctx, const Vec& v, const std::vector<int>& baseline, const char* label) {
	const auto actual = lifetime_serials(v);
	bool ok = actual.size() == baseline.size();
	for (std::size_t i = 0; ok && i < actual.size(); ++i) {
		ok = (actual[i] != baseline[i]);
	}
	check(ctx, ok, label);
}

template<class T, class Alloc>
using sso_vector_small = sw::universal::internal::sso_vector<T, 4, Alloc>;

// Convert any vector-like object into plain std::vector contents for parity comparison.
template<typename Vec>
std::vector<typename Vec::value_type> materialize(const Vec& v) {
	return std::vector<typename Vec::value_type>(v.begin(), v.end());
}

// Parity tests compare observable container state and contents, not exact growth policy.
template<typename LeftVec, typename RightVec>
void check_same_vector_state(const TestContext& ctx, const LeftVec& left, const RightVec& right, const char* label) {
	check(ctx, left.empty() == right.empty(), label);
	check(ctx, left.size() == right.size(), label);
	check(ctx, left.capacity() >= left.size(), label);
	check(ctx, right.capacity() >= right.size(), label);
	check(ctx, materialize(left) == materialize(right), label);
	if (!left.empty() && !right.empty()) {
		check(ctx, left.front() == right.front(), label);
		check(ctx, left.back() == right.back(), label);
	}
}

void run_vector_std_parity_suite(int& failures) {
	TestContext ctx{"sso_vector(parity)", failures};
	using CustomVec = sw::universal::internal::sso_vector_default<int>;
	using StdVec = std::vector<int>;

	{
		// Default state parity.
		CustomVec custom;
		StdVec standard;
		check_same_vector_state(ctx, custom, standard, "default construction parity");
	}

	{
		// Constructor parity for count and count+value forms.
		CustomVec custom(4);
		StdVec standard(4);
		check_same_vector_state(ctx, custom, standard, "count construction parity");

		CustomVec custom_fill(5, 9);
		StdVec standard_fill(5, 9);
		check_same_vector_state(ctx, custom_fill, standard_fill, "count-value construction parity");
	}

	{
		// Range and initializer-list construction parity.
		const std::array<int, 5> values{1, 2, 3, 4, 5};
		CustomVec custom_range(values.begin(), values.end());
		StdVec standard_range(values.begin(), values.end());
		check_same_vector_state(ctx, custom_range, standard_range, "range construction parity");

		CustomVec custom_init{7, 8, 9};
		StdVec standard_init{7, 8, 9};
		check_same_vector_state(ctx, custom_init, standard_init, "initializer-list construction parity");
	}

	{
		// Copy and move construction should preserve the same contents/order.
		CustomVec custom;
		StdVec standard;
		for (int i = 0; i < 6; ++i) {
			custom.push_back(i);
			standard.push_back(i);
		}
		CustomVec custom_copy(custom);
		StdVec standard_copy(standard);
		check_same_vector_state(ctx, custom_copy, standard_copy, "copy construction parity");

		CustomVec custom_move(std::move(custom_copy));
		StdVec standard_move(std::move(standard_copy));
		check_same_vector_state(ctx, custom_move, standard_move, "move construction parity");
	}

	{
		// Assignment scenarios cover both fill and range replacement.
		CustomVec custom;
		StdVec standard;
		custom.assign(3, 4);
		standard.assign(3, 4);
		check_same_vector_state(ctx, custom, standard, "assign count-value parity");

		const std::array<int, 4> values{8, 6, 7, 5};
		custom.assign(values.begin(), values.end());
		standard.assign(values.begin(), values.end());
		check_same_vector_state(ctx, custom, standard, "assign range parity");
	}

	{
		// Capacity tests stay conservative: semantic parity plus capacity >= size invariants.
		CustomVec custom;
		StdVec standard;
		custom.reserve(20);
		standard.reserve(20);
		check_same_vector_state(ctx, custom, standard, "reserve empty parity");

		for (int i = 0; i < 10; ++i) {
			custom.push_back(i);
			standard.push_back(i);
		}
		custom.shrink_to_fit();
		standard.shrink_to_fit();
		check_same_vector_state(ctx, custom, standard, "shrink_to_fit parity");
	}

	{
		// Append/remove paths exercise end growth without assuming matching growth factors.
		CustomVec custom;
		StdVec standard;
		int value = 1;
		custom.push_back(value);
		standard.push_back(value);
		custom.push_back(2);
		standard.push_back(2);
		custom.emplace_back(3);
		standard.emplace_back(3);
		check_same_vector_state(ctx, custom, standard, "push_back and emplace_back parity");

		custom.pop_back();
		standard.pop_back();
		check_same_vector_state(ctx, custom, standard, "pop_back parity");
	}

	{
		// Resize parity covers both growth and shrink paths.
		CustomVec custom{1, 2, 3};
		StdVec standard{1, 2, 3};
		custom.resize(6);
		standard.resize(6);
		check_same_vector_state(ctx, custom, standard, "resize(count) parity");
		custom.resize(8, 9);
		standard.resize(8, 9);
		check_same_vector_state(ctx, custom, standard, "resize(count, value) parity");
		custom.resize(4);
		standard.resize(4);
		check_same_vector_state(ctx, custom, standard, "resize shrink parity");
	}

	{
		// Element-access parity checks observable semantics, including mutable data().
		CustomVec custom{3, 1, 4, 1, 5};
		StdVec standard{3, 1, 4, 1, 5};
		check(ctx, custom[2] == standard[2], "operator[] read parity");
		custom[2] = 9;
		standard[2] = 9;
		check_same_vector_state(ctx, custom, standard, "operator[] write parity");
		check(ctx, custom.at(2) == standard.at(2), "at() parity");
		expect_throw<std::out_of_range>(ctx, "custom at throws parity", [&]() { (void)custom.at(99); });
		expect_throw<std::out_of_range>(ctx, "std at throws parity", [&]() { (void)standard.at(99); });
		int* cp = custom.data();
		int* sp = standard.data();
		cp[0] = 8;
		sp[0] = 8;
		check_same_vector_state(ctx, custom, standard, "data() mutation parity");
	}

	{
		// Insert/emplace parity covers both end and middle operations.
		CustomVec custom{0, 1, 2, 3};
		StdVec standard{0, 1, 2, 3};
		custom.insert(custom.begin() + 2, 99);
		standard.insert(standard.begin() + 2, 99);
		check_same_vector_state(ctx, custom, standard, "insert single parity");

		custom.insert(custom.begin() + 1, 2, 7);
		standard.insert(standard.begin() + 1, 2, 7);
		check_same_vector_state(ctx, custom, standard, "insert count parity");

		const std::array<int, 3> extra{4, 5, 6};
		custom.insert(custom.begin() + 3, extra.begin(), extra.end());
		standard.insert(standard.begin() + 3, extra.begin(), extra.end());
		check_same_vector_state(ctx, custom, standard, "insert range parity");

		custom.emplace(custom.begin() + 2, 123);
		standard.emplace(standard.begin() + 2, 123);
		check_same_vector_state(ctx, custom, standard, "emplace middle parity");
	}

	{
		// Erase parity checks order preservation after one-element and range erasure.
		CustomVec custom{0, 1, 2, 3, 4, 5, 6};
		StdVec standard{0, 1, 2, 3, 4, 5, 6};
		custom.erase(custom.begin() + 2);
		standard.erase(standard.begin() + 2);
		check_same_vector_state(ctx, custom, standard, "erase one parity");

		custom.erase(custom.begin() + 1, custom.begin() + 4);
		standard.erase(standard.begin() + 1, standard.begin() + 4);
		check_same_vector_state(ctx, custom, standard, "erase range parity");
	}

	{
		// Swap parity covers both content exchange and size/capacity invariants.
		CustomVec custom_a{1, 2, 3};
		CustomVec custom_b{8, 9};
		StdVec standard_a{1, 2, 3};
		StdVec standard_b{8, 9};
		custom_a.swap(custom_b);
		standard_a.swap(standard_b);
		check_same_vector_state(ctx, custom_a, standard_a, "swap left parity");
		check_same_vector_state(ctx, custom_b, standard_b, "swap right parity");
	}

	{
		// Mixed modifier sequence acts as a small scenario test over several public APIs.
		CustomVec custom;
		StdVec standard;
		for (int i = 0; i < 5; ++i) {
			custom.push_back(i);
			standard.push_back(i);
		}
		custom.insert(custom.begin() + 1, 42);
		standard.insert(standard.begin() + 1, 42);
		custom.erase(custom.begin() + 3);
		standard.erase(standard.begin() + 3);
		custom.resize(8, 7);
		standard.resize(8, 7);
		custom.pop_back();
		standard.pop_back();
		custom.clear();
		standard.clear();
		check_same_vector_state(ctx, custom, standard, "mixed modifier sequence parity");
	}
}

void run_sso_proxy_suite(int& failures) {
	using Vec = sw::universal::internal::sso_vector_default<int>;
	TestContext ctx{"sso_vector(proxy)", failures};

	// Custom-only behavior: non-const indexing/iteration uses proxy objects instead of raw references.
	Vec v;
	v.push_back(11);
	v.push_back(22);
	static_assert(!std::is_reference_v<decltype(v[0])>);
	static_assert(!std::is_reference_v<decltype(*v.begin())>);
	check(ctx, static_cast<int>(v[0]) == 11, "proxy read conversion");
	v[1] = 42;
	check(ctx, static_cast<int>(v[1]) == 42, "proxy write assignment");
}

void run_sso_cow_suite(int& failures) {
	using Vec = sw::universal::internal::sso_vector_default<int>;
	TestContext ctx{"sso_vector(cow)", failures};

	{
		// Copy-on-write: copying a shareable heap block should share until a write detaches it.
		Vec a;
		a.reserve(64);
		for (int i = 0; i < 12; ++i) a.push_back(i);
		const Vec& ca = a;
		const int* pa = ca.data();

		Vec b = a;
		const Vec& cb = b;
		check(ctx, cb.data() == pa, "copy shares heap when shareable");
		b[0] = 77; // proxy write detach
		check(ctx, static_cast<int>(a[0]) == 0, "proxy write does not mutate source");
		check(ctx, static_cast<int>(b[0]) == 77, "proxy write applies to destination");
		check(ctx, cb.data() != pa, "proxy write detaches from shared heap");
	}

	{
		// Mutable data() explicitly clears shareability so future copies deep-copy instead of sharing.
		Vec v;
		v.reserve(48);
		for (int i = 0; i < 10; ++i) v.push_back(i);
		int* p = v.data(); // pins and ensures unique
		p[2] = 1234;
		Vec copy = v; // pinned source must deep-copy
		const Vec& cv = v;
		const Vec& cc = copy;
		check(ctx, cv.data() != cc.data(), "copy of pinned source deep-copies");
		check(ctx, static_cast<int>(copy[2]) == 1234, "deep copy preserves payload");
		check(ctx, copy.capacity() == v.capacity(), "copied heap preserves capacity");
	}

	{
		// Mutating operations on a shared heap block should detach before modifying observable contents.
		Vec base;
		base.reserve(96);
		for (int i = 0; i < 16; ++i) base.push_back(i);
		const Vec& cbase = base;
		const int* pbase = cbase.data();
		Vec other = base;
		other.push_back(99); // mutating op should detach
		const Vec& cother = other;
		check(ctx, cother.data() != pbase, "push_back detaches shared heap");
		check(ctx, base.size() == 16, "source size unchanged after detached mutation");
		check(ctx, other.size() == 17, "destination size reflects mutation");
	}

	{
		// Concurrency smoke test: repeated share-only copies should leave the source vector unchanged.
		Vec shared;
		shared.reserve(64);
		for (int i = 0; i < 8; ++i) shared.push_back(i * 3);

		std::thread t1([&]() {
			for (int i = 0; i < 200; ++i) {
				Vec local = shared;
				volatile int sink = static_cast<int>(local[0]);
				(void)sink;
			}
		});
		std::thread t2([&]() {
			for (int i = 0; i < 200; ++i) {
				Vec local = shared;
				volatile int sink = static_cast<int>(local[1]);
				(void)sink;
			}
		});
		t1.join();
		t2.join();
		check(ctx, shared.size() == 8, "concurrency smoke preserves source state");
	}
}

} // namespace

template<template<class, class> class Vec, typename T>
using VecDefaultAlloc = Vec<T, std::allocator<T>>;

template<template<class, class> class Vec>
void run_vector_suite(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};
	using namespace sw::universal::internal;

	{
		VecDefaultAlloc<Vec, int> v;
		check(ctx, v.empty(), "default empty");
		check(ctx, v.size() == 0, "default size 0");
		check(ctx, v.capacity() >= v.size(), "capacity >= size");

		v.push_back(1);
		v.push_back(2);
		v.push_back(3);
		check(ctx, v.size() == 3, "push_back increases size");
		check(ctx, v.front() == 1, "front value");
		check(ctx, v.back() == 3, "back value");
		check_invariants(v, ctx, "basic invariants");
	}

	{
		VecDefaultAlloc<Vec, int> v;
		v.reserve(1);
		std::size_t prev_capacity = v.capacity();
		for (int i = 0; i < 32; ++i) {
			v.emplace_back(i);
			check(ctx, v.size() == static_cast<std::size_t>(i + 1), "emplace_back size");
			check(ctx, v.capacity() >= v.size(), "capacity >= size after emplace_back");
			if (v.capacity() != prev_capacity) {
				prev_capacity = v.capacity();
			}
		}
		check(ctx, v[0] == 0 && v[31] == 31, "emplace_back data");
		check_invariants(v, ctx, "emplace_back invariants");
	}

	{
		VecDefaultAlloc<Vec, int> v;
		for (int i = 0; i < 5; ++i) v.push_back(i);
		v.insert(v.begin() + 2, 42);
		check(ctx, v.size() == 6, "insert single size");
		check(ctx, v[2] == 42, "insert single value");

		v.insert(v.begin() + 1, 3, 7);
		check(ctx, v.size() == 9, "insert count size");
		check(ctx, v[1] == 7 && v[3] == 7, "insert count values");

		std::vector<int> src = {9, 8, 7};
		v.insert(v.begin() + 4, src.begin(), src.end());
		check(ctx, v.size() == 12, "insert forward range size");
		check(ctx, v[4] == 9 && v[6] == 7, "insert forward range values");

		int input_data[] = {4, 3, 2};
		InputIterator first(input_data, 0, 3);
		InputIterator last(input_data, 3, 3);
		v.insert(v.begin() + 2, first, last);
		check(ctx, v.size() == 15, "insert input range size");
		check(ctx, v[2] == 4 && v[4] == 2, "insert input range values");
		check_invariants(v, ctx, "insert invariants");
	}

	{
		VecDefaultAlloc<Vec, int> v;
		for (int i = 0; i < 10; ++i) v.push_back(i);
		v.erase(v.begin() + 3);
		check(ctx, v.size() == 9, "erase single size");
		check(ctx, v[3] == 4, "erase single order");

		v.erase(v.begin() + 2, v.begin() + 5);
		check(ctx, v.size() == 6, "erase range size");
		check(ctx, v[2] == 6, "erase range order");
		check_invariants(v, ctx, "erase invariants");
	}

	{
		VecDefaultAlloc<Vec, int> v;
		for (int i = 0; i < 10; ++i) v.push_back(i);
		const auto old_capacity = v.capacity();
		v.reserve(old_capacity + 10);
		check(ctx, v.capacity() >= old_capacity + 10, "reserve grows capacity");
		const auto reserve_capacity = v.capacity();
		v.shrink_to_fit();
		check(ctx, v.size() == 10, "shrink_to_fit keeps size");
		check(ctx, v.capacity() >= v.size(), "shrink_to_fit capacity >= size");
		check(ctx, v.capacity() <= reserve_capacity, "shrink_to_fit does not grow capacity");
		check_invariants(v, ctx, "shrink invariants");
	}

	{
		MoveOnly::live = 0;
		VecDefaultAlloc<Vec, MoveOnly> v;
		v.resize(3);
		check(ctx, v.size() == 3, "resize default size");
		check(ctx, v.data()[0].value == 123 && v.data()[1].value == 123 && v.data()[2].value == 123, "resize default values");
		check(ctx, MoveOnly::live == 3, "resize default live count");
		v.resize(1);
		check(ctx, v.size() == 1, "resize shrink size");
		check(ctx, MoveOnly::live == 1, "resize shrink live count");
		v.clear();
		check(ctx, MoveOnly::live == 0, "resize clear live count");
	}

	{
		VecDefaultAlloc<Vec, std::string> vs({"a", "b", "c"});
		VecDefaultAlloc<Vec, std::string> vs_copy(vs);
		check(ctx, vs_copy == vs, "copy equality");

		VecDefaultAlloc<Vec, std::string> vs_move(std::move(vs_copy));
		check(ctx, vs_move.size() == 3, "move size");
		check(ctx, vs_move[2] == "c", "move data");

		// Exercise copy-assignment on the same object to verify self-assignment is a no-op.
		vs_move = identity_ref(vs_move);
		check(ctx, vs_move.size() == 3, "self copy assignment");

		VecDefaultAlloc<Vec, std::string> vs_assign;
		vs_assign = vs_move;
		check(ctx, vs_assign == vs_move, "copy assignment");

		VecDefaultAlloc<Vec, std::string> vs_move_assign;
		vs_move_assign = std::move(vs_assign);
		check(ctx, vs_move_assign.size() == 3, "move assignment");
		check_invariants(vs_move_assign, ctx, "copy/move invariants");
	}

	{
		VecDefaultAlloc<Vec, int> a;
		VecDefaultAlloc<Vec, int> b;
		for (int i = 0; i < 5; ++i) a.push_back(i);
		for (int i = 0; i < 2; ++i) b.push_back(10 + i);
		a.reserve(32);
		b.reserve(4);
		a.swap(b);
		check(ctx, a.size() == 2 && b.size() == 5, "swap sizes");
		check(ctx, a[0] == 10 && b[0] == 0, "swap contents");
		check_invariants(a, ctx, "swap invariants a");
		check_invariants(b, ctx, "swap invariants b");
	}

	{
		ThrowingType::reset();
		VecDefaultAlloc<Vec, ThrowingType> v;
		v.reserve(1);
		v.emplace_back(1);
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(ctx, "push_back reallocation throws", [&]() {
			v.push_back(ThrowingType(2));
		});
		check(ctx, v.size() == 1, "size unchanged after throw");
		check(ctx, v.data()[0].value == 1, "value preserved after throw");
		check(ctx, ThrowingType::live == 1, "no leak after throw");
		v.clear();
		check(ctx, ThrowingType::live == 0, "clear destroys elements");
	}

	{
		ThrowingType::reset();
		VecDefaultAlloc<Vec, ThrowingType> v;
		for (int i = 0; i < 3; ++i) v.emplace_back(i);
		v.shrink_to_fit(); // normalize to size==capacity so next insert must grow
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(ctx, "insert reallocation throws", [&]() {
			v.insert(v.begin() + 1, ThrowingType(9));
		});
		check(ctx, ThrowingType::live == static_cast<int>(v.size()), "insert throw live count");
		check_invariants(v, ctx, "insert throw invariants");
		v.clear();
		check(ctx, ThrowingType::live == 0, "clear after insert throw");
	}

	{
		ThrowingType::reset();
		VecDefaultAlloc<Vec, ThrowingType> v;
		for (int i = 0; i < 3; ++i) v.emplace_back(i);
		ThrowingType::throw_on_default = 2;
		expect_throw<std::runtime_error>(ctx, "resize growth throws", [&]() {
			v.resize(6);
		});
		check(ctx, v.size() == 3, "resize throw size unchanged");
		check(ctx, v.data()[0].value == 0 && v.data()[1].value == 1 && v.data()[2].value == 2, "resize throw values preserved");
		check(ctx, ThrowingType::live == static_cast<int>(v.size()), "resize throw live count");
		v.clear();
		check(ctx, ThrowingType::live == 0, "resize throw clear live count");
	}

	{
		AllocState state;
		using AllocVec = Vec<int, CountingAllocator<int>>;
		AllocVec v{CountingAllocator<int>(&state)};
		state.throw_on_alloc = 1;
		expect_throw<std::bad_alloc>(ctx, "allocator throws on reserve", [&]() {
			v.reserve(5);
		});
		check(ctx, v.size() == 0, "allocator throw size unchanged");
		check_invariants(v, ctx, "allocator throw invariants");
		check(ctx, state.alloc_calls == 1, "allocator alloc count");
	}

	{
		AllocState state;
		using AllocVec = Vec<LiveCountedType, CountingAllocator<LiveCountedType>>;
		AllocVec v{CountingAllocator<LiveCountedType>(&state)};
		v.push_back(LiveCountedType(1));
		while (v.size() < v.capacity()) {
			v.push_back(LiveCountedType(static_cast<int>(v.size() + 1)));
		}
		const auto live_before_throw = LiveCountedType::live;
		state.throw_on_alloc = state.alloc_calls + 1;
		expect_throw<std::bad_alloc>(ctx, "allocator throws on growth", [&]() {
			v.push_back(LiveCountedType(2));
		});
		check(ctx, LiveCountedType::live == live_before_throw, "allocator throw preserves live count");
		v.clear();
		check(ctx, LiveCountedType::live == 0, "allocator throw clear live count");
	}
}

template<template<class, class> class VecTemplate>
void run_vector_lifetime_suite(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};

	using TrackedVec = VecTemplate<LifetimeTracked, std::allocator<LifetimeTracked>>;
	using ThrowingVec = VecTemplate<LifetimeThrowingTracked, std::allocator<LifetimeThrowingTracked>>;

	{
		LifetimeTracked::reset();
		LifetimeTrackedStats before = LifetimeTracked::snapshot();
		{
			TrackedVec v;
			v.reserve(8);
			LifetimeTrackedStats after_reserve = LifetimeTracked::snapshot();
			check(ctx, v.empty(), "reserve baseline empty");
			check(ctx, v.capacity() >= 8, "reserve baseline capacity");
			check(ctx, after_reserve.default_ctor == before.default_ctor, "reserve does not default-construct");
			check(ctx, after_reserve.value_ctor == before.value_ctor, "reserve does not value-construct");
			check(ctx, after_reserve.copy_ctor == before.copy_ctor, "reserve does not copy-construct");
			check(ctx, after_reserve.move_ctor == before.move_ctor, "reserve does not move-construct");
			check(ctx, after_reserve.dtor == before.dtor, "reserve does not destroy");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "reserved empty container destroys nothing");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec v;
			v.reserve(4);
			v.emplace_back(1);
			v.emplace_back(2);
			v.emplace_back(3);
			check_values(ctx, v, {1, 2, 3}, "append contents");
			check(ctx, v.size() == 3, "append size");
			check(ctx, LifetimeTracked::stats.live == 3, "append live count");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "append scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec v;
			v.reserve(5);
			v.emplace_back(10);
			v.emplace_back(20);
			const LifetimeTrackedStats before_grow = LifetimeTracked::snapshot();
			v.resize(5);
			const LifetimeTrackedStats after_grow = LifetimeTracked::snapshot();
			check(ctx, after_grow.default_ctor - before_grow.default_ctor == 3, "resize grow default-constructs new elements");
			check_values(ctx, v, {10, 20, 0, 0, 0}, "resize grow values");
			const int dtor_before_shrink = after_grow.dtor;
			v.resize(2);
			check_values(ctx, v, {10, 20}, "resize shrink values");
			check(ctx, LifetimeTracked::stats.dtor - dtor_before_shrink == 3, "resize shrink destroys removed tail");
			check(ctx, LifetimeTracked::stats.live == 2, "resize shrink live count");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "resize scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec v;
			v.reserve(8);
			v.emplace_back(1);
			v.emplace_back(2);
			v.emplace_back(3);
			v.insert(v.begin() + 1, LifetimeTracked(99));
			check_values(ctx, v, {1, 99, 2, 3}, "middle insert values");
			check(ctx, LifetimeTracked::stats.live == static_cast<int>(v.size()), "insert live count");
			v.erase(v.begin() + 2);
			check_values(ctx, v, {1, 99, 3}, "middle erase values");
			check(ctx, LifetimeTracked::stats.live == static_cast<int>(v.size()), "erase live count");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "insert erase scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec source;
			source.reserve(3);
			source.emplace_back(4);
			source.emplace_back(5);
			source.emplace_back(6);
			const LifetimeTrackedStats before_copy = LifetimeTracked::snapshot();
			TrackedVec copied(source);
			check_values(ctx, copied, {4, 5, 6}, "copy construction values");
			check(ctx, LifetimeTracked::stats.copy_ctor - before_copy.copy_ctor == 3, "copy construction copies each element");

			TrackedVec assigned;
			assigned.reserve(4);
			assigned.emplace_back(7);
			assigned.emplace_back(8);
			assigned = source;
			check_values(ctx, assigned, {4, 5, 6}, "copy assignment values");

			TrackedVec moved(std::move(copied));
			check_values(ctx, moved, {4, 5, 6}, "move construction values");

			TrackedVec move_assigned;
			move_assigned = std::move(assigned);
			check_values(ctx, move_assigned, {4, 5, 6}, "move assignment values");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "copy move scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec v;
			v.reserve(3);
			v.emplace_back(1);
			v.emplace_back(2);
			v.emplace_back(3);
			LifetimeTracked rhs(42);
			const LifetimeTrackedStats before_assign = LifetimeTracked::snapshot();
			v[1] = std::move(rhs);
			const LifetimeTrackedStats after_assign = LifetimeTracked::snapshot();
			check_values(ctx, v, {1, 42, 3}, "operator[] assignment values");
			check(ctx, after_assign.move_assign - before_assign.move_assign == 1, "operator[] uses move assignment");
			check(ctx, after_assign.copy_assign == before_assign.copy_assign, "operator[] does not copy-assign");
			check(ctx, after_assign.copy_ctor == before_assign.copy_ctor, "operator[] does not reconstruct by copy");
			check(ctx, after_assign.move_ctor == before_assign.move_ctor, "operator[] does not reconstruct by move");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "operator[] assignment scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			TrackedVec v;
			v.reserve(4);
			v.emplace_back(1);
			v.emplace_back(2);
			v.emplace_back(3);
			const int dtor_before_pop = LifetimeTracked::stats.dtor;
			v.pop_back();
			check_values(ctx, v, {1, 2}, "pop_back values");
			check(ctx, LifetimeTracked::stats.dtor - dtor_before_pop == 1, "pop_back destroys one element");
			const int dtor_before_clear = LifetimeTracked::stats.dtor;
			v.clear();
			check(ctx, v.empty(), "clear empties container");
			check(ctx, LifetimeTracked::stats.dtor - dtor_before_clear == 2, "clear destroys active elements");
			check(ctx, LifetimeTracked::stats.live == 0, "clear live count");
			v.emplace_back(9);
			check_values(ctx, v, {9}, "append after clear");
		}
		check_no_leak_after_scope<TrackedVec>(ctx, "clear reuse scope destruction");
	}

	{
		LifetimeThrowingTracked::reset();
		{
			ThrowingVec v;
			v.reserve(2);
			v.emplace_back(1);
			v.emplace_back(2);
			const auto before = lifetime_payloads(v);
			LifetimeThrowingTracked::throw_on_copy_ctor = 1;
			LifetimeThrowingTracked::throw_on_move_ctor = 1;
			expect_throw<std::runtime_error>(ctx, "insert growth throws", [&]() {
				v.insert(v.begin() + 1, LifetimeThrowingTracked(99));
			});
			check_values(ctx, v, {1, 2}, "insert throw preserves values");
			check(ctx, v.size() == before.size(), "insert throw preserves size");
			check(ctx, LifetimeThrowingTracked::live == static_cast<int>(v.size()), "insert throw no leaked elements");
		}
		check(ctx, LifetimeThrowingTracked::live == 0, "insert throw scope destruction");
	}

	{
		for (int throw_at : {1, 2, 3}) {
			LifetimeThrowingTracked::reset();
			{
				ThrowingVec v;
				v.reserve(8);
				v.emplace_back(10);
				v.emplace_back(20);
				LifetimeThrowingTracked::throw_on_default_ctor = throw_at;
				std::string label = "resize grow throws at step " + std::to_string(throw_at);
				expect_throw<std::runtime_error>(ctx, label.c_str(), [&]() {
					v.resize(5);
				});
				LifetimeThrowingTracked::throw_on_default_ctor = -1;
				check(ctx, v.size() >= 2 && v.size() <= 4, "resize throw leaves valid size");
				check_prefix(ctx, v, {10, 20}, "resize throw preserves prefix");
				for (std::size_t i = 2; i < v.size(); ++i) {
					check(ctx, lifetime_payloads(v)[i] == 0, "resize throw keeps constructed tail default-initialized");
				}
				check(ctx, LifetimeThrowingTracked::live == static_cast<int>(v.size()), "resize throw no leaked elements");
				v.clear();
				v.emplace_back(77);
				check_values(ctx, v, {77}, "resize throw leaves container reusable");
			}
			check(ctx, LifetimeThrowingTracked::live == 0, "resize throw scope destruction");
		}
	}

	{
		for (int throw_at : {1, 2, 3}) {
			LifetimeThrowingTracked::reset();
			{
				ThrowingVec v;
				v.reserve(8);
				v.emplace_back(1);
				v.emplace_back(2);
				std::vector<LifetimeThrowingTracked> src;
				src.reserve(3);
				src.emplace_back(30);
				src.emplace_back(40);
				src.emplace_back(50);
				LifetimeThrowingTracked::copy_ctor_count = 0;
				LifetimeThrowingTracked::throw_on_copy_ctor = throw_at;
				std::string label = "append sequence throws at step " + std::to_string(throw_at);
				expect_throw<std::runtime_error>(ctx, label.c_str(), [&]() {
					for (const auto& element : src) {
						v.push_back(element);
					}
				});
				LifetimeThrowingTracked::throw_on_copy_ctor = -1;
				check(ctx, v.size() == static_cast<std::size_t>(2 + throw_at - 1), "append throw leaves prefix of successful appends");
				if (throw_at == 1) check_values(ctx, v, {1, 2}, "append throw values at first failure");
				if (throw_at == 2) check_values(ctx, v, {1, 2, 30}, "append throw values at second failure");
				if (throw_at == 3) check_values(ctx, v, {1, 2, 30, 40}, "append throw values at third failure");
				check(ctx, LifetimeThrowingTracked::live == static_cast<int>(v.size() + src.size()), "append throw no leaked elements");
				v.clear();
				v.emplace_back(88);
				check_values(ctx, v, {88}, "append throw leaves container reusable");
			}
			check(ctx, LifetimeThrowingTracked::live == 0, "append throw scope destruction");
		}
	}
}

void run_sso_vector_specific_lifetime_suite(int& failures) {
	using Vec = sso_vector_small<LifetimeTracked, std::allocator<LifetimeTracked>>;
	using ThrowVec = sso_vector_small<LifetimeThrowingTracked, std::allocator<LifetimeThrowingTracked>>;
	TestContext ctx{"sso_vector_lifetime_specific", failures};

	{
		LifetimeTracked::reset();
		LifetimeTrackedStats before = LifetimeTracked::snapshot();
		{
			Vec v;
			v.reserve(16);
			LifetimeTrackedStats after = LifetimeTracked::snapshot();
			check(ctx, after.live == 0, "reserve on empty sso_vector creates no live elements");
			check(ctx, after.default_ctor == before.default_ctor, "reserve on empty does not default-construct");
			check(ctx, after.value_ctor == before.value_ctor, "reserve on empty does not value-construct");
			check(ctx, after.copy_ctor == before.copy_ctor, "reserve on empty does not copy-construct");
			check(ctx, after.move_ctor == before.move_ctor, "reserve on empty does not move-construct");
			check(ctx, after.dtor == before.dtor, "reserve on empty does not destroy");
		}
		check_no_leak_after_scope<Vec>(ctx, "reserve on empty scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			Vec a;
			a.reserve(16);
			for (int i = 0; i < 6; ++i) a.emplace_back(i);
			const Vec& ca = a;
			LifetimeTrackedStats after_share{};
			{
				Vec b = a;
				const Vec& cb = b;
				check(ctx, ca.data() == cb.data(), "heap-backed copy shares storage before teardown");
				after_share = LifetimeTracked::snapshot();
			}
			const LifetimeTrackedStats after_release = LifetimeTracked::snapshot();
			check_values(ctx, a, {0, 1, 2, 3, 4, 5}, "destroying one sharer preserves survivor contents");
			check(ctx, after_release.copy_ctor == after_share.copy_ctor, "teardown does not copy-construct");
			check(ctx, after_release.move_ctor == after_share.move_ctor, "teardown does not move-construct");
			check(ctx, after_release.value_ctor == after_share.value_ctor, "teardown does not value-construct");
			check(ctx, after_release.default_ctor == after_share.default_ctor, "teardown does not default-construct");
			check(ctx, after_release.dtor == after_share.dtor, "teardown of one sharer does not destroy shared elements");
		}
		check_no_leak_after_scope<Vec>(ctx, "shared teardown leaves no leaks");
	}

	{
		LifetimeTracked::reset();
		{
			Vec a;
			a.reserve(16);
			for (int i = 0; i < 6; ++i) a.emplace_back(i);
			Vec b = a;
			const Vec& cb = b;
			const LifetimeTracked* shared_before = cb.data();
			LifetimeTrackedStats before_const_data = LifetimeTracked::snapshot();
			const Vec& ca = a;
			(void)ca.data();
			(void)cb.data();
			LifetimeTrackedStats after_const_data = LifetimeTracked::snapshot();
			check(ctx, ca.data() == shared_before, "const data preserves shared pointer");
			check(ctx, cb.data() == shared_before, "const data on sibling preserves shared pointer");
			check(ctx, after_const_data.copy_ctor == before_const_data.copy_ctor, "const data does not copy-construct");
			check(ctx, after_const_data.move_ctor == before_const_data.move_ctor, "const data does not move-construct");
			check(ctx, after_const_data.dtor == before_const_data.dtor, "const data does not destroy");
		}
		check_no_leak_after_scope<Vec>(ctx, "const data scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			Vec a;
			a.reserve(16);
			for (int i = 0; i < 6; ++i) a.emplace_back(i);
			Vec b = a;
			const Vec& cb = b;
			const LifetimeTracked* b_before = cb.data();
			LifetimeTracked* raw = a.data();
			raw[0].value = 777;
			check_values(ctx, a, {777, 1, 2, 3, 4, 5}, "mutable data writes through detached storage");
			check_values(ctx, b, {0, 1, 2, 3, 4, 5}, "mutable data does not mutate sibling sharer");
			check(ctx, cb.data() == b_before, "mutable data on sibling leaves original block in place");

			Vec c = b;
			const Vec& cc = c;
			check(ctx, cc.data() == cb.data(), "untouched sharer remains shareable after sibling mutable data");
			check_values(ctx, c, {0, 1, 2, 3, 4, 5}, "copy after sibling mutable data preserves untouched values");
		}
		check_no_leak_after_scope<Vec>(ctx, "mutable data scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			Vec a;
			a.reserve(16);
			for (int i = 0; i < 6; ++i) a.emplace_back(i);
			Vec b = a;
			const LifetimeTrackedStats before_write = LifetimeTracked::snapshot();
			b[2] = LifetimeTracked(999);
			const LifetimeTrackedStats after_write = LifetimeTracked::snapshot();
			check_values(ctx, a, {0, 1, 2, 3, 4, 5}, "proxy write preserves source contents");
			check_values(ctx, b, {0, 1, 999, 3, 4, 5}, "proxy write updates destination contents");
			check(ctx, after_write.copy_ctor - before_write.copy_ctor == 6, "proxy write detaches by copying shared elements once");
			check(ctx, after_write.move_assign - before_write.move_assign == 1, "proxy write performs one element move-assignment");
		}
		check_no_leak_after_scope<Vec>(ctx, "proxy write scope destruction");
	}

	{
		LifetimeThrowingTracked::reset();
		{
			ThrowVec src;
			src.reserve(16);
			src.emplace_back(10);
			src.emplace_back(20);
			src.emplace_back(30);
			(void)src.data();
			ThrowVec dst;
			dst.emplace_back(1);
			dst.emplace_back(2);
			const auto src_before = lifetime_payloads(src);
			LifetimeThrowingTracked::copy_ctor_count = 0;
			LifetimeThrowingTracked::throw_on_copy_ctor = 1;
			expect_throw<std::runtime_error>(ctx, "sso copy assignment throws", [&]() {
				dst = src;
			});
			LifetimeThrowingTracked::throw_on_copy_ctor = -1;
			check_values(ctx, src, {10, 20, 30}, "copy assignment throw preserves source contents");
			check(ctx, LifetimeThrowingTracked::live == static_cast<int>(src.size() + dst.size()), "copy assignment throw leaves no leaked live objects");
			check(ctx, lifetime_payloads(src) == src_before, "copy assignment throw does not mutate source");
		}
		check(ctx, LifetimeThrowingTracked::live == 0, "copy assignment throw scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			Vec source;
			source.emplace_back(1);
			source.emplace_back(2);
			source.emplace_back(3);
			const auto source_serials = lifetime_serials(source);

			const LifetimeTrackedStats before_copy = LifetimeTracked::snapshot();
			Vec copied(source);
			check_values(ctx, copied, {1, 2, 3}, "inline copy construction preserves values");
			check(ctx, LifetimeTracked::stats.copy_ctor - before_copy.copy_ctor == 3, "inline copy construction copies each element");
			check_serials_differ(ctx, copied, source_serials, "inline copy construction creates distinct element objects");

			const LifetimeTrackedStats before_move = LifetimeTracked::snapshot();
			Vec moved(std::move(source));
			check_values(ctx, moved, {1, 2, 3}, "inline move construction preserves values");
			check(ctx, (LifetimeTracked::stats.move_ctor - before_move.move_ctor) + (LifetimeTracked::stats.copy_ctor - before_move.copy_ctor) == 3,
				"inline move construction constructs each destination element");
			check_serials_differ(ctx, moved, source_serials, "inline move construction creates distinct destination objects");
		}
		check_no_leak_after_scope<Vec>(ctx, "inline copy/move construction scope destruction");
	}

	{
		LifetimeTracked::reset();
		{
			Vec source;
			source.emplace_back(4);
			source.emplace_back(5);
			source.emplace_back(6);
			const auto source_serials = lifetime_serials(source);

			Vec copy_assigned;
			const LifetimeTrackedStats before_copy_assign = LifetimeTracked::snapshot();
			copy_assigned = source;
			check_values(ctx, copy_assigned, {4, 5, 6}, "inline copy assignment preserves values");
			check(ctx, LifetimeTracked::stats.copy_ctor - before_copy_assign.copy_ctor == 3, "inline copy assignment constructs copied elements");
			check_serials_differ(ctx, copy_assigned, source_serials, "inline copy assignment creates distinct destination objects");

			Vec move_source;
			move_source.emplace_back(7);
			move_source.emplace_back(8);
			move_source.emplace_back(9);
			const auto move_source_serials = lifetime_serials(move_source);
			Vec move_assigned;
			const LifetimeTrackedStats before_move_assign = LifetimeTracked::snapshot();
			move_assigned = std::move(move_source);
			check_values(ctx, move_assigned, {7, 8, 9}, "inline move assignment preserves values");
			check(ctx, (LifetimeTracked::stats.move_ctor - before_move_assign.move_ctor) + (LifetimeTracked::stats.copy_ctor - before_move_assign.copy_ctor) == 3,
				"inline move assignment constructs each destination element");
			check_serials_differ(ctx, move_assigned, move_source_serials, "inline move assignment creates distinct destination objects");

			const int dtor_before_erase = LifetimeTracked::stats.dtor;
			move_assigned.insert(move_assigned.begin() + 1, LifetimeTracked(11));
			check_values(ctx, move_assigned, {7, 11, 8, 9}, "inline insert preserves order");
			move_assigned.erase(move_assigned.begin() + 2);
			check_values(ctx, move_assigned, {7, 11, 9}, "inline erase preserves order");
			check(ctx, LifetimeTracked::stats.dtor - dtor_before_erase >= 1, "inline erase destroys removed element");

			const LifetimeTrackedStats before_shrink = LifetimeTracked::snapshot();
			move_assigned.shrink_to_fit();
			const LifetimeTrackedStats after_shrink = LifetimeTracked::snapshot();
			check(ctx, after_shrink.copy_ctor == before_shrink.copy_ctor, "inline shrink_to_fit does not copy inline elements");
			check(ctx, after_shrink.move_ctor == before_shrink.move_ctor, "inline shrink_to_fit does not move inline elements");

			Vec left;
			left.emplace_back(1);
			left.emplace_back(2);
			Vec right;
			right.emplace_back(9);
			right.emplace_back(8);
			left.swap(right);
			check_values(ctx, left, {9, 8}, "inline swap preserves right values");
			check_values(ctx, right, {1, 2}, "inline swap preserves left values");
		}
		check_no_leak_after_scope<Vec>(ctx, "inline assignment and modifier scope destruction");
	}
}

int main() {
	int nrOfFailedTestCases = 0;
	run_sso_proxy_suite(nrOfFailedTestCases);
	run_sso_cow_suite(nrOfFailedTestCases);
	run_vector_std_parity_suite(nrOfFailedTestCases);
	run_vector_suite<sso_vector_auto>("sso_vector", nrOfFailedTestCases);
	run_vector_suite<std::vector>("std::vector", nrOfFailedTestCases);
	run_vector_lifetime_suite<std::vector>("std::vector_lifetime", nrOfFailedTestCases);
	run_vector_lifetime_suite<sso_vector_small>("sso_vector_lifetime", nrOfFailedTestCases);
	run_sso_vector_specific_lifetime_suite(nrOfFailedTestCases);

	sw::universal::ReportTestResult(nrOfFailedTestCases, "sso_vector", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
