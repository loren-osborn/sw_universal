// sso_vector.cpp: unit tests for sso_vector
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/internal/sso_vector/sso_vector.hpp>
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

template<class V>
constexpr bool has_using_sso = requires(const V& v) { v.using_sso(); };

static_assert(!has_using_sso<sw::universal::internal::sso_vector<int>>);

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
		for (std::size_t i = 0; i < v.size(); ++i) {
			check(ctx, &v[i] == v.data() + i, "contiguous storage");
		}
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
		check(ctx, v[0].value == 123 && v[1].value == 123 && v[2].value == 123, "resize default values");
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

		vs_move = vs_move;
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
		check(ctx, v[0].value == 1, "value preserved after throw");
		check(ctx, ThrowingType::live == 1, "no leak after throw");
		v.clear();
		check(ctx, ThrowingType::live == 0, "clear destroys elements");
	}

	{
		ThrowingType::reset();
		VecDefaultAlloc<Vec, ThrowingType> v;
		for (int i = 0; i < 3; ++i) v.emplace_back(i);
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
		check(ctx, v[0].value == 0 && v[1].value == 1 && v[2].value == 2, "resize throw values preserved");
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
		state.throw_on_alloc = 2;
		expect_throw<std::bad_alloc>(ctx, "allocator throws on growth", [&]() {
			v.push_back(LiveCountedType(2));
		});
		check(ctx, LiveCountedType::live == 1, "allocator throw preserves live count");
		v.clear();
		check(ctx, LiveCountedType::live == 0, "allocator throw clear live count");
	}
}

int main() {
	int nrOfFailedTestCases = 0;
	run_vector_suite<sw::universal::internal::sso_vector>("sso_vector", nrOfFailedTestCases);
	run_vector_suite<std::vector>("std::vector", nrOfFailedTestCases);

	sw::universal::ReportTestResult(nrOfFailedTestCases, "sso_vector", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
