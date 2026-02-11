// sso_vector.cpp: unit tests for sso_vector
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/internal/sso_vector/sso_vector.hpp>
#include <universal/verification/test_status.hpp>

namespace {

void check(bool condition, int& failures, const char* label) {
	if (!condition) {
		std::cerr << "FAIL: " << label << "\n";
		++failures;
	}
}

template<typename Exception, typename Fn>
bool expect_throw(int& failures, const char* label, Fn&& fn) {
	try {
		fn();
		std::cerr << "FAIL: " << label << " did not throw\n";
		++failures;
		return false;
	} catch (const Exception&) {
		return true;
	} catch (...) {
		std::cerr << "FAIL: " << label << " threw unexpected exception\n";
		++failures;
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
void check_invariants(Vec& v, int& failures, const char* label) {
	check(v.size() <= v.capacity(), failures, label);
	if (v.size() > 0) {
		check(v.data() != nullptr, failures, "data non-null for non-empty");
		for (std::size_t i = 0; i < v.size(); ++i) {
			check(&v[i] == v.data() + i, failures, "contiguous storage");
		}
	}
}

} // namespace

int main() {
	using namespace sw::universal::internal;
	int nrOfFailedTestCases = 0;

	{
		sso_vector<int> v;
		check(v.empty(), nrOfFailedTestCases, "default empty");
		check(v.size() == 0, nrOfFailedTestCases, "default size 0");
		check(v.capacity() >= v.size(), nrOfFailedTestCases, "capacity >= size");

		v.push_back(1);
		v.push_back(2);
		v.push_back(3);
		check(v.size() == 3, nrOfFailedTestCases, "push_back increases size");
		check(v.front() == 1, nrOfFailedTestCases, "front value");
		check(v.back() == 3, nrOfFailedTestCases, "back value");
		check_invariants(v, nrOfFailedTestCases, "basic invariants");
	}

	{
		sso_vector<int> v;
		v.reserve(1);
		std::size_t prev_capacity = v.capacity();
		for (int i = 0; i < 32; ++i) {
			v.emplace_back(i);
			check(v.size() == static_cast<std::size_t>(i + 1), nrOfFailedTestCases, "emplace_back size");
			check(v.capacity() >= v.size(), nrOfFailedTestCases, "capacity >= size after emplace_back");
			if (v.capacity() != prev_capacity) {
				prev_capacity = v.capacity();
			}
		}
		check(v[0] == 0 && v[31] == 31, nrOfFailedTestCases, "emplace_back data");
		check_invariants(v, nrOfFailedTestCases, "emplace_back invariants");
	}

	{
		sso_vector<int> v;
		for (int i = 0; i < 5; ++i) v.push_back(i);
		v.insert(v.begin() + 2, 42);
		check(v.size() == 6, nrOfFailedTestCases, "insert single size");
		check(v[2] == 42, nrOfFailedTestCases, "insert single value");

		v.insert(v.begin() + 1, 3, 7);
		check(v.size() == 9, nrOfFailedTestCases, "insert count size");
		check(v[1] == 7 && v[3] == 7, nrOfFailedTestCases, "insert count values");

		std::vector<int> src = {9, 8, 7};
		v.insert(v.begin() + 4, src.begin(), src.end());
		check(v.size() == 12, nrOfFailedTestCases, "insert forward range size");
		check(v[4] == 9 && v[6] == 7, nrOfFailedTestCases, "insert forward range values");

		int input_data[] = {4, 3, 2};
		InputIterator first(input_data, 0, 3);
		InputIterator last(input_data, 3, 3);
		v.insert(v.begin() + 2, first, last);
		check(v.size() == 15, nrOfFailedTestCases, "insert input range size");
		check(v[2] == 4 && v[4] == 2, nrOfFailedTestCases, "insert input range values");
		check_invariants(v, nrOfFailedTestCases, "insert invariants");
	}

	{
		sso_vector<int> v;
		for (int i = 0; i < 10; ++i) v.push_back(i);
		v.erase(v.begin() + 3);
		check(v.size() == 9, nrOfFailedTestCases, "erase single size");
		check(v[3] == 4, nrOfFailedTestCases, "erase single order");

		v.erase(v.begin() + 2, v.begin() + 5);
		check(v.size() == 6, nrOfFailedTestCases, "erase range size");
		check(v[2] == 6, nrOfFailedTestCases, "erase range order");
		check_invariants(v, nrOfFailedTestCases, "erase invariants");
	}

	{
		sso_vector<int> v;
		for (int i = 0; i < 10; ++i) v.push_back(i);
		const auto old_capacity = v.capacity();
		v.reserve(old_capacity + 10);
		check(v.capacity() >= old_capacity + 10, nrOfFailedTestCases, "reserve grows capacity");
		const auto reserve_capacity = v.capacity();
		v.shrink_to_fit();
		check(v.size() == 10, nrOfFailedTestCases, "shrink_to_fit keeps size");
		check(v.capacity() >= v.size(), nrOfFailedTestCases, "shrink_to_fit capacity >= size");
		check(v.capacity() <= reserve_capacity, nrOfFailedTestCases, "shrink_to_fit does not grow capacity");
		check_invariants(v, nrOfFailedTestCases, "shrink invariants");
	}

	{
		sso_vector<std::string> vs({"a", "b", "c"});
		sso_vector<std::string> vs_copy(vs);
		check(vs_copy == vs, nrOfFailedTestCases, "copy equality");

		sso_vector<std::string> vs_move(std::move(vs_copy));
		check(vs_move.size() == 3, nrOfFailedTestCases, "move size");
		check(vs_move[2] == "c", nrOfFailedTestCases, "move data");

		vs_move = vs_move;
		check(vs_move.size() == 3, nrOfFailedTestCases, "self copy assignment");

		sso_vector<std::string> vs_assign;
		vs_assign = vs_move;
		check(vs_assign == vs_move, nrOfFailedTestCases, "copy assignment");

		sso_vector<std::string> vs_move_assign;
		vs_move_assign = std::move(vs_assign);
		check(vs_move_assign.size() == 3, nrOfFailedTestCases, "move assignment");
		check_invariants(vs_move_assign, nrOfFailedTestCases, "copy/move invariants");
	}

	{
		sso_vector<int> a;
		sso_vector<int> b;
		for (int i = 0; i < 5; ++i) a.push_back(i);
		for (int i = 0; i < 2; ++i) b.push_back(10 + i);
		a.reserve(32);
		b.reserve(4);
		a.swap(b);
		check(a.size() == 2 && b.size() == 5, nrOfFailedTestCases, "swap sizes");
		check(a[0] == 10 && b[0] == 0, nrOfFailedTestCases, "swap contents");
		check_invariants(a, nrOfFailedTestCases, "swap invariants a");
		check_invariants(b, nrOfFailedTestCases, "swap invariants b");
	}

	{
		ThrowingType::reset();
		sso_vector<ThrowingType> v;
		v.reserve(1);
		v.emplace_back(1);
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(nrOfFailedTestCases, "push_back reallocation throws", [&]() {
			v.push_back(ThrowingType(2));
		});
		check(v.size() == 1, nrOfFailedTestCases, "size unchanged after throw");
		check(v[0].value == 1, nrOfFailedTestCases, "value preserved after throw");
		check(ThrowingType::live == 1, nrOfFailedTestCases, "no leak after throw");
		v.clear();
		check(ThrowingType::live == 0, nrOfFailedTestCases, "clear destroys elements");
	}

	{
		ThrowingType::reset();
		sso_vector<ThrowingType> v;
		for (int i = 0; i < 3; ++i) v.emplace_back(i);
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(nrOfFailedTestCases, "insert reallocation throws", [&]() {
			v.insert(v.begin() + 1, ThrowingType(9));
		});
		check(ThrowingType::live == 3, nrOfFailedTestCases, "insert throw no leak");
		check(v.size() == 3, nrOfFailedTestCases, "insert throw size unchanged");
		check_invariants(v, nrOfFailedTestCases, "insert throw invariants");
		v.clear();
		check(ThrowingType::live == 0, nrOfFailedTestCases, "clear after insert throw");
	}

	{
		AllocState state;
		using AllocVec = sso_vector<int, CountingAllocator<int>>;
		AllocVec v{CountingAllocator<int>(&state)};
		state.throw_on_alloc = 1;
		expect_throw<std::bad_alloc>(nrOfFailedTestCases, "allocator throws on reserve", [&]() {
			v.reserve(5);
		});
		check(v.size() == 0, nrOfFailedTestCases, "allocator throw size unchanged");
		check_invariants(v, nrOfFailedTestCases, "allocator throw invariants");
		check(state.alloc_calls == 1, nrOfFailedTestCases, "allocator alloc count");
	}

	{
		AllocState state;
		using AllocVec = sso_vector<LiveCountedType, CountingAllocator<LiveCountedType>>;
		AllocVec v{CountingAllocator<LiveCountedType>(&state)};
		v.push_back(LiveCountedType(1));
		state.throw_on_alloc = 2;
		expect_throw<std::bad_alloc>(nrOfFailedTestCases, "allocator throws on growth", [&]() {
			v.push_back(LiveCountedType(2));
		});
		check(LiveCountedType::live == 1, nrOfFailedTestCases, "allocator throw preserves live count");
		v.clear();
		check(LiveCountedType::live == 0, nrOfFailedTestCases, "allocator throw clear live count");
	}

	sw::universal::ReportTestResult(nrOfFailedTestCases, "sso_vector", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
