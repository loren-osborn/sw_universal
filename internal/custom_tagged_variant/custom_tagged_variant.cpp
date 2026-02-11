// custom_tagged_variant.cpp: unit tests for custom_tagged_variant
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <universal/internal/custom_tagged_variant/custom_tagged_variant.hpp>
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

} // namespace

int main() {
	using namespace sw::universal::internal;
	int nrOfFailedTestCases = 0;

	using Variant = custom_tagged_variant<void, int, std::string, ThrowingType, LiveCountedType>;

	{
		Variant v;
		check(v.index() == 0, nrOfFailedTestCases, "default index is 0");
		check(v.holds_alternative<int>(), nrOfFailedTestCases, "default holds int");
		check(v.get<int>() == 0, nrOfFailedTestCases, "default int value is 0");

		Variant v2(std::in_place_type<std::string>, "hello");
		check(v2.index() == 1, nrOfFailedTestCases, "in_place_type sets index");
		check(v2.get<std::string>() == "hello", nrOfFailedTestCases, "in_place_type stores string");

		Variant v3(std::in_place_index<3>, LiveCountedType{42});
		check(v3.index() == 3, nrOfFailedTestCases, "in_place_index sets index");
		check(v3.get<3>().value == 42, nrOfFailedTestCases, "in_place_index stores value");
	}

	{
		Variant v(ThrowingType{7});
		check(v.index() == 2, nrOfFailedTestCases, "exact-type converting ctor");
		v = ThrowingType{9};
		check(v.get<ThrowingType>().value == 9, nrOfFailedTestCases, "assign from exact type");
	}

	{
		Variant v(std::in_place_type<std::string>, "text");
		check(v.get<1>() == "text", nrOfFailedTestCases, "get by index");
		check(v.get<std::string>() == "text", nrOfFailedTestCases, "get by type");
		check(get_if<1>(&v) != nullptr, nrOfFailedTestCases, "get_if by index non-null");
		check(get_if<2>(&v) == nullptr, nrOfFailedTestCases, "get_if by index null when inactive");
		check(get_if<std::string>(&v) != nullptr, nrOfFailedTestCases, "get_if by type non-null");
		check(holds_alternative<std::string>(v), nrOfFailedTestCases, "holds_alternative by type");

		const Variant cv(v);
		check(cv.get<1>() == "text", nrOfFailedTestCases, "const get by index");
		check(cv.get<std::string>() == "text", nrOfFailedTestCases, "const get by type");

		auto moved = std::move(v).get<1>();
		check(moved == "text", nrOfFailedTestCases, "rvalue get by index");
	}

	{
		Variant a(1);
		Variant b(2);
		a.swap(b);
		check(a.get<int>() == 2 && b.get<int>() == 1, nrOfFailedTestCases, "swap same index");

		Variant c(3);
		Variant d(std::in_place_type<std::string>, "world");
		c.swap(d);
		check(c.index() == 1, nrOfFailedTestCases, "swap different index");
		check(c.get<std::string>() == "world", nrOfFailedTestCases, "swap moves string");
	}

	{
		Variant v(std::in_place_type<LiveCountedType>, LiveCountedType{5});
		int visit_result = visit([](auto& value) {
			if constexpr (std::is_same_v<std::decay_t<decltype(value)>, int>) {
				return value;
			} else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
				return static_cast<int>(value.size());
			} else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, ThrowingType>) {
				return value.value;
			} else {
				return value.value;
			}
		}, v);
		check(visit_result == 5, nrOfFailedTestCases, "visit single variant");

		Variant x(4);
		Variant y(std::in_place_type<std::string>, "abc");
		int multi_result = visit([](auto& left, auto& right) {
			int lhs = 0;
			int rhs = 0;
			if constexpr (std::is_same_v<std::decay_t<decltype(left)>, int>) lhs = left;
			if constexpr (std::is_same_v<std::decay_t<decltype(right)>, std::string>) rhs = static_cast<int>(right.size());
			return lhs + rhs;
		}, x, y);
		check(multi_result == 7, nrOfFailedTestCases, "visit multiple variants");
	}

	{
		ThrowingType::reset();
		Variant v(1);
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(nrOfFailedTestCases, "emplace throws", [&]() {
			v.emplace<ThrowingType>();
		});
		check(v.valueless_by_exception(), nrOfFailedTestCases, "emplace leaves valueless_by_exception");
		check(v.index() == Variant::npos, nrOfFailedTestCases, "emplace valueless index");
		check(get_if<ThrowingType>(&v) == nullptr, nrOfFailedTestCases, "emplace valueless get_if null");
		v = 3;
		check(v.get<int>() == 3, nrOfFailedTestCases, "recovery after valueless");
	}

	{
		ThrowingType::reset();
		Variant v(2);
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(nrOfFailedTestCases, "assign different alternative throws", [&]() {
			v = ThrowingType(9);
		});
		check(v.valueless_by_exception(), nrOfFailedTestCases, "assign throw leaves valueless");
		check(v.index() == Variant::npos, nrOfFailedTestCases, "assign throw index npos");
		v = std::string("ok");
		check(v.get<std::string>() == "ok", nrOfFailedTestCases, "assign after valueless");
	}

	{
		check(LiveCountedType::live == 0, nrOfFailedTestCases, "live count starts at 0");
		{
			Variant v(std::in_place_type<LiveCountedType>, LiveCountedType{11});
			check(LiveCountedType::live == 1, nrOfFailedTestCases, "live count increments");
		}
		check(LiveCountedType::live == 0, nrOfFailedTestCases, "live count decrements");
	}

	sw::universal::ReportTestResult(nrOfFailedTestCases, "custom_tagged_variant", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
