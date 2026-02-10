// custom_tagged_variant.cpp: unit tests for custom_tagged_variant
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

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

struct Tracker {
	int value{0};
	static int live;
	Tracker() { ++live; }
	explicit Tracker(int v) : value(v) { ++live; }
	Tracker(const Tracker& other) : value(other.value) { ++live; }
	Tracker(Tracker&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	Tracker& operator=(const Tracker& other) { value = other.value; return *this; }
	Tracker& operator=(Tracker&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~Tracker() { --live; }
};

int Tracker::live = 0;

} // namespace

int main() {
	using namespace sw::universal::internal;
	int nrOfFailedTestCases = 0;

	using Variant = custom_tagged_variant<void, int, std::string, Tracker>;

	Variant v;
	check(v.index() == 0, nrOfFailedTestCases, "default index is 0");
	check(v.holds_alternative<int>(), nrOfFailedTestCases, "default holds int");
	check(v.get<int>() == 0, nrOfFailedTestCases, "default int value is 0");

	Variant v2(std::in_place_type<std::string>, "hello");
	check(v2.index() == 1, nrOfFailedTestCases, "in_place_type sets index");
	check(v2.get<std::string>() == "hello", nrOfFailedTestCases, "in_place_type stores string");

	v2.emplace<2>(Tracker{42});
	check(v2.index() == 2, nrOfFailedTestCases, "emplace by index sets index");
	check(v2.get<Tracker>().value == 42, nrOfFailedTestCases, "emplace by index stores tracker");
	check(Tracker::live > 0, nrOfFailedTestCases, "tracker live count positive");

	Variant v3 = v2;
	check(v3.index() == v2.index(), nrOfFailedTestCases, "copy preserves index");
	check(v3.get<Tracker>().value == 42, nrOfFailedTestCases, "copy preserves value");

	Variant v4 = std::move(v3);
	check(v4.index() == 2, nrOfFailedTestCases, "move preserves index");
	check(v4.get<Tracker>().value == 42, nrOfFailedTestCases, "move preserves value");

	Variant v5(Tracker{7});
	check(v5.index() == 2, nrOfFailedTestCases, "exact-type converting ctor");
	v5 = Tracker{9};
	check(v5.get<Tracker>().value == 9, nrOfFailedTestCases, "assign from exact type");

	Variant v6(std::in_place_index<1>, "text");
	check(v6.get<1>() == "text", nrOfFailedTestCases, "in_place_index stores value");

	check(get_if<1>(&v6) != nullptr, nrOfFailedTestCases, "get_if by index returns pointer");
	check(get_if<2>(&v6) == nullptr, nrOfFailedTestCases, "get_if by index returns null when inactive");

	expect_throw<bad_variant_access>(nrOfFailedTestCases, "get wrong index throws", [&]() {
		(void)v6.get<0>();
	});

	Variant a(Tracker{1});
	Variant b(std::in_place_type<std::string>, "world");
	a.swap(b);
	check(a.index() == 1, nrOfFailedTestCases, "swap changes index");
	check(a.get<std::string>() == "world", nrOfFailedTestCases, "swap moves string");

	Variant v7(Tracker{5});
	int visit_result = visit([](auto& value) {
		if constexpr (std::is_same_v<std::decay_t<decltype(value)>, int>) return value;
		if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) return static_cast<int>(value.size());
		return value.value;
	}, v7);
	check(visit_result == 5, nrOfFailedTestCases, "visit single variant");

	Variant x(Tracker{3});
	Variant y(std::in_place_type<std::string>, "abc");
	int multi_result = visit([](auto& left, auto& right) {
		int lhs = 0;
		int rhs = 0;
		if constexpr (std::is_same_v<std::decay_t<decltype(left)>, Tracker>) lhs = left.value;
		if constexpr (std::is_same_v<std::decay_t<decltype(right)>, std::string>) rhs = static_cast<int>(right.size());
		return lhs + rhs;
	}, x, y);
	check(multi_result == 6, nrOfFailedTestCases, "visit multiple variants");

	ReportTestResult(nrOfFailedTestCases, "custom_tagged_variant", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
