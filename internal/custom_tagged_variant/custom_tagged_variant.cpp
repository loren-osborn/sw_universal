// custom_tagged_variant.cpp: unit tests for custom_tagged_variant
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <universal/internal/custom_tagged_variant/custom_tagged_variant.hpp>
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

struct StrongGuaranteeExpectations {
	bool emplace_preserves = false;
	bool assign_preserves = false;
};

StrongGuaranteeExpectations compute_std_expectations() {
	StrongGuaranteeExpectations expectations{};
	using StdVariant = std::variant<int, std::string, ThrowingType, LiveCountedType>;
	{
		ThrowingType::reset();
		StdVariant v(123);
		ThrowingType::throw_on_default = 1;
		try {
			v.emplace<ThrowingType>();
		} catch (...) {
		}
		expectations.emplace_preserves = !v.valueless_by_exception() &&
			std::holds_alternative<int>(v) &&
			std::get<int>(v) == 123;
	}
	{
		ThrowingType::reset();
		StdVariant v(123);
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		try {
			v = ThrowingType(9);
		} catch (...) {
		}
		expectations.assign_preserves = !v.valueless_by_exception() &&
			std::holds_alternative<int>(v) &&
			std::get<int>(v) == 123;
	}
	return expectations;
}

namespace variant_test {
	using std::get;
	using std::get_if;
	using std::holds_alternative;
	using sw::universal::internal::get;
	using sw::universal::internal::get_if;
	using sw::universal::internal::holds_alternative;

	template<typename Variant>
	struct is_custom_variant : std::false_type {};

	template<template<std::size_t NTypes> class EncodedTag, typename... Ts>
	struct is_custom_variant<sw::universal::internal::custom_tagged_variant<EncodedTag, Ts...>> : std::true_type {};

	template<typename Variant>
	inline constexpr bool is_custom_variant_v = is_custom_variant<std::decay_t<Variant>>::value;

	template<typename Visitor, typename... Variants>
	decltype(auto) visit_adapter(Visitor&& vis, Variants&&... variants) {
		if constexpr ((is_custom_variant_v<Variants> || ...)) {
			return sw::universal::internal::visit(std::forward<Visitor>(vis), std::forward<Variants>(variants)...);
		} else {
			return std::visit(std::forward<Visitor>(vis), std::forward<Variants>(variants)...);
		}
	}
}

} // namespace

template<class... Ts>
using CustomVariant = sw::universal::internal::custom_tagged_variant<sw::universal::internal::simple_encoded_tag, Ts...>;

template<class... Ts>
using SidebandVariant = sw::universal::internal::custom_tagged_variant<sw::universal::internal::tag_encoded_with_sideband_data, Ts...>;

void run_encoded_tag_tests(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};

	{
		sw::universal::internal::simple_encoded_tag<3> tag{};
		check(ctx, tag.tag() == 0, "simple_encoded_tag default tag is 0");
		tag.set_tag(2);
		check(ctx, tag.tag() == 2, "simple_encoded_tag set/get normal value");
		tag.set_tag(std::variant_npos);
		check(ctx, tag.tag() == std::variant_npos, "simple_encoded_tag variant_npos round trip");
	}

	{
		using Tag = sw::universal::internal::tag_encoded_with_sideband_data<3>;
		Tag tag{};
		check(ctx, tag.tag() == 0, "tag_encoded_with_sideband_data default tag is 0");
		check(ctx, static_cast<std::size_t>(tag.sideband()) == 0, "tag_encoded_with_sideband_data default sideband is 0");
		tag.set_tag(1);
		check(ctx, tag.tag() == 1, "tag_encoded_with_sideband_data stores tag");
		tag.set_tag(2);
		tag.sideband() = 5;
		check(ctx, tag.tag() == 2, "tag_encoded_with_sideband_data sideband write preserves tag");
		check(ctx, static_cast<std::size_t>(tag.sideband()) == 5, "tag_encoded_with_sideband_data sideband shift semantics");
		tag.set_tag(std::variant_npos);
		check(ctx, tag.tag() == std::variant_npos, "tag_encoded_with_sideband_data npos round trip");
		check(ctx, static_cast<std::size_t>(tag.sideband()) == 5, "tag_encoded_with_sideband_data npos preserves sideband");
		tag.set_tag(1);
		check(ctx, static_cast<std::size_t>(tag.sideband()) == 5, "tag_encoded_with_sideband_data tag write preserves sideband");
		tag.sideband() = 3;
		check(ctx, tag.tag() == 1, "tag_encoded_with_sideband_data sideband write preserves tag");

		auto proxy_a = tag.sideband();
		auto proxy_b = tag.sideband();
		proxy_a = 7;
		check(ctx, static_cast<std::size_t>(proxy_b) == 7, "tag_encoded_with_sideband_data proxy coherence A->B");
		proxy_b = 2;
		check(ctx, static_cast<std::size_t>(proxy_a) == 2, "tag_encoded_with_sideband_data proxy coherence B->A");
	}

	{
		using Tag = sw::universal::internal::tag_encoded_with_sideband_data<7>;
		Tag tag{};
		check(ctx, tag.tag() == 0, "tag_encoded_with_sideband_data(7) default tag is 0");
		tag.set_tag(6);
		check(ctx, tag.tag() == 6, "tag_encoded_with_sideband_data(7) stores tag");
		tag.sideband() = 12;
		check(ctx, static_cast<std::size_t>(tag.sideband()) == 12, "tag_encoded_with_sideband_data(7) sideband round trip");
	}
}

template<template<class...> class Variant>
void run_variant_suite(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};
	using namespace variant_test;
	using VariantT = Variant<int, std::string, ThrowingType, LiveCountedType>;
	const auto expectations = compute_std_expectations();

	{
		VariantT v;
		check(ctx, v.index() == 0, "default index is 0");
		check(ctx, holds_alternative<int>(v), "default holds int");
		check(ctx, get<int>(v) == 0, "default int value is 0");

		VariantT v2(std::in_place_type<std::string>, "hello");
		check(ctx, v2.index() == 1, "in_place_type sets index");
		check(ctx, get<std::string>(v2) == "hello", "in_place_type stores string");

		VariantT v3(std::in_place_index<3>, LiveCountedType{42});
		check(ctx, v3.index() == 3, "in_place_index sets index");
		check(ctx, get<3>(v3).value == 42, "in_place_index stores value");
	}

	{
		VariantT v(ThrowingType{7});
		check(ctx, v.index() == 2, "exact-type converting ctor");
		v = ThrowingType{9};
		check(ctx, get<ThrowingType>(v).value == 9, "assign from exact type");
	}

	{
		VariantT v(std::in_place_type<std::string>, "text");
		check(ctx, get<1>(v) == "text", "get by index");
		check(ctx, get<std::string>(v) == "text", "get by type");
		check(ctx, get_if<1>(&v) != nullptr, "get_if by index non-null");
		check(ctx, get_if<2>(&v) == nullptr, "get_if by index null when inactive");
		check(ctx, get_if<std::string>(&v) != nullptr, "get_if by type non-null");
		check(ctx, holds_alternative<std::string>(v), "holds_alternative by type");

		const VariantT cv(v);
		check(ctx, get<1>(cv) == "text", "const get by index");
		check(ctx, get<std::string>(cv) == "text", "const get by type");

		auto moved = get<1>(std::move(v));
		check(ctx, moved == "text", "rvalue get by index");
	}

	{
		VariantT a(1);
		VariantT b(2);
		a.swap(b);
		check(ctx, get<int>(a) == 2 && get<int>(b) == 1, "swap same index");

		VariantT c(3);
		VariantT d(std::in_place_type<std::string>, "world");
		c.swap(d);
		check(ctx, c.index() == 1, "swap different index");
		check(ctx, get<std::string>(c) == "world", "swap moves string");
	}

	{
		VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{5});
		int visit_result = visit_adapter([](auto& value) {
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
		check(ctx, visit_result == 5, "visit single variant");

		VariantT x(4);
		VariantT y(std::in_place_type<std::string>, "abc");
		int multi_result = visit_adapter([](auto& left, auto& right) {
			int lhs = 0;
			int rhs = 0;
			if constexpr (std::is_same_v<std::decay_t<decltype(left)>, int>) lhs = left;
			if constexpr (std::is_same_v<std::decay_t<decltype(right)>, std::string>) rhs = static_cast<int>(right.size());
			return lhs + rhs;
		}, x, y);
		check(ctx, multi_result == 7, "visit multiple variants");
	}

	{
		ThrowingType::reset();
		VariantT v(123);
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "emplace throws", [&]() {
			v.template emplace<ThrowingType>();
		});
		const bool preserved = !v.valueless_by_exception() &&
			holds_alternative<int>(v) &&
			get<int>(v) == 123;
		check(ctx, preserved == expectations.emplace_preserves, "emplace throw preserves per std");
		if (!preserved) {
			check(ctx, v.valueless_by_exception(), "emplace throw leaves valueless when not preserved");
		}
		v = 3;
		check(ctx, get<int>(v) == 3, "recovery after emplace throw");
	}

	{
		ThrowingType::reset();
		VariantT v(123);
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		expect_throw<std::runtime_error>(ctx, "assign different alternative throws", [&]() {
			v = ThrowingType(9);
		});
		const bool preserved = !v.valueless_by_exception() &&
			holds_alternative<int>(v) &&
			get<int>(v) == 123;
		check(ctx, preserved == expectations.assign_preserves, "assign throw preserves per std");
		if (!preserved) {
			check(ctx, v.valueless_by_exception(), "assign throw leaves valueless when not preserved");
		}
		v = std::string("ok");
		check(ctx, get<std::string>(v) == "ok", "assign after throw");
	}

	{
		check(ctx, LiveCountedType::live == 0, "live count starts at 0");
		{
			VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{11});
			check(ctx, LiveCountedType::live == 1, "live count increments");
		}
		check(ctx, LiveCountedType::live == 0, "live count decrements");
	}

	{
		check(ctx, LiveCountedType::live == 0, "live count before repeated assigns");
		{
			VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{5});
			for (int i = 0; i < 8; ++i) {
				v = LiveCountedType{i};
				v = std::string("swap");
				v = i;
			}
		}
		check(ctx, LiveCountedType::live == 0, "live count after repeated assigns");
	}

	{
		ThrowingType::reset();
		VariantT left(1);
		VariantT right(std::in_place_type<ThrowingType>, ThrowingType{7});
		ThrowingType::throw_on_copy = 1;
		ThrowingType::throw_on_move = 1;
		ThrowingType::throw_on_copy_assign = 1;
		ThrowingType::throw_on_move_assign = 1;
		bool threw = false;
		try {
			left.swap(right);
		} catch (const std::runtime_error&) {
			threw = true;
		}
		int held_throwing = 0;
		if (!left.valueless_by_exception() && holds_alternative<ThrowingType>(left)) {
			++held_throwing;
		}
		if (!right.valueless_by_exception() && holds_alternative<ThrowingType>(right)) {
			++held_throwing;
		}
		check(ctx, ThrowingType::live == held_throwing, "swap throw live count matches");
		if (!threw) {
			check(ctx, ThrowingType::live == 1, "swap no-throw keeps one ThrowingType alive");
		}
		left = 7;
		right = std::string("ok");
		check(ctx, holds_alternative<int>(left) && get<int>(left) == 7, "swap throw recovery left");
		check(ctx, holds_alternative<std::string>(right) && get<std::string>(right) == "ok", "swap throw recovery right");
	}
}

int main() {
	int nrOfFailedTestCases = 0;
	run_encoded_tag_tests("encoded_tag", nrOfFailedTestCases);
	run_variant_suite<CustomVariant>("custom_tagged_variant", nrOfFailedTestCases);
	run_variant_suite<SidebandVariant>("custom_tagged_variant_sideband", nrOfFailedTestCases);
	run_variant_suite<std::variant>("std::variant", nrOfFailedTestCases);

	sw::universal::ReportTestResult(nrOfFailedTestCases, "custom_tagged_variant", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
