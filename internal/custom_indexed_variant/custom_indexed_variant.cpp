// custom_indexed_variant.cpp: unit tests for custom_indexed_variant
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <universal/internal/custom_indexed_variant/custom_indexed_variant.hpp>
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
	static int min_live;
	int value{0};
	static void reset() {
		live = 0;
		min_live = 0;
	}
	LiveCountedType() { ++live; }
	explicit LiveCountedType(int v) : value(v) { ++live; }
	LiveCountedType(const LiveCountedType& other) : value(other.value) { ++live; }
	LiveCountedType(LiveCountedType&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	LiveCountedType& operator=(const LiveCountedType& other) { value = other.value; return *this; }
	LiveCountedType& operator=(LiveCountedType&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~LiveCountedType() { --live; min_live = (std::min)(min_live, live); }
};

int LiveCountedType::live = 0;
int LiveCountedType::min_live = 0;

struct ThrowBeforeDestroyType {
	static int live;
	static int ctor_count;
	static int throw_on_ctor;
	int value{0};

	static void reset() {
		live = 0;
		ctor_count = 0;
		throw_on_ctor = -1;
	}

	explicit ThrowBeforeDestroyType(int v = 0) {
		++ctor_count;
		if (throw_on_ctor >= 0 && ctor_count == throw_on_ctor) {
			throw std::runtime_error("ThrowBeforeDestroyType configured throw");
		}
		value = v;
		++live;
	}

	ThrowBeforeDestroyType(const ThrowBeforeDestroyType&) = delete;
	ThrowBeforeDestroyType& operator=(const ThrowBeforeDestroyType&) = delete;
	ThrowBeforeDestroyType(ThrowBeforeDestroyType&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	ThrowBeforeDestroyType& operator=(ThrowBeforeDestroyType&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~ThrowBeforeDestroyType() { --live; }
};

int ThrowBeforeDestroyType::live = 0;
int ThrowBeforeDestroyType::ctor_count = 0;
int ThrowBeforeDestroyType::throw_on_ctor = -1;

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

enum class VisitRefCategory {
	LValue = 1,
	ConstLValue = 2,
	RValue = 3,
	ConstRValue = 4
};

struct VisitCategoryVisitor {
	VisitRefCategory operator()(int&) const { return VisitRefCategory::LValue; }
	VisitRefCategory operator()(const int&) const { return VisitRefCategory::ConstLValue; }
	VisitRefCategory operator()(int&&) const { return VisitRefCategory::RValue; }
	VisitRefCategory operator()(const int&&) const { return VisitRefCategory::ConstRValue; }
};

struct AmbiguousSource {};

struct AmbiguousA {
	int value{0};
	AmbiguousA() = default;
	AmbiguousA(AmbiguousSource) noexcept : value(1) {}
	AmbiguousA& operator=(AmbiguousSource) noexcept { value = 11; return *this; }
};

struct AmbiguousB {
	int value{0};
	AmbiguousB() = default;
	AmbiguousB(AmbiguousSource) noexcept : value(2) {}
	AmbiguousB& operator=(AmbiguousSource) noexcept { value = 22; return *this; }
};

struct ExplicitOnlyInt {
	int value{0};
	ExplicitOnlyInt() = default;
	explicit ExplicitOnlyInt(int v) noexcept : value(v) {}
	ExplicitOnlyInt& operator=(int v) noexcept { value = v; return *this; }
};

struct AssignToken {};

struct CtorOnlyFromAssignToken {
	int value{0};
	CtorOnlyFromAssignToken() = default;
	CtorOnlyFromAssignToken(AssignToken) noexcept : value(1) {}
	CtorOnlyFromAssignToken& operator=(AssignToken) = delete;
};

struct AssignableFromAssignToken {
	int value{0};
	AssignableFromAssignToken() = default;
	AssignableFromAssignToken(AssignToken) noexcept : value(3) {}
	AssignableFromAssignToken& operator=(AssignToken) noexcept { value = 7; return *this; }
};

struct NoTokenSupport {
	int value{0};
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

	template<template<std::size_t NTypes> class EncodedIndex, typename... Ts>
	struct is_custom_variant<sw::universal::internal::custom_indexed_variant<EncodedIndex, Ts...>> : std::true_type {};

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
using CustomVariant = sw::universal::internal::custom_indexed_variant<sw::universal::internal::simple_encoded_index, Ts...>;

template<class... Ts>
using SidebandVariant = sw::universal::internal::custom_indexed_variant<sw::universal::internal::index_encoded_with_sideband_data, Ts...>;

template<typename V>
constexpr bool variant_has_sideband() {
	if constexpr (requires(V& v) { v.sideband(); }) {
		return true;
	} else {
		return false;
	}
}

template<template<class...> class Variant>
constexpr bool run_converting_selection_static_checks_common() {
	{
		using V = Variant<int, std::string>;
		static_assert(std::is_constructible_v<V, int>);
	}
	{
		using V = Variant<const char*, std::string>;
		static_assert(std::is_constructible_v<V, const char*>);
		static_assert(std::is_assignable_v<V&, const char*>);
	}
	{
		using V = Variant<long, std::string>;
		static_assert(std::is_constructible_v<V, int>);
	}
	{
		using V = Variant<AmbiguousA, AmbiguousB>;
		static_assert(!std::is_constructible_v<V, AmbiguousSource>);
		static_assert(!std::is_assignable_v<V&, AmbiguousSource>);
	}
	{
		using V = Variant<NoTokenSupport, AssignableFromAssignToken>;
		static_assert(std::is_assignable_v<V&, AssignToken>);
	}
	return true;
}

template<template<class...> class Variant>
constexpr bool run_converting_selection_static_checks_custom_only() {
	{
		using V = Variant<ExplicitOnlyInt>;
		static_assert(std::is_constructible_v<V, int>);
		static_assert(!std::is_convertible_v<int, V>);
		static_assert(requires { V{1}; });
	}
	return true;
}

template<typename V>
constexpr bool const_sideband_readable() {
	return requires(const V& v) { v.sideband().val(); };
}

template<typename V>
constexpr bool const_sideband_writable() {
	return requires(const V& v) { v.sideband().set_val(1); };
}

template<std::size_t>
struct NonNoexceptEncodedIndex {
	std::size_t index() const { return std::variant_npos; }
	void set_index(std::size_t) {}
};

static_assert(sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	sw::universal::internal::simple_encoded_index<4>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	sw::universal::internal::index_encoded_with_sideband_data<4>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	NonNoexceptEncodedIndex<4>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_sideband_v<
	sw::universal::internal::simple_encoded_index<1>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::has_sideband_v<
	sw::universal::internal::index_encoded_with_sideband_data<1>>);
static_assert(!variant_has_sideband<CustomVariant<int>>());
static_assert(variant_has_sideband<SidebandVariant<int>>());
static_assert(!variant_has_sideband<std::variant<int>>());
static_assert(run_converting_selection_static_checks_common<CustomVariant>());
static_assert(run_converting_selection_static_checks_common<SidebandVariant>());
static_assert(run_converting_selection_static_checks_common<std::variant>());
static_assert(run_converting_selection_static_checks_custom_only<CustomVariant>());
static_assert(run_converting_selection_static_checks_custom_only<SidebandVariant>());
static_assert(const_sideband_readable<SidebandVariant<int>>());
static_assert(!const_sideband_writable<SidebandVariant<int>>());

#ifdef UNIVERSAL_COMPILE_FAIL_TESTS
struct ThrowingDestructorType {
	~ThrowingDestructorType() noexcept(false) {}
};
using CompileFailNothrowDtorVariant = CustomVariant<int, ThrowingDestructorType>;
static_assert(sizeof(CompileFailNothrowDtorVariant) > 0, "compile-fail guard for nothrow dtor requirement");
#endif

void run_encoded_index_tests(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};

		{
			sw::universal::internal::simple_encoded_index<3> index{};
			check(ctx, index.index() == std::variant_npos, "simple_encoded_index default index is std::variant_npos");
		index.set_index(2);
		check(ctx, index.index() == 2, "simple_encoded_index set/get normal value");
		index.set_index(std::variant_npos);
		check(ctx, index.index() == std::variant_npos, "simple_encoded_index variant_npos round trip");
	}

	{
		using Index = sw::universal::internal::index_encoded_with_sideband_data<3>;
		Index index{};
		check(ctx, index.index() == std::variant_npos, "index_encoded_with_sideband_data default index is std::variant_npos");
		check(ctx, static_cast<std::size_t>(index.sideband().val()) == 0, "index_encoded_with_sideband_data default sideband is 0");
		index.set_index(1);
		check(ctx, index.index() == 1, "index_encoded_with_sideband_data stores index");
		index.set_index(2);
		index.sideband().set_val(5);
		check(ctx, index.index() == 2, "index_encoded_with_sideband_data sideband write preserves index");
		check(ctx, static_cast<std::size_t>(index.sideband().val()) == 5, "index_encoded_with_sideband_data sideband shift semantics");
		index.set_index(std::variant_npos);
		check(ctx, index.index() == std::variant_npos, "index_encoded_with_sideband_data npos round trip");
		check(ctx, static_cast<std::size_t>(index.sideband().val()) == 5, "index_encoded_with_sideband_data npos preserves sideband");
		index.set_index(1);
		check(ctx, static_cast<std::size_t>(index.sideband().val()) == 5, "index_encoded_with_sideband_data index write preserves sideband");
		index.sideband().set_val(3);
		check(ctx, index.index() == 1, "index_encoded_with_sideband_data sideband write preserves index");

		auto proxy_a = index.sideband();
		auto proxy_b = index.sideband();
		proxy_a.set_val(7);
		check(ctx, static_cast<std::size_t>(proxy_b.val()) == 7, "index_encoded_with_sideband_data proxy coherence A->B");
		proxy_b.set_val(2);
		check(ctx, static_cast<std::size_t>(proxy_a.val()) == 2, "index_encoded_with_sideband_data proxy coherence B->A");
	}

	{
		using Index = sw::universal::internal::index_encoded_with_sideband_data<7>;
		Index index{};
		check(ctx, index.index() == std::variant_npos, "index_encoded_with_sideband_data(7) default index is std::variant_npos");
		index.set_index(6);
		check(ctx, index.index() == 6, "index_encoded_with_sideband_data(7) stores index");
		index.sideband().set_val(12);
		check(ctx, static_cast<std::size_t>(index.sideband().val()) == 12, "index_encoded_with_sideband_data(7) sideband round trip");
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
		using ExactNumeric = Variant<int, std::string>;
		ExactNumeric v = 7;
		check(ctx, v.index() == 0, "converting ctor exact match wins for int");
		check(ctx, get<int>(v) == 7, "converting ctor exact match stores int");
	}

	{
		using ExactPtr = Variant<const char*, std::string>;
		ExactPtr v = "hi";
		check(ctx, v.index() == 0, "converting ctor exact match wins over std::string");
		check(ctx, std::string(get<const char*>(v)) == "hi", "converting ctor exact match stores const char*");
		v = std::string("seed");
		v = "bye";
		check(ctx, v.index() == 0, "converting assignment exact match wins over std::string");
		check(ctx, std::string(get<const char*>(v)) == "bye", "converting assignment exact match stores const char*");
	}

	{
		using UniqueConvertible = Variant<long, std::string>;
		UniqueConvertible v = 7;
		check(ctx, v.index() == 0, "converting ctor unique convertible candidate selected");
		check(ctx, get<long>(v) == 7L, "converting ctor stores selected long alternative");
	}

	{
		using AssignFiltered = Variant<NoTokenSupport, AssignableFromAssignToken>;
		AssignFiltered v(std::in_place_index<0>);
		v = AssignToken{};
		check(ctx, v.index() == 1, "converting assignment selects unique assignable/constructible alternative");
		check(ctx, get<AssignableFromAssignToken>(v).value == 3, "converting assignment constructs selected alternative");
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

	if constexpr (requires(VariantT& v) { v.sideband(); }) {
		VariantT v(std::in_place_type<std::string>, "sb");
		auto sa = v.sideband();
		auto sb = v.sideband();
		sa.set_val(9);
		check(ctx, static_cast<std::size_t>(sb.val()) == 9, "sideband proxies are coherent");
		check(ctx, v.index() == 1, "sideband write preserves active index");
		v.template emplace<0>(42);
		check(ctx, static_cast<std::size_t>(v.sideband().val()) == 9, "sideband survives index update");
		check(ctx, v.index() == 0, "emplace updates active index with sideband present");
		const VariantT& cv = v;
		auto csb = cv.sideband();
		check(ctx, static_cast<std::size_t>(csb.val()) == 9, "const sideband proxy reads sideband");
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
		using RefVariant = Variant<int>;
		RefVariant v(7);
		const RefVariant cv(9);
		check(
			ctx,
			visit_adapter(VisitCategoryVisitor{}, v) == VisitRefCategory::LValue,
			"visit forwards lvalue variant as T&");
		check(
			ctx,
			visit_adapter(VisitCategoryVisitor{}, cv) == VisitRefCategory::ConstLValue,
			"visit forwards const lvalue variant as const T&");
		check(
			ctx,
			visit_adapter(VisitCategoryVisitor{}, std::move(v)) == VisitRefCategory::RValue,
			"visit forwards rvalue variant as T&&");
		check(
			ctx,
			visit_adapter(VisitCategoryVisitor{}, std::move(cv)) == VisitRefCategory::ConstRValue,
			"visit forwards const rvalue variant as const T&&");
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
			check(ctx, v.index() == std::variant_npos, "emplace throw valueless index is npos");
		}
		v = 3;
		check(ctx, get<int>(v) == 3, "recovery after emplace throw");
	}

	{
		ThrowingType::reset();
		LiveCountedType::reset();
		VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{55});
		check(ctx, LiveCountedType::live == 1, "throw-emplace baseline live count");
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "throw-emplace leaves valueless", [&]() {
			v.template emplace<ThrowingType>();
		});
		check(ctx, v.valueless_by_exception(), "throw-emplace produces valueless state");
		check(ctx, v.index() == std::variant_npos, "throw-emplace valueless index is npos");
		check(ctx, ThrowingType::live == 0, "throw-emplace does not leak ThrowingType");
		check(ctx, LiveCountedType::live == 0, "throw-emplace destroys prior alternative exactly once");
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
			check(ctx, v.index() == std::variant_npos, "assign throw valueless index is npos");
		}
		v = std::string("ok");
		check(ctx, get<std::string>(v) == "ok", "assign after throw");
	}

	{
		ThrowingType::reset();
		VariantT source(7);
		VariantT dest(std::in_place_type<std::string>, "keep");
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "force source valueless", [&]() {
			source.template emplace<ThrowingType>();
		});
		if (source.valueless_by_exception()) {
			check(ctx, source.index() == std::variant_npos, "source valueless index is npos");
			dest = source;
			check(ctx, dest.valueless_by_exception(), "assign from valueless resets destination");
			check(ctx, dest.index() == std::variant_npos, "assign from valueless sets destination index to npos");
		}
	}

	{
		LiveCountedType::reset();
		check(ctx, LiveCountedType::live == 0, "live count starts at 0");
		{
			VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{11});
			check(ctx, LiveCountedType::live == 1, "live count increments");
		}
		check(ctx, LiveCountedType::live == 0, "live count decrements");
	}

	{
		LiveCountedType::reset();
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
		LiveCountedType::reset();
		{
			VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{1});
			check(ctx, LiveCountedType::live == 1, "emplace double-destroy baseline live count");
			for (int i = 0; i < 12; ++i) {
				v.template emplace<0>(i);
				check(ctx, LiveCountedType::live == 0, "emplace to int destroys LiveCountedType exactly once");
				v.template emplace<3>(LiveCountedType{i + 10});
				check(ctx, LiveCountedType::live == 1, "emplace back to LiveCountedType constructs exactly once");
			}
		}
		check(ctx, LiveCountedType::live == 0, "emplace double-destroy final live count");
		check(ctx, LiveCountedType::min_live >= 0, "LiveCountedType live count never negative during emplace churn");
	}

	if constexpr (variant_test::is_custom_variant_v<VariantT>) {
		using ThrowBeforeDestroyVariant = Variant<int, ThrowBeforeDestroyType, std::string, LiveCountedType>;
		ThrowBeforeDestroyType::reset();
		ThrowBeforeDestroyVariant v(77);
		ThrowBeforeDestroyType::throw_on_ctor = 1;
		expect_throw<std::runtime_error>(ctx, "throw-before-destroy emplace", [&]() {
			v.template emplace<1>(1234);
		});
		check(ctx, !v.valueless_by_exception(), "throw-before-destroy keeps prior state");
		check(ctx, v.index() == 0, "throw-before-destroy preserves prior index");
		check(ctx, get<int>(v) == 77, "throw-before-destroy preserves prior value");
		check(ctx, ThrowBeforeDestroyType::live == 0, "throw-before-destroy does not leak construction");
	}

	if constexpr (variant_test::is_custom_variant_v<VariantT>) {
		ThrowingType::reset();
		LiveCountedType::reset();
		VariantT source(std::in_place_type<LiveCountedType>, LiveCountedType{44});
		check(ctx, LiveCountedType::live == 1, "destroy idempotence setup live count");
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "destroy idempotence force valueless source", [&]() {
			source.template emplace<ThrowingType>();
		});
		check(ctx, source.valueless_by_exception(), "destroy idempotence source valueless");
		check(ctx, source.index() == std::variant_npos, "destroy idempotence source index is npos");
		check(ctx, LiveCountedType::live == 0, "destroy idempotence source destruction accounted");

		VariantT target;
		target = source; // engaged -> valueless (destroy_active work path)
		check(ctx, target.valueless_by_exception(), "assign from valueless yields valueless target");
		check(ctx, target.index() == std::variant_npos, "assign from valueless sets npos");

		const int live_before = LiveCountedType::live;
		target = source; // valueless -> valueless (destroy_active no-op path)
		check(ctx, target.valueless_by_exception(), "assign valueless to already valueless stays valueless");
		check(ctx, target.index() == std::variant_npos, "assign valueless to already valueless keeps npos");
		check(ctx, LiveCountedType::live == live_before, "assign valueless to already valueless does not destroy anything");
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
		if (threw && left.valueless_by_exception()) {
			check(ctx, left.index() == std::variant_npos, "swap throw valueless left index is npos");
		}
		if (threw && right.valueless_by_exception()) {
			check(ctx, right.index() == std::variant_npos, "swap throw valueless right index is npos");
		}
		if (!threw) {
			check(ctx, ThrowingType::live == 1, "swap no-throw keeps one ThrowingType alive");
		}
		left = 7;
		right = std::string("ok");
		check(ctx, holds_alternative<int>(left) && get<int>(left) == 7, "swap throw recovery left");
		check(ctx, holds_alternative<std::string>(right) && get<std::string>(right) == "ok", "swap throw recovery right");
	}

	if constexpr (variant_test::is_custom_variant_v<VariantT>) {
		ThrowingType::reset();
		VariantT left(11);
		VariantT right(std::in_place_type<ThrowingType>, ThrowingType{5});
		ThrowingType::throw_on_move = ThrowingType::move_count + 1;
		bool threw_before_destroy = false;
		try {
			left.swap(right);
		} catch (const std::runtime_error&) {
			threw_before_destroy = true;
		}
		if (threw_before_destroy) {
			check(ctx, !left.valueless_by_exception(), "swap throw-before-destroy keeps left engaged");
			check(ctx, !right.valueless_by_exception(), "swap throw-before-destroy keeps right engaged");
			check(ctx, left.index() == 0, "swap throw-before-destroy preserves left index");
			check(ctx, right.index() == 2, "swap throw-before-destroy preserves right index");
		}
	}
}

int main() {
	int nrOfFailedTestCases = 0;
	run_encoded_index_tests("encoded_index", nrOfFailedTestCases);
	run_variant_suite<CustomVariant>("custom_indexed_variant", nrOfFailedTestCases);
	run_variant_suite<SidebandVariant>("custom_indexed_variant_sideband", nrOfFailedTestCases);
	run_variant_suite<std::variant>("std::variant", nrOfFailedTestCases);

	sw::universal::ReportTestResult(nrOfFailedTestCases, "custom_indexed_variant", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
