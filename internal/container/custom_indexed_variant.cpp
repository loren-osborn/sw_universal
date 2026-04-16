// custom_indexed_variant.cpp: unit tests for custom_indexed_variant
//
// Organization:
// - encoded-index tests validate the pluggable index policies directly
// - parity tests compare `custom_indexed_variant` against `std::variant` where semantics are intended
//   to match closely
// - custom-only suites document sideband behavior and the stronger "throw before destroy" cases used by
//   internal consumers such as `sso_vector`
// - lifetime suites use tracked alternatives to verify switching, assignment, and destruction invariants
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <universal/internal/container/custom_indexed_variant.hpp>
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

// Lightweight throw checker used throughout the parity and custom-only suites.
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

// Small probe for leak / double-destroy style checks.
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

// Constructor that can throw before any old alternative is destroyed. Tests use this to distinguish
// "stage new value first" behavior from the simpler destroy-then-construct path.
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

	ThrowBeforeDestroyType(const ThrowBeforeDestroyType& other) : value(other.value) { ++live; }
	ThrowBeforeDestroyType& operator=(const ThrowBeforeDestroyType& other) { value = other.value; return *this; }
	ThrowBeforeDestroyType(ThrowBeforeDestroyType&& other) noexcept : value(other.value) { other.value = 0; ++live; }
	ThrowBeforeDestroyType& operator=(ThrowBeforeDestroyType&& other) noexcept { value = other.value; other.value = 0; return *this; }
	~ThrowBeforeDestroyType() { --live; }
};

int ThrowBeforeDestroyType::live = 0;
int ThrowBeforeDestroyType::ctor_count = 0;
int ThrowBeforeDestroyType::throw_on_ctor = -1;

// General exception probe for std::variant parity and custom-only recovery checks.
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

// Captures what the local standard library actually does for selected throwing transitions so the
// custom variant can compare against behavior instead of assuming one library's interpretation.
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

// Used to verify that `visit` preserves cv/ref category instead of silently decaying everything.
struct VisitCategoryVisitor {
	VisitRefCategory operator()(int&) const { return VisitRefCategory::LValue; }
	VisitRefCategory operator()(const int&) const { return VisitRefCategory::ConstLValue; }
	VisitRefCategory operator()(int&&) const { return VisitRefCategory::RValue; }
	VisitRefCategory operator()(const int&&) const { return VisitRefCategory::ConstRValue; }
};

struct StrictConstRvalueVisitVisitor {
	bool operator()(const int&&) const { return true; }
	bool operator()(const std::string&&) const { return true; }
	template<class T>
	bool operator()(T&&) const = delete;
};

struct OnlyConstRvalueOK {
	bool operator()(const int&&) const { return true; }
	bool operator()(const std::string&&) const { return true; }
	template<class T>
	bool operator()(T&&) const = delete;
};

struct RejectConstRvalue {
	bool operator()(int&&) const { return true; }
	bool operator()(const int&) const { return true; }
	bool operator()(std::string&&) const { return true; }
	bool operator()(const std::string&) const { return true; }
	bool operator()(const int&&) const = delete;
	bool operator()(const std::string&&) const = delete;
};

// Distinct tracked alternative types used to make "same active alternative" and "different active
// alternative" assignment paths visible in the counters.
template<int Tag>
struct LifetimeVariantTracked {
	inline static int live = 0;
	inline static int value_ctor = 0;
	inline static int copy_ctor = 0;
	inline static int move_ctor = 0;
	inline static int copy_assign = 0;
	inline static int move_assign = 0;
	inline static int dtor = 0;
	inline static int next_serial = 0;

	int value = 0;
	int serial = 0;

	static void reset() {
		live = 0;
		value_ctor = 0;
		copy_ctor = 0;
		move_ctor = 0;
		copy_assign = 0;
		move_assign = 0;
		dtor = 0;
		next_serial = 0;
	}

	explicit LifetimeVariantTracked(int v = 0) : value(v), serial(++next_serial) {
		++value_ctor;
		++live;
	}

	LifetimeVariantTracked(const LifetimeVariantTracked& other) : value(other.value), serial(++next_serial) {
		++copy_ctor;
		++live;
	}

	LifetimeVariantTracked(LifetimeVariantTracked&& other) noexcept : value(other.value), serial(++next_serial) {
		other.value = -1;
		++move_ctor;
		++live;
	}

	LifetimeVariantTracked& operator=(const LifetimeVariantTracked& other) {
		value = other.value;
		++copy_assign;
		return *this;
	}

	LifetimeVariantTracked& operator=(LifetimeVariantTracked&& other) noexcept {
		value = other.value;
		other.value = -1;
		++move_assign;
		return *this;
	}

	~LifetimeVariantTracked() {
		++dtor;
		--live;
	}
};

// Throwing tracked alternative used for tests that stage construction before replacing the current value.
struct LifetimeVariantThrowingTracked {
	inline static int live = 0;
	inline static int ctor_count = 0;
	inline static int next_serial = 0;
	inline static int throw_on_ctor = -1;

	int value = 0;
	int serial = 0;

	static void reset() {
		live = 0;
		ctor_count = 0;
		next_serial = 0;
		throw_on_ctor = -1;
	}

	explicit LifetimeVariantThrowingTracked(int v = 0) : value(v), serial(++next_serial) {
		++ctor_count;
		if (throw_on_ctor >= 0 && ctor_count == throw_on_ctor) {
			throw std::runtime_error("LifetimeVariantThrowingTracked configured throw");
		}
		++live;
	}

	LifetimeVariantThrowingTracked(const LifetimeVariantThrowingTracked& other) : value(other.value), serial(++next_serial) {
		++live;
	}

	LifetimeVariantThrowingTracked& operator=(const LifetimeVariantThrowingTracked& other) {
		value = other.value;
		return *this;
	}

	LifetimeVariantThrowingTracked(LifetimeVariantThrowingTracked&& other) noexcept : value(other.value), serial(++next_serial) {
		other.value = -1;
		++live;
	}

	LifetimeVariantThrowingTracked& operator=(LifetimeVariantThrowingTracked&& other) noexcept {
		value = other.value;
		other.value = -1;
		return *this;
	}

	~LifetimeVariantThrowingTracked() {
		--live;
	}
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
	// These adapters let the same parity tests target std::variant and custom_indexed_variant.
	// The test bodies stay focused on observable behavior while the adapter picks the right API surface.
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
		requires ((is_custom_variant_v<Variants> || ...)) &&
			requires {
				sw::universal::internal::visit(std::declval<Visitor>(), std::declval<Variants>()...);
			}
	decltype(auto) visit_adapter(Visitor&& vis, Variants&&... variants) {
		return sw::universal::internal::visit(std::forward<Visitor>(vis), std::forward<Variants>(variants)...);
	}

	template<typename Visitor, typename... Variants>
		requires (!(is_custom_variant_v<Variants> || ...)) &&
			requires {
				std::visit(std::declval<Visitor>(), std::declval<Variants>()...);
			}
	decltype(auto) visit_adapter(Visitor&& vis, Variants&&... variants) {
		return std::visit(std::forward<Visitor>(vis), std::forward<Variants>(variants)...);
	}
}

struct VariantStateSummary {
	bool valueless = false;
	std::size_t index = std::variant_npos;
	std::string held;
};

// Summarize the observable variant state so parity tests can compare std/custom behavior directly
// without depending on implementation details.
template<typename Variant>
VariantStateSummary summarize_variant_state(const Variant& v) {
	using namespace variant_test;
	VariantStateSummary out{};
	out.valueless = v.valueless_by_exception();
	out.index = v.index();
	if (out.valueless) {
		out.held = "valueless";
		return out;
	}
	out.held = visit_adapter([](const auto& value) -> std::string {
		using Value = std::decay_t<decltype(value)>;
		if constexpr (std::is_same_v<Value, int>) {
			return "int:" + std::to_string(value);
		} else if constexpr (std::is_same_v<Value, std::string>) {
			return "string:" + value;
		} else if constexpr (std::is_same_v<Value, ThrowingType>) {
			return "throwing:" + std::to_string(value.value);
		} else {
			return "live:" + std::to_string(value.value);
		}
	}, v);
	return out;
}

template<typename Left, typename Right>
void check_same_variant_state(const TestContext& ctx, const Left& left, const Right& right, const char* label) {
	const auto lhs = summarize_variant_state(left);
	const auto rhs = summarize_variant_state(right);
	check(ctx, lhs.valueless == rhs.valueless, label);
	check(ctx, lhs.index == rhs.index, label);
	check(ctx, lhs.held == rhs.held, label);
}

} // namespace

template<class... Ts>
using CustomVariant = sw::universal::internal::custom_indexed_variant<sw::universal::internal::simple_encoded_index, Ts...>;

template<class... Ts>
using SidebandVariant = sw::universal::internal::custom_indexed_variant<sw::universal::internal::sideband_encoded_index, Ts...>;

template<typename V>
constexpr bool variant_has_sideband() {
	if constexpr (requires(V& v) { v.sideband(); }) {
		return true;
	} else {
		return false;
	}
}

template<typename V>
constexpr bool custom_visit_const_rvalue_is_strictly_forwarded() {
	using namespace variant_test;
	return requires {
		visit_adapter(StrictConstRvalueVisitVisitor{}, std::declval<const V&&>());
	};
}

template<typename V>
constexpr bool custom_visit_const_rvalue_accepts_only_const_rvalue_visitor() {
	return requires {
		variant_test::visit_adapter(OnlyConstRvalueOK{}, std::declval<const V&&>());
	};
}

template<typename V>
constexpr bool custom_visit_const_rvalue_rejects_wrong_overloads() {
	return !requires {
		variant_test::visit_adapter(RejectConstRvalue{}, std::declval<const V&&>());
	};
}

template<template<class...> class Variant, typename Source, typename... Ts>
constexpr bool construct_assign_traits_match_std_variant() {
	using V = Variant<Ts...>;
	using SV = std::variant<Ts...>;
	return (std::is_constructible_v<V, Source> == std::is_constructible_v<SV, Source>) &&
		(std::is_assignable_v<V&, Source> == std::is_assignable_v<SV&, Source>) &&
		(std::is_convertible_v<Source, V> == std::is_convertible_v<Source, SV>);
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
		using V = Variant<int>;
		static_assert(!std::is_constructible_v<V, double>);
		static_assert(!std::is_assignable_v<V&, double>);
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
	{
		using V = Variant<ExplicitOnlyInt>;
		static_assert(!std::is_convertible_v<int, V>);
	}
	static_assert(construct_assign_traits_match_std_variant<Variant, int, int>());
	static_assert(construct_assign_traits_match_std_variant<Variant, double, int>());
	static_assert(construct_assign_traits_match_std_variant<Variant, double, short, int>());
	static_assert(construct_assign_traits_match_std_variant<Variant, long long, short, int>());
	return true;
}

void report_std_variant_explicit_only_conversion_sentinel() {
	// Sentinel only: standard libraries have disagreed on explicit-only converting variant(T)/operator=(T).
	// If these values change, stdlib behavior likely changed/converged; consider aligning custom_indexed_variant.
#if defined(_LIBCPP_VERSION)
	std::cout << "[sentinel] stdlib/libc++ _LIBCPP_VERSION=" << _LIBCPP_VERSION;
#elif defined(__GLIBCXX__)
	std::cout << "[sentinel] stdlib/libstdc++ __GLIBCXX__=" << __GLIBCXX__;
#else
	std::cout << "[sentinel] stdlib=unknown";
#endif
#if defined(__clang__)
	std::cout << " compiler=clang-" << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
	std::cout << " compiler=gcc-" << __GNUC__ << "." << __GNUC_MINOR__;
#else
	std::cout << " compiler=unknown";
#endif
	std::cout << "\n";
	using SV1 = std::variant<ExplicitOnlyInt>;
	using SV2 = std::variant<ExplicitOnlyInt, std::string>;
	std::cout
		<< "[sentinel] std::variant explicit-only from int: "
		<< "SV1{C=" << std::is_constructible_v<SV1, int>
		<< ",A=" << std::is_assignable_v<SV1&, int>
		<< ",V=" << std::is_convertible_v<int, SV1>
		<< "} "
		<< "SV2{C=" << std::is_constructible_v<SV2, int>
		<< ",A=" << std::is_assignable_v<SV2&, int>
		<< ",V=" << std::is_convertible_v<int, SV2>
		<< "}\n";
}

void report_custom_variant_explicit_only_conversion_sentinel() {
	using CV1 = CustomVariant<ExplicitOnlyInt>;
	using CV2 = CustomVariant<ExplicitOnlyInt, std::string>;
	using SB1 = SidebandVariant<ExplicitOnlyInt>;
	using SB2 = SidebandVariant<ExplicitOnlyInt, std::string>;
	std::cout
		<< "[sentinel] custom baseline explicit-only from int: "
		<< "CV1{C=" << std::is_constructible_v<CV1, int>
		<< ",A=" << std::is_assignable_v<CV1&, int>
		<< ",V=" << std::is_convertible_v<int, CV1>
		<< "} "
		<< "CV2{C=" << std::is_constructible_v<CV2, int>
		<< ",A=" << std::is_assignable_v<CV2&, int>
		<< ",V=" << std::is_convertible_v<int, CV2>
		<< "} "
		<< "SB1{C=" << std::is_constructible_v<SB1, int>
		<< ",A=" << std::is_assignable_v<SB1&, int>
		<< ",V=" << std::is_convertible_v<int, SB1>
		<< "} "
		<< "SB2{C=" << std::is_constructible_v<SB2, int>
		<< ",A=" << std::is_assignable_v<SB2&, int>
		<< ",V=" << std::is_convertible_v<int, SB2>
		<< "}\n";
}

template<typename V>
constexpr bool const_sideband_readable() {
	return requires(const V& v) { v.sideband().get(); };
}

template<typename V>
constexpr bool const_sideband_writable() {
	return requires(const V& v) { v.sideband().set(1); };
}

template<typename V>
constexpr bool mutable_sideband_writable() {
	return requires(V& v) { v.sideband().set(v.sideband().get()); };
}

template<typename V>
using sideband_accessor_t = decltype(std::declval<V&>().sideband());

template<typename V>
using const_sideband_accessor_t = decltype(std::declval<const V&>().sideband());

template<typename V>
constexpr bool sideband_accessor_assignable() {
	return requires(sideband_accessor_t<V> a, sideband_accessor_t<V> b) {
		a = b;
	};
}

template<typename V>
constexpr bool mutable_sideband_assignable_from_const_accessor() {
	return requires(sideband_accessor_t<V> a, const_sideband_accessor_t<V> b) {
		a = b;
	};
}

struct AccessorByValue {
	int get() const noexcept { return 0; }
	void set(int) noexcept {}
};

struct AccessorByConstRef {
	int value = 0;
	const int& get() const noexcept { return value; }
	void set(const int& next) noexcept { value = next; }
};

struct AccessorGetterByRef {
	int value = 0;
	int& get() noexcept { return value; }
	void set(int) noexcept {}
};

struct AccessorGetterVolatile {
	volatile int value = 0;
	const volatile int& get() const noexcept { return value; }
	void set(int) noexcept {}
};

struct AccessorSetterByRef {
	int get() const noexcept { return 0; }
	void set(int&) noexcept {}
};

struct AccessorSetterVolatileRef {
	int get() const noexcept { return 0; }
	void set(const volatile int&) noexcept {}
};

struct AccessorSetterMismatched {
	int get() const noexcept { return 0; }
	void set(long) noexcept {}
};

struct AccessorSetterOverloaded {
	int get() const noexcept { return 0; }
	void set(int) noexcept {}
	void set(const int&) noexcept {}
};

struct AccessorSetterTemplated {
	int get() const noexcept { return 0; }
	template<typename U>
	void set(U&&) noexcept {}
};

template<std::size_t>
struct SidebandIndexWithoutHooks {
	int sideband_state = 0;

	struct accessor {
		SidebandIndexWithoutHooks* owner;
		int get() const noexcept { return owner->sideband_state; }
		void set(int next) noexcept { owner->sideband_state = next; }
	};

	struct const_accessor {
		const SidebandIndexWithoutHooks* owner;
		int get() const noexcept { return owner->sideband_state; }
	};

	std::size_t index() const noexcept { return std::variant_npos; }
	void set_index(std::size_t) noexcept {}
	accessor sideband() noexcept { return accessor{this}; }
	const_accessor sideband() const noexcept { return const_accessor{this}; }
};

template<std::size_t>
struct SidebandIndexMissingSwap {
	int sideband_state = 0;

	struct accessor {
		SidebandIndexMissingSwap* owner;
		int get() const noexcept { return owner->sideband_state; }
		void set(int next) noexcept { owner->sideband_state = next; }
	};

	struct const_accessor {
		const SidebandIndexMissingSwap* owner;
		int get() const noexcept { return owner->sideband_state; }
	};

	std::size_t index() const noexcept { return std::variant_npos; }
	void set_index(std::size_t) noexcept {}
	void copy_sideband_from(const SidebandIndexMissingSwap& other) noexcept { sideband_state = other.sideband_state; }
	accessor sideband() noexcept { return accessor{this}; }
	const_accessor sideband() const noexcept { return const_accessor{this}; }
};

template<std::size_t>
struct NonNoexceptEncodedIndex {
	std::size_t index() const { return std::variant_npos; }
	void set_index(std::size_t) {}
};

struct NothrowMoveCtorThrowingMoveAssign {
	int value = 0;
	NothrowMoveCtorThrowingMoveAssign() = default;
	explicit NothrowMoveCtorThrowingMoveAssign(int v) noexcept : value(v) {}
	NothrowMoveCtorThrowingMoveAssign(const NothrowMoveCtorThrowingMoveAssign&) = default;
	NothrowMoveCtorThrowingMoveAssign(NothrowMoveCtorThrowingMoveAssign&&) noexcept = default;
	NothrowMoveCtorThrowingMoveAssign& operator=(const NothrowMoveCtorThrowingMoveAssign&) = default;
	NothrowMoveCtorThrowingMoveAssign& operator=(NothrowMoveCtorThrowingMoveAssign&& other) noexcept(false) {
		value = other.value;
		return *this;
	}
};

struct NonCopyConstructibleAlt {
	NonCopyConstructibleAlt() = default;
	NonCopyConstructibleAlt(const NonCopyConstructibleAlt&) = delete;
	NonCopyConstructibleAlt(NonCopyConstructibleAlt&&) = default;
	NonCopyConstructibleAlt& operator=(const NonCopyConstructibleAlt&) = default;
	NonCopyConstructibleAlt& operator=(NonCopyConstructibleAlt&&) = default;
};

struct NotAssignableFromConstRefAlt {
	NotAssignableFromConstRefAlt() = default;
	NotAssignableFromConstRefAlt(const NotAssignableFromConstRefAlt&) = default;
	NotAssignableFromConstRefAlt(NotAssignableFromConstRefAlt&&) = default;
	NotAssignableFromConstRefAlt& operator=(const NotAssignableFromConstRefAlt&) = delete;
	NotAssignableFromConstRefAlt& operator=(NotAssignableFromConstRefAlt&&) = default;
};

static_assert(sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	sw::universal::internal::simple_encoded_index<4>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	sw::universal::internal::sideband_encoded_index<4>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	NonNoexceptEncodedIndex<4>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_sideband_accessor_v<
	sw::universal::internal::simple_encoded_index<1>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::has_sideband_accessor_v<
	sw::universal::internal::sideband_encoded_index<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_copy_sideband<
	sw::universal::internal::simple_encoded_index<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_swap_sideband<
	sw::universal::internal::simple_encoded_index<1>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::has_copy_sideband<
	sw::universal::internal::sideband_encoded_index<1>>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::has_swap_sideband<
	sw::universal::internal::sideband_encoded_index<1>>);
static_assert(!variant_has_sideband<CustomVariant<int>>());
static_assert(variant_has_sideband<SidebandVariant<int>>());
static_assert(!variant_has_sideband<std::variant<int>>());
static_assert(run_converting_selection_static_checks_common<CustomVariant>());
static_assert(run_converting_selection_static_checks_common<SidebandVariant>());
static_assert(run_converting_selection_static_checks_common<std::variant>());
static_assert(sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorByValue>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorByConstRef>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorGetterByRef>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorGetterVolatile>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorSetterByRef>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorSetterVolatileRef>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorSetterMismatched>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorSetterOverloaded>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<AccessorSetterTemplated>);
static_assert(const_sideband_readable<SidebandVariant<int>>());
static_assert(!const_sideband_writable<SidebandVariant<int>>());
static_assert(mutable_sideband_writable<SidebandVariant<int>>());
static_assert(sideband_accessor_assignable<SidebandVariant<int>>());
static_assert(mutable_sideband_assignable_from_const_accessor<SidebandVariant<int>>());
static_assert(sw::universal::internal::custom_indexed_variant_detail::sideband_accessor<
	decltype(std::declval<SidebandVariant<int>&>().sideband())>);
static_assert(sw::universal::internal::custom_indexed_variant_detail::has_sideband_accessor<
	SidebandIndexWithoutHooks<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_copy_sideband<
	SidebandIndexWithoutHooks<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::has_swap_sideband<
	SidebandIndexWithoutHooks<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_encoded_index_api<
	SidebandIndexWithoutHooks<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	SidebandIndexWithoutHooks<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::sideband_encoded_index_api<
	SidebandIndexMissingSwap<1>>);
static_assert(!sw::universal::internal::custom_indexed_variant_detail::encoded_index_noexcept_api<
	SidebandIndexMissingSwap<1>>);
static_assert(custom_visit_const_rvalue_is_strictly_forwarded<CustomVariant<int, std::string>>());
static_assert(custom_visit_const_rvalue_is_strictly_forwarded<SidebandVariant<int, std::string>>());
static_assert(custom_visit_const_rvalue_accepts_only_const_rvalue_visitor<CustomVariant<int, std::string>>());
static_assert(custom_visit_const_rvalue_accepts_only_const_rvalue_visitor<SidebandVariant<int, std::string>>());
static_assert(custom_visit_const_rvalue_rejects_wrong_overloads<CustomVariant<int, std::string>>());
static_assert(custom_visit_const_rvalue_rejects_wrong_overloads<SidebandVariant<int, std::string>>());
static_assert(!noexcept(std::declval<CustomVariant<NothrowMoveCtorThrowingMoveAssign>&>() =
                        std::declval<CustomVariant<NothrowMoveCtorThrowingMoveAssign>&&>()),
	"custom_indexed_variant move assignment must account for same-alternative move assignment");
static_assert(sw::universal::internal::custom_indexed_variant_detail::custom_indexed_variant_alternative_requirements<int, std::string>,
	"custom_indexed_variant should accept ordinary copyable/assignable alternatives");
static_assert(!sw::universal::internal::custom_indexed_variant_detail::custom_indexed_variant_alternative_requirements<int, NonCopyConstructibleAlt>,
	"custom_indexed_variant should reject non-copy-constructible alternatives");
static_assert(!sw::universal::internal::custom_indexed_variant_detail::custom_indexed_variant_alternative_requirements<int, NotAssignableFromConstRefAlt>,
	"custom_indexed_variant should reject alternatives not assignable from const same-type reference");
static_assert(sw::universal::internal::custom_indexed_variant_detail::custom_indexed_variant_alternative_requirements<int, std::string>,
	"custom_indexed_variant requirements should accept ordinary copyable/assignable alternatives");

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
			// Simple encoded-index policy: only the active index / valueless sentinel.
			sw::universal::internal::simple_encoded_index<3> index{};
			check(ctx, index.index() == std::variant_npos, "simple_encoded_index default index is std::variant_npos");
		index.set_index(2);
		check(ctx, index.index() == 2, "simple_encoded_index set/get normal value");
		index.set_index(std::variant_npos);
		check(ctx, index.index() == std::variant_npos, "simple_encoded_index variant_npos round trip");
	}

	{
		// Sideband-carrying encoded index: index updates must preserve the sideband payload and vice versa.
		// This is the property `sso_vector` depends on when it stores size in the sideband while the
		// alternative index tracks inline-vs-heap representation.
		using Index = sw::universal::internal::sideband_encoded_index<3>;
		Index index{};
		check(ctx, index.index() == std::variant_npos, "sideband_encoded_index default index is std::variant_npos");
		check(ctx, static_cast<std::size_t>(index.sideband().get()) == 0, "sideband_encoded_index default sideband is 0");
		index.set_index(1);
		check(ctx, index.index() == 1, "sideband_encoded_index stores index");
		index.set_index(2);
		index.sideband().set(5);
		check(ctx, index.index() == 2, "sideband_encoded_index sideband write preserves index");
		check(ctx, static_cast<std::size_t>(index.sideband().get()) == 5, "sideband_encoded_index sideband round trip");
		index.set_index(std::variant_npos);
		check(ctx, index.index() == std::variant_npos, "sideband_encoded_index npos round trip");
		check(ctx, static_cast<std::size_t>(index.sideband().get()) == 5, "sideband_encoded_index npos preserves sideband");
		index.set_index(1);
		check(ctx, static_cast<std::size_t>(index.sideband().get()) == 5, "sideband_encoded_index index write preserves sideband");
		index.sideband().set(3);
		check(ctx, index.index() == 1, "sideband_encoded_index sideband write preserves index");

		auto accessor_a = index.sideband();
		auto accessor_b = index.sideband();
		accessor_a.set(7);
		check(ctx, static_cast<std::size_t>(accessor_b.get()) == 7, "sideband accessor coherence A->B");
		accessor_b.set(2);
		check(ctx, static_cast<std::size_t>(accessor_a.get()) == 2, "sideband accessor coherence B->A");
		const Index& cindex = index;
		accessor_a = cindex.sideband();
		check(ctx, static_cast<std::size_t>(accessor_a.get()) == 2, "sideband accessor assignment copies exposed state");
	}

	{
		// Repeat at a different arity so the bit-width computation is exercised again.
		using Index = sw::universal::internal::sideband_encoded_index<7>;
		Index index{};
		check(ctx, index.index() == std::variant_npos, "sideband_encoded_index(7) default index is std::variant_npos");
		index.set_index(6);
		check(ctx, index.index() == 6, "sideband_encoded_index(7) stores index");
		index.sideband().set(12);
		check(ctx, static_cast<std::size_t>(index.sideband().get()) == 12, "sideband_encoded_index(7) sideband round trip");
	}
}

void run_variant_std_parity_tests(int& failures) {
	TestContext ctx{"custom_indexed_variant(parity)", failures};
	using namespace variant_test;
	using Std = std::variant<int, std::string, ThrowingType, LiveCountedType>;
	using Custom = CustomVariant<int, std::string, ThrowingType, LiveCountedType>;

	{
		// Default construction should match std::variant's first-alternative semantics.
		Std sv;
		Custom cv;
		check_same_variant_state(ctx, cv, sv, "default construction parity");
		check(ctx, get<0>(cv) == get<0>(sv), "default get by index parity");
		check(ctx, get<int>(cv) == get<int>(sv), "default get by type parity");
		check(ctx, holds_alternative<int>(cv) == std::holds_alternative<int>(sv), "default holds_alternative parity");
	}

	{
		// In-place construction by both index and type should agree with std::variant.
		Std sv(std::in_place_index<1>, "hello");
		Custom cv(std::in_place_index<1>, "hello");
		check_same_variant_state(ctx, cv, sv, "in_place_index parity");
		check(ctx, get<1>(cv) == std::get<1>(sv), "in_place_index get parity");

		Std st(std::in_place_type<LiveCountedType>, LiveCountedType{7});
		Custom ct(std::in_place_type<LiveCountedType>, LiveCountedType{7});
		check_same_variant_state(ctx, ct, st, "in_place_type parity");
		check(ctx, get<LiveCountedType>(ct).value == std::get<LiveCountedType>(st).value, "in_place_type get parity");
	}

	{
		// Copy/move constructors should preserve the same observable active state.
		Std sv(std::in_place_type<std::string>, "copy");
		Custom cv(std::in_place_type<std::string>, "copy");
		Std sv_copy(sv);
		Custom cv_copy(cv);
		check_same_variant_state(ctx, cv_copy, sv_copy, "copy construction parity");

		Std sv_move(std::move(sv_copy));
		Custom cv_move(std::move(cv_copy));
		check_same_variant_state(ctx, cv_move, sv_move, "move construction parity");
	}

	{
		// Assignments across alternatives are one of the easiest places for refactors to drift.
		Std sv(3);
		Custom cv(3);
		sv = std::string("assign");
		cv = std::string("assign");
		check_same_variant_state(ctx, cv, sv, "assignment to string parity");

		sv = LiveCountedType{9};
		cv = LiveCountedType{9};
		check_same_variant_state(ctx, cv, sv, "assignment to LiveCountedType parity");

		Std source_sv(77);
		Custom source_cv(77);
		sv = source_sv;
		cv = source_cv;
		check_same_variant_state(ctx, cv, sv, "copy assignment parity");

		Std move_source_sv(std::in_place_type<std::string>, "moved");
		Custom move_source_cv(std::in_place_type<std::string>, "moved");
		sv = std::move(move_source_sv);
		cv = std::move(move_source_cv);
		check_same_variant_state(ctx, cv, sv, "move assignment parity");
	}

	{
		// `emplace` is checked separately because it exercises destroy/reconstruct transitions.
		// The first assignment here is the "same alternative, reuse object" case; the later emplaces are
		// the "destroy current alternative and construct a new one" cases.
		Std sv(5);
		Custom cv(5);
		std::get<0>(sv) = 6;
		get<0>(cv) = 6;
		check_same_variant_state(ctx, cv, sv, "same-alternative assignment parity");

		sv.emplace<1>("emplaced");
		cv.emplace<1>("emplaced");
		check_same_variant_state(ctx, cv, sv, "emplace by index parity");

		sv.emplace<LiveCountedType>(LiveCountedType{33});
		cv.emplace<LiveCountedType>(LiveCountedType{33});
		check_same_variant_state(ctx, cv, sv, "emplace by type parity");
	}

	{
		// Accessor parity covers const, non-const, pointer, and rvalue paths.
		Std sv(std::in_place_type<std::string>, "access");
		Custom cv(std::in_place_type<std::string>, "access");
		const Std& csv = sv;
		const Custom& ccv = cv;

		check(ctx, std::get<1>(sv) == get<1>(cv), "non-const get by index parity");
		check(ctx, std::get<std::string>(sv) == get<std::string>(cv), "non-const get by type parity");
		check(ctx, std::get<1>(csv) == get<1>(ccv), "const get by index parity");
		check(ctx, std::get<std::string>(csv) == get<std::string>(ccv), "const get by type parity");

		auto* sp = std::get_if<1>(&sv);
		auto* cp = get_if<1>(&cv);
		check(ctx, (sp != nullptr) == (cp != nullptr), "get_if by index engagement parity");
		check(ctx, cp && *cp == *sp, "get_if by index value parity");

		auto* stp = std::get_if<std::string>(&sv);
		auto* ctp = get_if<std::string>(&cv);
		check(ctx, (stp != nullptr) == (ctp != nullptr), "get_if by type engagement parity");
		check(ctx, ctp && *ctp == *stp, "get_if by type value parity");

		check(ctx, static_cast<std::string>(get<1>(std::move(cv))) == std::get<1>(std::move(sv)), "rvalue get by index parity");
	}

	{
		// Swapping same-index and different-index alternatives exercises the active-state machine.
		// Same-index swap should reuse live objects; different-index swap must exchange lifetimes.
		Std left_sv(10);
		Custom left_cv(10);
		Std right_sv(20);
		Custom right_cv(20);
		left_sv.swap(right_sv);
		left_cv.swap(right_cv);
		check_same_variant_state(ctx, left_cv, left_sv, "swap same-index left parity");
		check_same_variant_state(ctx, right_cv, right_sv, "swap same-index right parity");

		Std diff_left_sv(11);
		Custom diff_left_cv(11);
		Std diff_right_sv(std::in_place_type<std::string>, "right");
		Custom diff_right_cv(std::in_place_type<std::string>, "right");
		diff_left_sv.swap(diff_right_sv);
		diff_left_cv.swap(diff_right_cv);
		check_same_variant_state(ctx, diff_left_cv, diff_left_sv, "swap different-index left parity");
		check_same_variant_state(ctx, diff_right_cv, diff_right_sv, "swap different-index right parity");
	}

	{
		// This scenario deliberately churns through different alternatives to compare index transitions.
		Std sv(1);
		Custom cv(1);
		std::array<int, 4> seen_std{};
		std::array<int, 4> seen_custom{};
		for (int i = 0; i < 4; ++i) {
			switch (i) {
			case 0:
				sv = std::string("seq");
				cv = std::string("seq");
				break;
			case 1:
				sv = ThrowingType{44};
				cv = ThrowingType{44};
				break;
			case 2:
				sv.emplace<LiveCountedType>(LiveCountedType{55});
				cv.emplace<LiveCountedType>(LiveCountedType{55});
				break;
			default:
				sv = 99;
				cv = 99;
				break;
			}
			seen_std[i] = static_cast<int>(sv.index());
			seen_custom[i] = static_cast<int>(cv.index());
			check_same_variant_state(ctx, cv, sv, "transition parity");
		}
		check(ctx, seen_custom == seen_std, "transition index sequence parity");
	}

	{
		// Invalid-access behavior should surface as std::bad_variant_access in both implementations.
		Std sv(std::in_place_type<std::string>, "bad");
		Custom cv(std::in_place_type<std::string>, "bad");
		expect_throw<std::bad_variant_access>(ctx, "bad get by index parity std", [&]() {
			(void)std::get<0>(sv);
		});
		expect_throw<std::bad_variant_access>(ctx, "bad get by index parity custom", [&]() {
			(void)get<0>(cv);
		});
		expect_throw<std::bad_variant_access>(ctx, "bad get by type parity std", [&]() {
			(void)std::get<int>(sv);
		});
		expect_throw<std::bad_variant_access>(ctx, "bad get by type parity custom", [&]() {
			(void)get<int>(cv);
		});
	}

	{
		// Visiting a valueless variant is another parity boundary worth making explicit for reviewers.
		ThrowingType::reset();
		Std sv(123);
		Custom cv(123);
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "force std valueless for visit parity", [&]() {
			sv.template emplace<2>();
		});
		ThrowingType::reset();
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "force custom valueless for visit parity", [&]() {
			cv.template emplace<2>();
		});
		if (sv.valueless_by_exception() && cv.valueless_by_exception()) {
			expect_throw<std::bad_variant_access>(ctx, "std visit on valueless throws", [&]() {
				(void)std::visit([](const auto&) { return 0; }, sv);
			});
			expect_throw<std::bad_variant_access>(ctx, "custom visit on valueless throws", [&]() {
				(void)visit_adapter([](const auto&) { return 0; }, cv);
			});
		}
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
		// Exact-match conversion is an important documented choice: the exact type should win before
		// broader implicit conversion ranking is considered.
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
		using ArithmeticVariant = Variant<int, double>;
		using StdArithmeticVariant = std::variant<int, double>;
		ArithmeticVariant v = 3.25;
		StdArithmeticVariant sv = 3.25;
		check(ctx, v.index() == sv.index(), "non-narrowing converting ctor index matches std::variant");
		v = 8.5;
		sv = 8.5;
		check(ctx, v.index() == sv.index(), "non-narrowing converting assignment index matches std::variant");
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
		// Custom-only behavior: sideband is part of the variant's logical value, but same-object payload
		// transitions preserve the current object's sideband unless they are importing state from another variant.
		VariantT v(std::in_place_type<std::string>, "sb");
		auto sa = v.sideband();
		auto sb = v.sideband();
		sa.set(9);
		check(ctx, static_cast<std::size_t>(sb.get()) == 9, "sideband accessors are coherent");
		check(ctx, v.index() == 1, "sideband write preserves active index");
		v.template emplace<0>(42);
		check(ctx, static_cast<std::size_t>(v.sideband().get()) == 9, "same-object emplace preserves sideband");
		check(ctx, v.index() == 0, "emplace updates active index with sideband present");
		v = std::string("again");
		check(ctx, static_cast<std::size_t>(v.sideband().get()) == 9, "same-object converting assignment preserves sideband");
		const VariantT& cv = v;
		auto csb = cv.sideband();
		check(ctx, static_cast<std::size_t>(csb.get()) == 9, "const sideband accessor reads sideband");
	}

	if constexpr (requires(VariantT& v) { v.sideband(); }) {
		VariantT src(std::in_place_type<std::string>, "copy");
		src.sideband().set(13);
		VariantT copy(src);
		check(ctx, static_cast<std::size_t>(copy.sideband().get()) == 13, "copy construction copies sideband");

		VariantT moved(std::move(src));
		check(ctx, static_cast<std::size_t>(moved.sideband().get()) == 13, "move construction transfers sideband into destination");

		VariantT copy_assign_dst(1);
		copy_assign_dst.sideband().set(2);
		copy_assign_dst = copy;
		check(ctx, static_cast<std::size_t>(copy_assign_dst.sideband().get()) == 13, "copy assignment imports sideband");

		VariantT move_assign_src(std::in_place_type<std::string>, "move");
		move_assign_src.sideband().set(17);
		VariantT move_assign_dst(3);
		move_assign_dst.sideband().set(4);
		move_assign_dst = std::move(move_assign_src);
		check(ctx, static_cast<std::size_t>(move_assign_dst.sideband().get()) == 17, "move assignment imports sideband");

		VariantT same_left(10);
		VariantT same_right(20);
		same_left.sideband().set(21);
		same_right.sideband().set(22);
		same_left.swap(same_right);
		check(ctx, static_cast<std::size_t>(same_left.sideband().get()) == 22, "same-index swap swaps sideband");
		check(ctx, static_cast<std::size_t>(same_right.sideband().get()) == 21, "same-index swap swaps sideband both directions");

		VariantT diff_left(30);
		VariantT diff_right(std::in_place_type<std::string>, "rhs");
		diff_left.sideband().set(31);
		diff_right.sideband().set(32);
		diff_left.swap(diff_right);
		check(ctx, static_cast<std::size_t>(diff_left.sideband().get()) == 32, "different-index swap swaps sideband");
		check(ctx, static_cast<std::size_t>(diff_right.sideband().get()) == 31, "different-index swap swaps sideband both directions");

		VariantT valueless_src(5);
		valueless_src.sideband().set(44);
		ThrowingType::reset();
		ThrowingType::throw_on_default = 1;
		expect_throw<std::runtime_error>(ctx, "force sideband source valueless", [&]() {
			valueless_src.template emplace<ThrowingType>();
		});
		if (valueless_src.valueless_by_exception()) {
			VariantT valueless_copy(valueless_src);
			check(ctx, valueless_copy.valueless_by_exception(), "copy construction preserves valueless state with sideband policy");
			check(ctx, static_cast<std::size_t>(valueless_copy.sideband().get()) == 44, "copy construction preserves valueless sideband state");

			VariantT valueless_moved(std::move(valueless_src));
			check(ctx, valueless_moved.valueless_by_exception(), "move construction preserves valueless state with sideband policy");
			check(ctx, static_cast<std::size_t>(valueless_moved.sideband().get()) == 44, "move construction preserves valueless sideband state");

			VariantT valueless_dst(std::in_place_type<std::string>, "dst");
			valueless_dst.sideband().set(9);
			valueless_dst = valueless_copy;
			check(ctx, valueless_dst.valueless_by_exception(), "assign from valueless sideband source yields valueless destination");
			check(ctx, static_cast<std::size_t>(valueless_dst.sideband().get()) == 44, "assign from valueless source imports sideband");
		}
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
		// `visit` tests are primarily about forwarding discipline, not visitor business logic.
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
		// This mirrors std::variant's library-specific behavior for throwing emplace when replacing the
		// active alternative. The test records whether the old state is preserved or the variant becomes valueless.
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

	{
		// Reusing the same raw variant storage for different active alternatives must keep get_if/get
		// aligned with the current lifetime and must not leak the replaced object.
		LiveCountedType::reset();
		{
			VariantT v(std::in_place_type<LiveCountedType>, LiveCountedType{21});
			check(ctx, get_if<LiveCountedType>(&v) != nullptr, "storage reuse initial get_if finds active object");
			check(ctx, get_if<std::string>(&v) == nullptr, "storage reuse initial inactive get_if is null");
			v.template emplace<std::string>("reused");
			check(ctx, get_if<LiveCountedType>(&v) == nullptr, "storage reuse get_if drops replaced object");
			check(ctx, get_if<std::string>(&v) != nullptr, "storage reuse get_if finds replacement object");
			check(ctx, get<std::string>(v) == "reused", "storage reuse get reads replacement object");
			v.template emplace<LiveCountedType>(LiveCountedType{22});
			check(ctx, get_if<std::string>(&v) == nullptr, "storage reuse second replacement clears prior alternative");
			check(ctx, get_if<LiveCountedType>(&v) != nullptr, "storage reuse second replacement restores typed access");
			check(ctx, get<LiveCountedType>(v).value == 22, "storage reuse second replacement stores new value");
			check(ctx, LiveCountedType::live == 1, "storage reuse keeps exactly one live tracked alternative");
		}
		check(ctx, LiveCountedType::live == 0, "storage reuse final live count returns to zero");
	}

	if constexpr (variant_test::is_custom_variant_v<VariantT>) {
		// Custom-only behavior: construction failure before destruction should preserve the old state.
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

	if constexpr (requires(VariantT& v) { v.sideband(); }) {
		ThrowingType::reset();
		VariantT same_assign_src(std::in_place_type<ThrowingType>, ThrowingType{7});
		VariantT same_assign_dst(std::in_place_type<ThrowingType>, ThrowingType{8});
		same_assign_src.sideband().set(51);
		same_assign_dst.sideband().set(52);
		const auto same_assign_dst_index_before = same_assign_dst.index();
		const auto same_assign_dst_value_before = get<ThrowingType>(same_assign_dst).value;
		ThrowingType::throw_on_copy_assign = 1;
		expect_throw<std::runtime_error>(ctx, "same-index copy assignment sideband publication waits for success", [&]() {
			same_assign_dst = same_assign_src;
		});
		check(ctx, !same_assign_dst.valueless_by_exception(), "failed same-index copy assignment keeps destination engaged");
		check(ctx, same_assign_dst.index() == same_assign_dst_index_before, "failed same-index copy assignment keeps destination discriminator");
		check(ctx, get<ThrowingType>(same_assign_dst).value == same_assign_dst_value_before, "failed same-index copy assignment keeps destination payload");
		check(ctx, static_cast<std::size_t>(same_assign_dst.sideband().get()) == 52, "failed same-index copy assignment does not import sideband early");

		ThrowingType::reset();
		VariantT move_same_src(std::in_place_type<ThrowingType>, ThrowingType{9});
		VariantT move_same_dst(std::in_place_type<ThrowingType>, ThrowingType{10});
		move_same_src.sideband().set(61);
		move_same_dst.sideband().set(62);
		const auto move_same_dst_index_before = move_same_dst.index();
		const auto move_same_dst_value_before = get<ThrowingType>(move_same_dst).value;
		ThrowingType::throw_on_move_assign = 1;
		expect_throw<std::runtime_error>(ctx, "same-index move assignment sideband publication waits for success", [&]() {
			move_same_dst = std::move(move_same_src);
		});
		check(ctx, !move_same_dst.valueless_by_exception(), "failed same-index move assignment keeps destination engaged");
		check(ctx, move_same_dst.index() == move_same_dst_index_before, "failed same-index move assignment keeps destination discriminator");
		check(ctx, get<ThrowingType>(move_same_dst).value == move_same_dst_value_before, "failed same-index move assignment keeps destination payload");
		check(ctx, static_cast<std::size_t>(move_same_dst.sideband().get()) == 62, "failed same-index move assignment does not import sideband early");

		ThrowingType::reset();
		VariantT diff_assign_src(std::in_place_type<ThrowingType>, ThrowingType{11});
		VariantT diff_assign_dst(12);
		diff_assign_src.sideband().set(71);
		diff_assign_dst.sideband().set(72);
		ThrowingType::throw_on_copy = 1;
		// Different-index assignment destroys the old alternative before attempting to construct the
		// new one, so this failure path intentionally exercises valueless-by-exception with sideband
		// publication still deferred.
		expect_throw<std::runtime_error>(ctx, "different-index copy assignment sideband publication waits for success", [&]() {
			diff_assign_dst = diff_assign_src;
		});
		check(ctx, diff_assign_dst.valueless_by_exception(), "failed different-index copy assignment becomes valueless");
		check(ctx, diff_assign_dst.index() == std::variant_npos, "failed different-index copy assignment publishes npos");
		check(ctx, static_cast<std::size_t>(diff_assign_dst.sideband().get()) == 72, "failed different-index copy assignment does not import sideband early");

		ThrowingType::reset();
		VariantT diff_move_src(std::in_place_type<ThrowingType>, ThrowingType{13});
		VariantT diff_move_dst(14);
		diff_move_src.sideband().set(81);
		diff_move_dst.sideband().set(82);
		ThrowingType::throw_on_move = ThrowingType::move_count + 1;
		expect_throw<std::runtime_error>(ctx, "different-index move assignment sideband publication waits for success", [&]() {
			diff_move_dst = std::move(diff_move_src);
		});
		check(ctx, diff_move_dst.valueless_by_exception(), "failed different-index move assignment becomes valueless");
		check(ctx, diff_move_dst.index() == std::variant_npos, "failed different-index move assignment publishes npos");
		check(ctx, static_cast<std::size_t>(diff_move_dst.sideband().get()) == 82, "failed different-index move assignment does not import sideband early");

		ThrowingType::reset();
		VariantT diff_assign_success_src(std::in_place_type<ThrowingType>, ThrowingType{21});
		VariantT diff_assign_success_dst(22);
		diff_assign_success_src.sideband().set(91);
		diff_assign_success_dst.sideband().set(92);
		diff_assign_success_dst = diff_assign_success_src;
		check(ctx, !diff_assign_success_dst.valueless_by_exception(), "successful different-index copy assignment stays engaged");
		check(ctx, holds_alternative<ThrowingType>(diff_assign_success_dst), "successful different-index copy assignment activates source alternative");
		check(ctx, get<ThrowingType>(diff_assign_success_dst).value == 21, "successful different-index copy assignment transfers payload");
		check(ctx, static_cast<std::size_t>(diff_assign_success_dst.sideband().get()) == 91, "successful different-index copy assignment imports sideband after success");

		ThrowingType::reset();
		VariantT diff_move_success_src(std::in_place_type<ThrowingType>, ThrowingType{31});
		VariantT diff_move_success_dst(32);
		diff_move_success_src.sideband().set(101);
		diff_move_success_dst.sideband().set(102);
		diff_move_success_dst = std::move(diff_move_success_src);
		check(ctx, !diff_move_success_dst.valueless_by_exception(), "successful different-index move assignment stays engaged");
		check(ctx, holds_alternative<ThrowingType>(diff_move_success_dst), "successful different-index move assignment activates source alternative");
		check(ctx, get<ThrowingType>(diff_move_success_dst).value == 31, "successful different-index move assignment transfers payload");
		check(ctx, static_cast<std::size_t>(diff_move_success_dst.sideband().get()) == 101, "successful different-index move assignment imports sideband after success");
	}

	if constexpr (variant_test::is_custom_variant_v<VariantT>) {
		// Custom-only behavior: assigning a valueless source exercises both destroy-active and no-op destroy paths.
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
		// Swap can fail while staging/moving values. The assertions focus on invariant preservation:
		// any surviving ThrowingType instances must match the engaged variants, and recovery must remain possible.
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
		// Custom-only behavior: this path checks the "throw before either side is destroyed" swap case.
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

template<template<class...> class VariantTemplate>
void run_custom_variant_lifetime_suite(const char* impl_name, int& failures) {
	TestContext ctx{impl_name, failures};
	using A = LifetimeVariantTracked<0>;
	using B = LifetimeVariantTracked<1>;
	using V = VariantTemplate<int, A, B, LifetimeVariantThrowingTracked>;

	A::reset();
	B::reset();
	LifetimeVariantThrowingTracked::reset();

	{
		V v;
		check(ctx, v.index() == 0, "default construction selects first alternative");
		check(ctx, sw::universal::internal::holds_alternative<int>(v), "default construction holds int");
		check(ctx, sw::universal::internal::get<int>(v) == 0, "default int payload");
	}

	A::reset();
	B::reset();
	LifetimeVariantThrowingTracked::reset();

	{
		// Switching alternatives should destroy the old active lifetime exactly once before the new one
		// becomes observable.
		V v(std::in_place_type<A>, 7);
		check(ctx, A::live == 1, "A emplace constructs one live object");
		check(ctx, sw::universal::internal::get<A>(v).value == 7, "A payload stored");

		v.template emplace<B>(9);
		check(ctx, A::live == 0, "switching to B destroys A");
		check(ctx, A::dtor == 1, "switching to B destroys A exactly once");
		check(ctx, B::live == 1, "switching to B constructs B");
		check(ctx, sw::universal::internal::get<B>(v).value == 9, "B payload stored");

		const int b_dtor_before_int = B::dtor;
		v.template emplace<0>(42);
		check(ctx, B::live == 0, "switching to int destroys B");
		check(ctx, B::dtor - b_dtor_before_int == 1, "switching to int destroys active B exactly once");
		check(ctx, v.index() == 0, "switching to int updates active index");
		check(ctx, sw::universal::internal::get<int>(v) == 42, "int payload after switch");
	}
	check(ctx, A::live == 0 && B::live == 0, "scope destruction leaves no tracked alternatives alive");

	A::reset();
	B::reset();

	{
		V source(std::in_place_type<A>, 11);
		V copy(source);
		check(ctx, sw::universal::internal::holds_alternative<A>(copy), "copy construction preserves active alternative");
		check(ctx, sw::universal::internal::get<A>(copy).value == 11, "copy construction preserves payload");
		check(ctx, A::copy_ctor == 1, "copy construction copies active alternative exactly once");
		check(ctx, A::live == 2, "copy construction leaves two live A alternatives");

		V moved(std::move(copy));
		check(ctx, sw::universal::internal::holds_alternative<A>(moved), "move construction preserves active alternative");
		check(ctx, sw::universal::internal::get<A>(moved).value == 11, "move construction preserves payload");
		check(ctx, A::move_ctor == 1, "move construction moves active alternative exactly once");
	}
	check(ctx, A::live == 0 && B::live == 0, "copy move destruction leaves no leaks");

	A::reset();
	B::reset();

	{
		// Assignment splits into two important cases:
		// - same active alternative: use assignment on the live object
		// - different active alternative: destroy/reconstruct
		V same_dst(std::in_place_type<A>, 2);
		V same_src(std::in_place_type<A>, 5);
		same_dst = same_src;
		check(ctx, sw::universal::internal::holds_alternative<A>(same_dst), "same-alternative copy assignment keeps active alternative");
		check(ctx, sw::universal::internal::get<A>(same_dst).value == 5, "same-alternative copy assignment updates payload");
		check(ctx, A::copy_assign == 1, "same-alternative copy assignment uses assignment");

		V dst(std::in_place_type<B>, 3);
		V src(std::in_place_type<A>, 12);
		dst = src;
		check(ctx, sw::universal::internal::holds_alternative<A>(dst), "copy assignment switches active alternative");
		check(ctx, sw::universal::internal::get<A>(dst).value == 12, "copy assignment preserves payload");
		check(ctx, B::live == 0, "copy assignment destroys replaced alternative");
		check(ctx, B::dtor == 1, "copy assignment destroys replaced alternative exactly once");

		V same_move_dst(std::in_place_type<A>, 20);
		V same_move_src(std::in_place_type<A>, 21);
		same_move_dst = std::move(same_move_src);
		check(ctx, sw::universal::internal::holds_alternative<A>(same_move_dst), "same-alternative move assignment keeps active alternative");
		check(ctx, sw::universal::internal::get<A>(same_move_dst).value == 21, "same-alternative move assignment updates payload");
		check(ctx, A::move_assign == 1, "same-alternative move assignment uses move assignment");

		V move_src(std::in_place_type<B>, 25);
		dst = std::move(move_src);
		check(ctx, sw::universal::internal::holds_alternative<B>(dst), "move assignment switches active alternative");
		check(ctx, sw::universal::internal::get<B>(dst).value == 25, "move assignment preserves payload");
	}
	check(ctx, A::live == 0 && B::live == 0, "assignment scope destruction leaves no leaks");

	A::reset();
	B::reset();

	{
		V v(std::in_place_type<A>, 1);
		v = B(2);
		check(ctx, sw::universal::internal::holds_alternative<B>(v), "assignment to B activates B");
		check(ctx, A::live == 0 && B::live == 1, "assignment to B destroys A");
		v = 17;
		check(ctx, v.index() == 0, "assignment to int activates int");
		check(ctx, B::live == 0, "assignment to int destroys B");
		v.template emplace<A>(33);
		check(ctx, A::live == 1, "re-emplace A reconstructs one object");
		check(ctx, sw::universal::internal::get<A>(v).value == 33, "re-emplace A payload");
	}
	check(ctx, A::live == 0 && B::live == 0, "repeated reassignment scope destruction leaves no leaks");

	if constexpr (requires(V& v) { v.sideband(); }) {
		// Sideband payload must be orthogonal to payload switching.
		A::reset();
		B::reset();
		V v(std::in_place_type<A>, 4);
		v.sideband().set(6);
		check(ctx, static_cast<std::size_t>(v.sideband().get()) == 6, "sideband write round-trips");
		v.template emplace<B>(8);
		check(ctx, static_cast<std::size_t>(v.sideband().get()) == 6, "sideband survives alternative switch");
		check(ctx, sw::universal::internal::get<B>(v).value == 8, "sideband variant keeps payload accessible");
	}

	{
		// Throwing emplace should preserve the prior alternative when construction fails before replacement.
		A::reset();
		B::reset();
		LifetimeVariantThrowingTracked::reset();
		V v(std::in_place_type<A>, 55);
		LifetimeVariantThrowingTracked::throw_on_ctor = 1;
		expect_throw<std::runtime_error>(ctx, "throwing emplace", [&]() {
			v.template emplace<LifetimeVariantThrowingTracked>(99);
		});
		check(ctx, !v.valueless_by_exception(), "throwing emplace preserves prior active alternative");
		check(ctx, sw::universal::internal::holds_alternative<A>(v), "throwing emplace keeps prior alternative active");
		check(ctx, sw::universal::internal::get<A>(v).value == 55, "throwing emplace preserves prior payload");
		check(ctx, A::live == 1, "throwing emplace leaves replaced alternative alive");
		check(ctx, LifetimeVariantThrowingTracked::live == 0, "throwing emplace does not leak failed construction");
	}

	check(ctx, A::live == 0 && B::live == 0, "final scope leaves no tracked alternatives alive");
}

int main() {
	int nrOfFailedTestCases = 0;
	report_std_variant_explicit_only_conversion_sentinel();
	report_custom_variant_explicit_only_conversion_sentinel();
	run_encoded_index_tests("encoded_index", nrOfFailedTestCases);
	run_variant_std_parity_tests(nrOfFailedTestCases);
	run_variant_suite<CustomVariant>("custom_indexed_variant", nrOfFailedTestCases);
	run_variant_suite<SidebandVariant>("custom_indexed_variant_sideband", nrOfFailedTestCases);
	run_variant_suite<std::variant>("std::variant", nrOfFailedTestCases);
	run_custom_variant_lifetime_suite<CustomVariant>("custom_indexed_variant_lifetime", nrOfFailedTestCases);
	run_custom_variant_lifetime_suite<SidebandVariant>("custom_indexed_variant_lifetime_sideband", nrOfFailedTestCases);

	sw::universal::ReportTestResult(nrOfFailedTestCases, "custom_indexed_variant", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
