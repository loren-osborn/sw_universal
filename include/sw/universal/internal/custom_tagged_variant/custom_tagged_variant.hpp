#pragma once
// custom_tagged_variant.hpp: tagged variant utility mirroring std::variant (C++20)
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sw { namespace universal {

namespace internal {

/// @brief Exception type thrown on invalid variant access.
class bad_variant_access final : public std::exception {
public:
	const char* what() const noexcept override {
		return "bad custom_tagged_variant access";
	}
};

/// @brief Sentinel index value for valueless variants.
inline constexpr std::size_t variant_npos = static_cast<std::size_t>(-1);

namespace custom_tagged_variant_detail {

	template<std::size_t I, typename... Ts>
	struct type_at;

	template<std::size_t I, typename T, typename... Ts>
	struct type_at<I, T, Ts...> : type_at<I - 1, Ts...> {};

	template<typename T, typename... Ts>
	struct type_at<0, T, Ts...> { using type = T; };

	template<std::size_t I, typename... Ts>
	using type_at_t = typename type_at<I, Ts...>::type;

	template<typename T, typename... Ts>
	struct index_of_exact;

	template<typename T, typename U, typename... Ts>
	struct index_of_exact<T, U, Ts...> {
		static constexpr std::size_t value = std::is_same_v<T, U>
			? 0
			: (index_of_exact<T, Ts...>::value == variant_npos ? variant_npos : 1 + index_of_exact<T, Ts...>::value);
	};

	template<typename T>
	struct index_of_exact<T> {
		static constexpr std::size_t value = variant_npos;
	};

	template<typename T, typename... Ts>
	inline constexpr std::size_t index_of_exact_v = index_of_exact<T, Ts...>::value;

	template<typename... Ts>
	struct storage_traits {
		static constexpr std::size_t max_size = (std::max)({sizeof(Ts)...});
		static constexpr std::size_t max_align = (std::max)({alignof(Ts)...});
		using storage_t = std::aligned_storage_t<max_size, max_align>;
	};

	template<typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

	template<typename... Ts>
	struct all_copy_constructible : std::conjunction<std::is_copy_constructible<Ts>...> {};

	template<typename... Ts>
	struct all_move_constructible : std::conjunction<std::is_move_constructible<Ts>...> {};

	template<typename... Ts>
	struct all_copy_assignable : std::conjunction<std::is_copy_assignable<Ts>...> {};

	template<typename... Ts>
	struct all_move_assignable : std::conjunction<std::is_move_assignable<Ts>...> {};

	template<typename... Ts>
	struct all_swappable : std::conjunction<std::is_swappable<Ts>...> {};

	template<typename... Ts>
	struct all_nothrow_move_constructible : std::conjunction<std::is_nothrow_move_constructible<Ts>...> {};

	template<typename T>
	struct is_in_place_index : std::false_type {};

	template<std::size_t I>
	struct is_in_place_index<std::in_place_index_t<I>> : std::true_type {};

	template<typename T>
	struct is_in_place_type : std::false_type {};

	template<typename T>
	struct is_in_place_type<std::in_place_type_t<T>> : std::true_type {};

	template<typename T>
	inline constexpr bool is_in_place_index_v = is_in_place_index<T>::value;

	template<typename T>
	inline constexpr bool is_in_place_type_v = is_in_place_type<T>::value;

} // namespace custom_tagged_variant_detail

/// @brief A tagged variant implementation modeled after std::variant (C++20).
/// @tparam TagEncoding Reserved tag encoding selector (currently unused).
/// @tparam Types Alternative types stored in the variant.
/// @note This implementation prefers exact-type construction when using converting constructors.
template<typename TagEncoding, typename... Types>
class custom_tagged_variant {
	static_assert(sizeof...(Types) > 0, "custom_tagged_variant must have at least one alternative");

	using detail = custom_tagged_variant_detail;
	using storage_traits = detail::storage_traits<Types...>;

public:
	using tag_encoding = TagEncoding;
	static constexpr std::size_t npos = variant_npos;

	using index_type = std::size_t;

	/// @brief Default-constructs the first alternative.
	custom_tagged_variant() noexcept(std::is_nothrow_default_constructible_v<detail::type_at_t<0, Types...>>)
		requires std::is_default_constructible_v<detail::type_at_t<0, Types...>>
		: index_(0) {
		construct<0>();
	}

	/// @brief Copy-constructs from another variant.
	custom_tagged_variant(const custom_tagged_variant& other)
		requires detail::all_copy_constructible<Types...>::value
		: index_(other.index_) {
		if (!other.valueless_by_exception()) {
			visit_active(other, [&](auto index_tag, const auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct<I>(value);
			});
		}
	}

	/// @brief Move-constructs from another variant.
	custom_tagged_variant(custom_tagged_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
		requires detail::all_move_constructible<Types...>::value
		: index_(other.index_) {
		if (!other.valueless_by_exception()) {
			visit_active(other, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct<I>(std::move(value));
			});
		}
	}

	/// @brief Constructs the specified alternative in-place by index.
	template<std::size_t I, typename... Args>
	constexpr explicit custom_tagged_variant(std::in_place_index_t<I>, Args&&... args)
		: index_(I) {
		construct<I>(std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	constexpr explicit custom_tagged_variant(std::in_place_index_t<I>, std::initializer_list<U> init, Args&&... args)
		: index_(I) {
		construct<I>(init, std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by type.
	template<typename T, typename... Args>
	constexpr explicit custom_tagged_variant(std::in_place_type_t<T>, Args&&... args)
		: index_(detail::index_of_exact_v<T, Types...>) {
		static_assert(index_ != npos, "Type not found in custom_tagged_variant");
		construct<detail::index_of_exact_v<T, Types...>>(std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	constexpr explicit custom_tagged_variant(std::in_place_type_t<T>, std::initializer_list<U> init, Args&&... args)
		: index_(detail::index_of_exact_v<T, Types...>) {
		static_assert(index_ != npos, "Type not found in custom_tagged_variant");
		construct<detail::index_of_exact_v<T, Types...>>(init, std::forward<Args>(args)...);
	}

	/// @brief Converting constructor from a value (exact-type match).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::enable_if_t<!std::is_same_v<U, custom_tagged_variant> &&
							!detail::is_in_place_index_v<U> &&
							!detail::is_in_place_type_v<U> &&
							(detail::index_of_exact_v<U, Types...> != npos), int> = 0>
		constexpr custom_tagged_variant(T&& value)
		: index_(detail::index_of_exact_v<U, Types...>) {
		construct<detail::index_of_exact_v<U, Types...>>(std::forward<T>(value));
	}

	~custom_tagged_variant() {
		destroy_active();
	}

	/// @brief Copy-assigns from another variant.
	custom_tagged_variant& operator=(const custom_tagged_variant& other)
		requires detail::all_copy_constructible<Types...>::value && detail::all_copy_assignable<Types...>::value {
		if (this == &other) {
			return *this;
		}
		if (other.valueless_by_exception()) {
			destroy_active();
			index_ = npos;
			return *this;
		}
		assign_from_other(other);
		return *this;
	}

	/// @brief Move-assigns from another variant.
	custom_tagged_variant& operator=(custom_tagged_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
		requires detail::all_move_constructible<Types...>::value && detail::all_move_assignable<Types...>::value {
		if (this == &other) {
			return *this;
		}
		if (other.valueless_by_exception()) {
			destroy_active();
			index_ = npos;
			return *this;
		}
		assign_from_other(std::move(other));
		return *this;
	}

	/// @brief Assigns from a value (exact-type match).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::enable_if_t<(detail::index_of_exact_v<U, Types...> != npos), int> = 0>
	custom_tagged_variant& operator=(T&& value) {
		constexpr std::size_t I = detail::index_of_exact_v<U, Types...>;
		if (index_ == I) {
			*std::launder(reinterpret_cast<U*>(&storage_)) = std::forward<T>(value);
		} else {
			destroy_active();
			index_ = I;
			construct<I>(std::forward<T>(value));
		}
		return *this;
	}

	/// @brief Checks if the variant has no active alternative.
	bool valueless_by_exception() const noexcept { return index_ == npos; }

	/// @brief Returns the index of the active alternative.
	std::size_t index() const noexcept { return index_; }

	/// @brief Constructs a new alternative in-place by index.
	template<std::size_t I, typename... Args>
	decltype(auto) emplace(Args&&... args) {
		destroy_active();
		index_ = I;
		construct<I>(std::forward<Args>(args)...);
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		destroy_active();
		index_ = I;
		construct<I>(init, std::forward<Args>(args)...);
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by type.
	template<typename T, typename... Args>
	decltype(auto) emplace(Args&&... args) {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return emplace<I>(std::forward<Args>(args)...);
	}

	/// @brief Constructs a new alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return emplace<I>(init, std::forward<Args>(args)...);
	}

	/// @brief Swaps this variant with another.
	void swap(custom_tagged_variant& other)
		requires detail::all_swappable<Types...>::value {
		if (this == &other) {
			return;
		}
		if (valueless_by_exception() && other.valueless_by_exception()) {
			return;
		}
		if (index_ == other.index_) {
			swap_same_index(other);
			return;
		}
		custom_tagged_variant tmp(std::move(*this));
		*this = std::move(other);
		other = std::move(tmp);
	}

	/// @brief Access the active alternative by index.
	template<std::size_t I>
	decltype(auto) get() & {
		if (index_ != I) {
			throw bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return *std::launder(reinterpret_cast<T*>(&storage_));
	}

	/// @brief Access the active alternative by index (const).
	template<std::size_t I>
	decltype(auto) get() const & {
		if (index_ != I) {
			throw bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return *std::launder(reinterpret_cast<const T*>(&storage_));
	}

	/// @brief Access the active alternative by index (rvalue).
	template<std::size_t I>
	decltype(auto) get() && {
		if (index_ != I) {
			throw bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return std::move(*std::launder(reinterpret_cast<T*>(&storage_)));
	}

	/// @brief Access the active alternative by index (const rvalue).
	template<std::size_t I>
	decltype(auto) get() const && {
		if (index_ != I) {
			throw bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return std::move(*std::launder(reinterpret_cast<const T*>(&storage_)));
	}

	/// @brief Access the active alternative by type.
	template<typename T>
	decltype(auto) get() & {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (const).
	template<typename T>
	decltype(auto) get() const & {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (rvalue).
	template<typename T>
	decltype(auto) get() && {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return std::move(get<I>());
	}

	/// @brief Access the active alternative by type (const rvalue).
	template<typename T>
	decltype(auto) get() const && {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return std::move(get<I>());
	}

	/// @brief Pointer access to alternative by index.
	template<std::size_t I>
	auto get_if() noexcept {
		using T = detail::type_at_t<I, Types...>;
		return (index_ == I) ? std::launder(reinterpret_cast<T*>(&storage_)) : nullptr;
	}

	/// @brief Pointer access to alternative by index (const).
	template<std::size_t I>
	auto get_if() const noexcept {
		using T = detail::type_at_t<I, Types...>;
		return (index_ == I) ? std::launder(reinterpret_cast<const T*>(&storage_)) : nullptr;
	}

	/// @brief Pointer access to alternative by type.
	template<typename T>
	auto get_if() noexcept {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return get_if<I>();
	}

	/// @brief Pointer access to alternative by type (const).
	template<typename T>
	auto get_if() const noexcept {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return get_if<I>();
	}

	/// @brief Returns true if the active alternative matches T.
	template<typename T>
	bool holds_alternative() const noexcept {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_tagged_variant");
		return index_ == I;
	}

private:
	template<std::size_t I, typename... Args>	
	void construct(Args&&... args) {
		using T = detail::type_at_t<I, Types...>;
		::new (static_cast<void*>(&storage_)) T(std::forward<Args>(args)...);
	}

	void destroy_active() noexcept {
		if (valueless_by_exception()) {
			return;
		}
		visit_active(*this, [&](auto index_tag, auto& value) {
			using T = std::decay_t<decltype(value)>;
			value.~T();
			index_ = npos;
		});
	}

	template<typename Variant, typename Fn>
	static void visit_active(Variant&& variant, Fn&& fn) {
		using VariantNoRef = std::remove_reference_t<Variant>;
		switch (variant.index_) {
		case 0:
			fn(std::integral_constant<std::size_t, 0>{},
				*std::launder(reinterpret_cast<std::conditional_t<std::is_volatile_v<VariantNoRef>,
					std::add_volatile_t<std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<detail::type_at_t<0, Types...>>, detail::type_at_t<0, Types...>>>,
					std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<detail::type_at_t<0, Types...>>, detail::type_at_t<0, Types...>>>*>(&variant.storage_)));
			return;
		default:
			visit_active_impl(std::forward<Variant>(variant), std::forward<Fn>(fn), std::make_index_sequence<sizeof...(Types)>{});
			return;
		}
	}

	template<typename Variant, typename Fn, std::size_t... Is>
	static void visit_active_impl(Variant&& variant, Fn&& fn, std::index_sequence<Is...>) {
		using VariantNoRef = std::remove_reference_t<Variant>;
		((variant.index_ == Is ? (fn(std::integral_constant<std::size_t, Is>{},
			*std::launder(reinterpret_cast<std::conditional_t<std::is_volatile_v<VariantNoRef>,
				std::add_volatile_t<std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<detail::type_at_t<Is, Types...>>, detail::type_at_t<Is, Types...>>>,
				std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<detail::type_at_t<Is, Types...>>, detail::type_at_t<Is, Types...>>>*>(&variant.storage_))), true) : false) || ...);
	}

	template<typename Other>
	void assign_from_other(Other&& other) {
		if (index_ == other.index_) {
			visit_active(*this, [&](auto index_tag, auto& value) {
				value = std::forward<Other>(other).template get<decltype(index_tag)::value>();
			});
			return;
		}
		destroy_active();
		index_ = other.index_;
		visit_active(other, [&](auto index_tag, auto&& value) {
			constexpr std::size_t I = decltype(index_tag)::value;
			construct<I>(std::forward<decltype(value)>(value));
		});
	}

	void swap_same_index(custom_tagged_variant& other) {
		visit_active(*this, [&](auto index_tag, auto& value) {
			using std::swap;
			swap(value, other.template get<decltype(index_tag)::value>());
		});
	}

	typename storage_traits::storage_t storage_{};
	std::size_t index_ = npos;
};

/// @brief Variant size trait for custom_tagged_variant.
template<typename Variant>
struct variant_size;

template<typename TagEncoding, typename... Types>
struct variant_size<custom_tagged_variant<TagEncoding, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<typename TagEncoding, typename... Types>
struct variant_size<const custom_tagged_variant<TagEncoding, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<typename TagEncoding, typename... Types>
struct variant_size<volatile custom_tagged_variant<TagEncoding, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<typename TagEncoding, typename... Types>
struct variant_size<const volatile custom_tagged_variant<TagEncoding, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<typename Variant>
inline constexpr std::size_t variant_size_v = variant_size<Variant>::value;

/// @brief Variant alternative trait for custom_tagged_variant.
template<std::size_t I, typename Variant>
struct variant_alternative;

template<std::size_t I, typename TagEncoding, typename... Types>
struct variant_alternative<I, custom_tagged_variant<TagEncoding, Types...>> {
	static_assert(I < sizeof...(Types), "variant alternative index out of bounds");
	using type = custom_tagged_variant_detail::type_at_t<I, Types...>;
};

template<std::size_t I, typename TagEncoding, typename... Types>
struct variant_alternative<I, const custom_tagged_variant<TagEncoding, Types...>> {
	using type = std::add_const_t<typename variant_alternative<I, custom_tagged_variant<TagEncoding, Types...>>::type>;
};

template<std::size_t I, typename TagEncoding, typename... Types>
struct variant_alternative<I, volatile custom_tagged_variant<TagEncoding, Types...>> {
	using type = std::add_volatile_t<typename variant_alternative<I, custom_tagged_variant<TagEncoding, Types...>>::type>;
};

template<std::size_t I, typename TagEncoding, typename... Types>
struct variant_alternative<I, const volatile custom_tagged_variant<TagEncoding, Types...>> {
	using type = std::add_cv_t<typename variant_alternative<I, custom_tagged_variant<TagEncoding, Types...>>::type>;
};

template<std::size_t I, typename Variant>
using variant_alternative_t = typename variant_alternative<I, Variant>::type;

/// @brief True if variant holds the alternative T.
template<typename T, typename TagEncoding, typename... Types>
inline bool holds_alternative(const custom_tagged_variant<TagEncoding, Types...>& v) noexcept {
	return v.template holds_alternative<T>();
}

/// @brief Access the alternative by index.
template<std::size_t I, typename TagEncoding, typename... Types>
inline decltype(auto) get(custom_tagged_variant<TagEncoding, Types...>& v) {
	return v.template get<I>();
}

/// @brief Access the alternative by index (const).
template<std::size_t I, typename TagEncoding, typename... Types>
inline decltype(auto) get(const custom_tagged_variant<TagEncoding, Types...>& v) {
	return v.template get<I>();
}

/// @brief Access the alternative by index (rvalue).
template<std::size_t I, typename TagEncoding, typename... Types>
inline decltype(auto) get(custom_tagged_variant<TagEncoding, Types...>&& v) {
	return std::move(v).template get<I>();
}

/// @brief Access the alternative by index (const rvalue).
template<std::size_t I, typename TagEncoding, typename... Types>
inline decltype(auto) get(const custom_tagged_variant<TagEncoding, Types...>&& v) {
	return std::move(v).template get<I>();
}

/// @brief Access the alternative by type.
template<typename T, typename TagEncoding, typename... Types>
inline decltype(auto) get(custom_tagged_variant<TagEncoding, Types...>& v) {
	return v.template get<T>();
}

/// @brief Access the alternative by type (const).
template<typename T, typename TagEncoding, typename... Types>
inline decltype(auto) get(const custom_tagged_variant<TagEncoding, Types...>& v) {
	return v.template get<T>();
}

/// @brief Access the alternative by type (rvalue).
template<typename T, typename TagEncoding, typename... Types>
inline decltype(auto) get(custom_tagged_variant<TagEncoding, Types...>&& v) {
	return std::move(v).template get<T>();
}

/// @brief Access the alternative by type (const rvalue).
template<typename T, typename TagEncoding, typename... Types>
inline decltype(auto) get(const custom_tagged_variant<TagEncoding, Types...>&& v) {
	return std::move(v).template get<T>();
}

/// @brief Pointer access to alternative by index.
template<std::size_t I, typename TagEncoding, typename... Types>
inline auto get_if(custom_tagged_variant<TagEncoding, Types...>* v) noexcept {
	return v ? v->template get_if<I>() : nullptr;
}

/// @brief Pointer access to alternative by index (const).
template<std::size_t I, typename TagEncoding, typename... Types>
inline auto get_if(const custom_tagged_variant<TagEncoding, Types...>* v) noexcept {
	return v ? v->template get_if<I>() : nullptr;
}

/// @brief Pointer access to alternative by type.
template<typename T, typename TagEncoding, typename... Types>
inline auto get_if(custom_tagged_variant<TagEncoding, Types...>* v) noexcept {
	return v ? v->template get_if<T>() : nullptr;
}

/// @brief Pointer access to alternative by type (const).
template<typename T, typename TagEncoding, typename... Types>
inline auto get_if(const custom_tagged_variant<TagEncoding, Types...>* v) noexcept {
	return v ? v->template get_if<T>() : nullptr;
}

namespace custom_tagged_variant_detail {

	template<typename Variant, std::size_t... Is>
	auto make_reference_variant_impl(Variant& v, std::index_sequence<Is...>) {
		using ref_variant = std::variant<std::reference_wrapper<variant_alternative_t<Is, Variant>>...>;
		ref_variant refs;
		switch (v.index()) {
		case variant_npos:
			break;
		default:
			((v.index() == Is ? (refs.template emplace<Is>(std::ref(v.template get<Is>())), true) : false) || ...);
			break;
		}
		return refs;
	}

	template<typename Variant>
	auto make_reference_variant(Variant& v) {
		return make_reference_variant_impl(v, std::make_index_sequence<variant_size_v<Variant>>{});
	}

} // namespace custom_tagged_variant_detail

/// @brief Visit the active alternative(s) with a callable.
template<typename Visitor, typename... Variants>
inline decltype(auto) visit(Visitor&& vis, Variants&&... variants) {
	if ((variants.valueless_by_exception() || ...)) {
		throw bad_variant_access{};
	}
	auto ref_variants = std::make_tuple(custom_tagged_variant_detail::make_reference_variant(variants)...);
	return std::apply([
		&](auto&... refs) -> decltype(auto) {
			auto wrapper = [&](auto&... ref_wrappers) -> decltype(auto) {
				return std::invoke(std::forward<Visitor>(vis), ref_wrappers.get()...);
			};
			return std::visit(wrapper, refs...);
		}, ref_variants);
}

} // namespace internal

}} // namespace sw::universal
