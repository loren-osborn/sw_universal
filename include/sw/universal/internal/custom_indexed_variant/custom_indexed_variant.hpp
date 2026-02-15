#pragma once
// custom_indexed_variant.hpp: indexed variant utility mirroring std::variant (C++20)
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
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace sw { namespace universal {

namespace internal {

namespace custom_indexed_variant_detail {

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
			: (index_of_exact<T, Ts...>::value == std::variant_npos ? std::variant_npos : 1 + index_of_exact<T, Ts...>::value);
	};

	template<typename T>
	struct index_of_exact<T> {
		static constexpr std::size_t value = std::variant_npos;
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

} // namespace custom_indexed_variant_detail

namespace detail = custom_indexed_variant_detail;

template<std::size_t NTypes>
struct simple_encoded_index {
	simple_encoded_index() = default;
	std::size_t index() const { return index_; }
	void set_index(std::size_t val) { index_ = val; }

private:
	std::size_t index_ = 0;
};

template<std::size_t NTypes>
struct index_encoded_with_sideband_data {
	static constexpr std::size_t width = std::numeric_limits<std::size_t>::digits;
	static constexpr std::size_t npos_code = NTypes;

	static constexpr std::size_t ceil_log2(std::size_t value) {
		std::size_t bits = 0;
		std::size_t v = value > 0 ? value - 1 : 0;
		while (v > 0) {
			v >>= 1;
			++bits;
		}
		return bits;
	}

	static constexpr std::size_t index_bits = ceil_log2(NTypes + 1);
	static constexpr std::size_t index_mask = index_bits == 0 ? 0
		: (index_bits >= width ? ~std::size_t(0) : ((std::size_t(1) << index_bits) - 1));
	static constexpr std::size_t sideband_mask = ~index_mask;

	struct sideband_proxy {
		std::size_t* data = nullptr;

		operator std::size_t() const {
			if constexpr (index_bits >= width) {
				return 0;
			} else {
				return ((*data) & sideband_mask) >> index_bits;
			}
		}

		sideband_proxy& operator=(std::size_t value) {
			if constexpr (index_bits < width) {
				const std::size_t shifted = (value << index_bits) & sideband_mask;
				*data = ((*data) & index_mask) | shifted;
			}
			return *this;
		}
	};

	index_encoded_with_sideband_data() = default;

	std::size_t index() const {
		const std::size_t stored = data_ & index_mask;
		return stored == npos_code ? std::variant_npos : stored;
	}

	void set_index(std::size_t val) {
		const std::size_t stored = (val == std::variant_npos) ? npos_code : (val & index_mask);
		data_ = (data_ & sideband_mask) | stored;
	}

	sideband_proxy sideband() { return sideband_proxy{&data_}; }

private:
	std::size_t data_ = 0;
};

/// @brief An indexed variant implementation modeled after std::variant (C++20).
/// @tparam EncodedIndex Reserved index encoding selector (currently unused).
/// @tparam Types Alternative types stored in the variant.
/// @note This implementation prefers exact-type construction when using converting constructors.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
class custom_indexed_variant {
	static_assert(sizeof...(Types) > 0, "custom_indexed_variant must have at least one alternative");
	using storage_traits = detail::storage_traits<Types...>;
	static constexpr std::size_t ntypes = sizeof...(Types);

public:
	using encoded_index_t = EncodedIndex<ntypes>;
	using index_encoding = encoded_index_t;
	static constexpr std::size_t npos = std::variant_npos;

	using index_type = std::size_t;

	/// @brief Default-constructs the first alternative.
	custom_indexed_variant() noexcept(std::is_nothrow_default_constructible_v<detail::type_at_t<0, Types...>>)
		requires std::is_default_constructible_v<detail::type_at_t<0, Types...>>
		: index_(npos) {
		construct<0>();
		index_ = 0;
	}

	/// @brief Copy-constructs from another variant.
	custom_indexed_variant(const custom_indexed_variant& other)
		requires detail::all_copy_constructible<Types...>::value
		: index_(npos) {
		if (!other.valueless_by_exception()) {
			visit_active(other, [&](auto index_tag, const auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct<I>(value);
				index_ = I;
			});
		}
	}

	/// @brief Move-constructs from another variant.
	custom_indexed_variant(custom_indexed_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
		requires detail::all_move_constructible<Types...>::value
		: index_(npos) {
		if (!other.valueless_by_exception()) {
			visit_active(other, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct<I>(std::move(value));
				index_ = I;
			});
		}
	}

	/// @brief Constructs the specified alternative in-place by index.
	template<std::size_t I, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_index_t<I>, Args&&... args)
		: index_(npos) {
		construct<I>(std::forward<Args>(args)...);
		index_ = I;
	}

	/// @brief Constructs the specified alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_index_t<I>, std::initializer_list<U> init, Args&&... args)
		: index_(npos) {
		construct<I>(init, std::forward<Args>(args)...);
		index_ = I;
	}

	/// @brief Constructs the specified alternative in-place by type.
	template<typename T, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_type_t<T>, Args&&... args)
		: index_(npos) {
		constexpr std::size_t index = detail::index_of_exact_v<T, Types...>;
		static_assert(index != npos, "Type not found in custom_indexed_variant");
		construct<index>(std::forward<Args>(args)...);
		index_ = index;
	}

	/// @brief Constructs the specified alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_type_t<T>, std::initializer_list<U> init, Args&&... args)
		: index_(npos) {
		constexpr std::size_t index = detail::index_of_exact_v<T, Types...>;
		static_assert(index != npos, "Type not found in custom_indexed_variant");
		construct<index>(init, std::forward<Args>(args)...);
		index_ = index;
	}

	/// @brief Converting constructor from a value (exact-type match).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::enable_if_t<!std::is_same_v<U, custom_indexed_variant> &&
							!detail::is_in_place_index_v<U> &&
							!detail::is_in_place_type_v<U> &&
							(detail::index_of_exact_v<U, Types...> != npos), int> = 0>
		constexpr custom_indexed_variant(T&& value)
		: index_(npos) {
		constexpr std::size_t index = detail::index_of_exact_v<U, Types...>;
		construct<index>(std::forward<T>(value));
		index_ = index;
	}

	~custom_indexed_variant() {
		destroy_active();
	}

	/// @brief Copy-assigns from another variant.
	custom_indexed_variant& operator=(const custom_indexed_variant& other)
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
	custom_indexed_variant& operator=(custom_indexed_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
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
	custom_indexed_variant& operator=(T&& value) {
		constexpr std::size_t I = detail::index_of_exact_v<U, Types...>;
		if (index_ == I) {
			*std::launder(reinterpret_cast<U*>(&storage_)) = std::forward<T>(value);
		} else {
			if constexpr (std::is_nothrow_move_constructible_v<U> || std::is_nothrow_copy_constructible_v<U>) {
				std::optional<U> temp;
				temp.emplace(std::forward<T>(value));
				destroy_active();
				if constexpr (std::is_nothrow_move_constructible_v<U>) {
					construct<I>(std::move(*temp));
				} else {
					construct<I>(*temp);
				}
				index_ = I;
			} else {
				destroy_active();
				construct<I>(std::forward<T>(value));
				index_ = I;
			}
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
		using T = detail::type_at_t<I, Types...>;
		if constexpr (std::is_nothrow_move_constructible_v<T> || std::is_nothrow_copy_constructible_v<T>) {
			std::optional<T> temp;
			temp.emplace(std::forward<Args>(args)...);
			destroy_active();
			if constexpr (std::is_nothrow_move_constructible_v<T>) {
				construct<I>(std::move(*temp));
			} else {
				construct<I>(*temp);
			}
			index_ = I;
		} else {
			destroy_active();
			construct<I>(std::forward<Args>(args)...);
			index_ = I;
		}
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		using T = detail::type_at_t<I, Types...>;
		if constexpr (std::is_nothrow_move_constructible_v<T> || std::is_nothrow_copy_constructible_v<T>) {
			std::optional<T> temp;
			temp.emplace(init, std::forward<Args>(args)...);
			destroy_active();
			if constexpr (std::is_nothrow_move_constructible_v<T>) {
				construct<I>(std::move(*temp));
			} else {
				construct<I>(*temp);
			}
			index_ = I;
		} else {
			destroy_active();
			construct<I>(init, std::forward<Args>(args)...);
			index_ = I;
		}
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by type.
	template<typename T, typename... Args>
	decltype(auto) emplace(Args&&... args) {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return emplace<I>(std::forward<Args>(args)...);
	}

	/// @brief Constructs a new alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return emplace<I>(init, std::forward<Args>(args)...);
	}

	/// @brief Swaps this variant with another.
	void swap(custom_indexed_variant& other)
		requires detail::all_swappable<Types...>::value && detail::all_move_constructible<Types...>::value {
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
		if (valueless_by_exception()) {
			bool constructed = false;
			visit_active(other, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct<I>(std::move(value));
				index_ = I;
				constructed = true;
			});
			if (constructed) {
				other.destroy_active();
			}
			return;
		}
		if (other.valueless_by_exception()) {
			bool constructed = false;
			visit_active(*this, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				other.template construct<I>(std::move(value));
				other.index_ = I;
				constructed = true;
			});
			if (constructed) {
				destroy_active();
			}
			return;
		}
		visit_active(*this, [&](auto left_tag, auto&) {
			constexpr std::size_t L = decltype(left_tag)::value;
			visit_active(other, [&](auto right_tag, auto&) {
				constexpr std::size_t R = decltype(right_tag)::value;
				swap_different_index<L, R>(other);
			});
		});
	}

	/// @brief Access the active alternative by index.
	template<std::size_t I>
	decltype(auto) get() & {
		if (index_ != I) {
			throw std::bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return *std::launder(reinterpret_cast<T*>(&storage_));
	}

	/// @brief Access the active alternative by index (const).
	template<std::size_t I>
	decltype(auto) get() const & {
		if (index_ != I) {
			throw std::bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return *std::launder(reinterpret_cast<const T*>(&storage_));
	}

	/// @brief Access the active alternative by index (rvalue).
	template<std::size_t I>
	decltype(auto) get() && {
		if (index_ != I) {
			throw std::bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return std::move(*std::launder(reinterpret_cast<T*>(&storage_)));
	}

	/// @brief Access the active alternative by index (const rvalue).
	template<std::size_t I>
	decltype(auto) get() const && {
		if (index_ != I) {
			throw std::bad_variant_access{};
		}
		using T = detail::type_at_t<I, Types...>;
		return std::move(*std::launder(reinterpret_cast<const T*>(&storage_)));
	}

	/// @brief Access the active alternative by type.
	template<typename T>
	decltype(auto) get() & {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (const).
	template<typename T>
	decltype(auto) get() const & {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (rvalue).
	template<typename T>
	decltype(auto) get() && {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return std::move(get<I>());
	}

	/// @brief Access the active alternative by type (const rvalue).
	template<typename T>
	decltype(auto) get() const && {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
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
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get_if<I>();
	}

	/// @brief Pointer access to alternative by type (const).
	template<typename T>
	auto get_if() const noexcept {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get_if<I>();
	}

	/// @brief Returns true if the active alternative matches T.
	template<typename T>
	bool holds_alternative() const noexcept {
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
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
		const std::size_t active = index_;
		destroy_active_impl<0>(active);
		index_ = npos;
	}

	template<std::size_t I, typename Variant>
	static auto active_ptr(Variant& variant) noexcept {
		using VariantNoRef = std::remove_reference_t<Variant>;
		using Base = detail::type_at_t<I, Types...>;
		using CvBase = std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<Base>, Base>;
		using CvVolBase = std::conditional_t<std::is_volatile_v<VariantNoRef>, std::add_volatile_t<CvBase>, CvBase>;
		return std::launder(reinterpret_cast<CvVolBase*>(&variant.storage_));
	}

	template<typename Variant, typename Fn>
	static void visit_active(Variant&& variant, Fn&& fn) {
		visit_active_impl<0>(std::forward<Variant>(variant), std::forward<Fn>(fn));
	}

	template<std::size_t I, typename Variant, typename Fn>
	static void visit_active_impl(Variant&& variant, Fn&& fn) {
		if constexpr (I < sizeof...(Types)) {
			switch (variant.index_) {
			case I:
				fn(std::integral_constant<std::size_t, I>{}, *active_ptr<I>(variant));
				return;
			default:
				visit_active_impl<I + 1>(std::forward<Variant>(variant), std::forward<Fn>(fn));
				return;
			}
		}
	}

	template<std::size_t I>
	void destroy_active_impl(std::size_t active) noexcept {
		if constexpr (I < sizeof...(Types)) {
			using T = detail::type_at_t<I, Types...>;
			switch (active) {
			case I:
				active_ptr<I>(*this)->~T();
				return;
			default:
				destroy_active_impl<I + 1>(active);
				return;
			}
		}
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
		visit_active(other, [&](auto index_tag, auto&& value) {
			constexpr std::size_t I = decltype(index_tag)::value;
			construct<I>(std::forward<decltype(value)>(value));
			index_ = I;
		});
	}

	void swap_same_index(custom_indexed_variant& other) {
		visit_active(*this, [&](auto index_tag, auto& value) {
			using std::swap;
			swap(value, other.template get<decltype(index_tag)::value>());
		});
	}

	template<std::size_t L, std::size_t R>
	void swap_different_index(custom_indexed_variant& other) {
		using Left = detail::type_at_t<L, Types...>;
		using Right = detail::type_at_t<R, Types...>;
		std::optional<Left> left_value;
		std::optional<Right> right_value;
		left_value.emplace(std::move(*active_ptr<L>(*this)));
		right_value.emplace(std::move(*active_ptr<R>(other)));
		destroy_active();
		other.destroy_active();
		bool left_constructed = false;
		try {
			construct<R>(std::move(*right_value));
			index_ = R;
			left_constructed = true;
		} catch (...) {
			index_ = npos;
			throw;
		}
		try {
			other.template construct<L>(std::move(*left_value));
			other.index_ = L;
		} catch (...) {
			other.index_ = npos;
			if (left_constructed) {
				destroy_active();
			}
			throw;
		}
	}

	typename storage_traits::storage_t storage_{};
	std::size_t index_ = npos;
};

/// @brief Variant size trait for custom_indexed_variant.
template<typename Variant>
struct variant_size;

template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<const custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<volatile custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<const volatile custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

template<typename Variant>
inline constexpr std::size_t variant_size_v = variant_size<Variant>::value;

/// @brief Variant alternative trait for custom_indexed_variant.
template<std::size_t I, typename Variant>
struct variant_alternative;

template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>> {
	static_assert(I < sizeof...(Types), "variant alternative index out of bounds");
	using type = custom_indexed_variant_detail::type_at_t<I, Types...>;
};

template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, const custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_const_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, volatile custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_volatile_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, const volatile custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_cv_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

template<std::size_t I, typename Variant>
using variant_alternative_t = typename variant_alternative<I, Variant>::type;

/// @brief True if variant holds the alternative T.
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline bool holds_alternative(const custom_indexed_variant<EncodedIndex, Types...>& v) noexcept {
	return v.template holds_alternative<T>();
}

/// @brief Access the alternative by index.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(custom_indexed_variant<EncodedIndex, Types...>& v) {
	return v.template get<I>();
}

/// @brief Access the alternative by index (const).
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(const custom_indexed_variant<EncodedIndex, Types...>& v) {
	return v.template get<I>();
}

/// @brief Access the alternative by index (rvalue).
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(custom_indexed_variant<EncodedIndex, Types...>&& v) {
	return std::move(v).template get<I>();
}

/// @brief Access the alternative by index (const rvalue).
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(const custom_indexed_variant<EncodedIndex, Types...>&& v) {
	return std::move(v).template get<I>();
}

/// @brief Access the alternative by type.
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(custom_indexed_variant<EncodedIndex, Types...>& v) {
	return v.template get<T>();
}

/// @brief Access the alternative by type (const).
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(const custom_indexed_variant<EncodedIndex, Types...>& v) {
	return v.template get<T>();
}

/// @brief Access the alternative by type (rvalue).
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(custom_indexed_variant<EncodedIndex, Types...>&& v) {
	return std::move(v).template get<T>();
}

/// @brief Access the alternative by type (const rvalue).
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline decltype(auto) get(const custom_indexed_variant<EncodedIndex, Types...>&& v) {
	return std::move(v).template get<T>();
}

/// @brief Pointer access to alternative by index.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline auto get_if(custom_indexed_variant<EncodedIndex, Types...>* v) noexcept {
	return v ? v->template get_if<I>() : nullptr;
}

/// @brief Pointer access to alternative by index (const).
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline auto get_if(const custom_indexed_variant<EncodedIndex, Types...>* v) noexcept {
	return v ? v->template get_if<I>() : nullptr;
}

/// @brief Pointer access to alternative by type.
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline auto get_if(custom_indexed_variant<EncodedIndex, Types...>* v) noexcept {
	return v ? v->template get_if<T>() : nullptr;
}

/// @brief Pointer access to alternative by type (const).
template<typename T, template<std::size_t NTypes> class EncodedIndex, typename... Types>
inline auto get_if(const custom_indexed_variant<EncodedIndex, Types...>* v) noexcept {
	return v ? v->template get_if<T>() : nullptr;
}

namespace custom_indexed_variant_detail {

	template<typename Variant, std::size_t... Is>
	auto make_reference_variant_impl(Variant& v, std::index_sequence<Is...>) {
		using ref_variant = std::variant<std::reference_wrapper<variant_alternative_t<Is, Variant>>...>;
		if (v.index() == std::variant_npos) {
			throw std::bad_variant_access{};
		}
		std::optional<ref_variant> refs;
		((v.index() == Is ? (refs.emplace(std::in_place_index<Is>, std::ref(v.template get<Is>())), true) : false) || ...);
		return *refs;
	}

	template<typename Variant>
	auto make_reference_variant(Variant& v) {
		return make_reference_variant_impl(v, std::make_index_sequence<variant_size_v<Variant>>{});
	}

} // namespace custom_indexed_variant_detail

/// @brief Visit the active alternative(s) with a callable.
template<typename Visitor, typename... Variants>
inline decltype(auto) visit(Visitor&& vis, Variants&&... variants) {
	if ((variants.valueless_by_exception() || ...)) {
		throw std::bad_variant_access{};
	}
	auto ref_variants = std::make_tuple(custom_indexed_variant_detail::make_reference_variant(variants)...);
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
