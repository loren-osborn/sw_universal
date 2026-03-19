#pragma once
/**
 * @file custom_indexed_variant.hpp
 * @brief Indexed variant utility that mirrors key std::variant semantics.
 *
 * This header provides:
 * - encoded index helpers used by custom_indexed_variant,
 * - the custom_indexed_variant container template,
 * - trait adapters (variant_size / variant_alternative),
 * - and free-function helpers (get/get_if/holds_alternative/visit).
 */
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <concepts>
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
#include <bit>

#include "universal/internal/bitvector/bitfield_pack.hpp"
#include "universal/internal/utility/pack_element.hpp"

namespace sw { namespace universal {

namespace internal {

namespace custom_indexed_variant_detail {

	/// @brief Selects the Ith alternative type from a variant parameter pack.
	template<std::size_t I, typename... Ts>
	using type_at_t = internal_utility::pack_element_t<I, Ts...>;

	/// @brief Finds the unique exact-match index of `T` in `Ts...`, or `std::variant_npos`.
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

	/// @brief Counts how many times `T` appears exactly in `Ts...`.
	template<typename T, typename... Ts>
	inline constexpr std::size_t count_exact_v = (std::size_t{0} + ... + (std::is_same_v<T, Ts> ? std::size_t{1} : std::size_t{0}));

	/// @brief True when `To{From}` is well-formed, excluding narrowing list-initialization.
	template<typename To, typename From, typename = void>
	struct non_narrowing_list_initializable : std::false_type {};

	template<typename To, typename From>
	struct non_narrowing_list_initializable<To, From, std::void_t<decltype(To{std::declval<From>()})>> : std::true_type {};

	template<typename To, typename From>
	inline constexpr bool non_narrowing_list_initializable_v = non_narrowing_list_initializable<To, From>::value;

	template<typename To, typename From>
	inline constexpr bool non_narrowing_convertible_v =
		std::is_convertible_v<From, To> && non_narrowing_list_initializable_v<To, From>;

	template<typename To, typename From>
	inline constexpr bool non_narrowing_constructible_v =
		std::is_constructible_v<To, From> && non_narrowing_list_initializable_v<To, From>;

	/// @brief Overload-set element used to ask "which alternative would overload resolution choose?".
	/// @details This is part of the converting-constructor/assignment ranking model. If `Source`
	///          can be converted to `T` without narrowing, this overload contributes alternative `I`.
	template<std::size_t I, typename T, typename Source, bool Enable = non_narrowing_convertible_v<T, Source&&>>
	struct construct_select_overload {
		void operator()() const = delete;
	};

	template<std::size_t I, typename T, typename Source>
	struct construct_select_overload<I, T, Source, true> {
		auto operator()(T) const -> std::integral_constant<std::size_t, I>;
	};

	/// @brief Aggregates one viable overload candidate per alternative type.
	template<typename Source, typename Seq, typename... Ts>
	struct construct_select_overload_set_impl;

	template<typename Source, std::size_t... Is, typename... Ts>
	struct construct_select_overload_set_impl<Source, std::index_sequence<Is...>, Ts...>
		: construct_select_overload<Is, Ts, Source>... {
		using construct_select_overload<Is, Ts, Source>::operator()...;
	};

	/// @brief Overload set modeling converting-constructor candidate selection for `Source`.
	template<typename Source, typename... Ts>
	using construct_select_overload_set =
		construct_select_overload_set_impl<Source, std::index_sequence_for<Ts...>, Ts...>;

	/// @brief Result tag naming the alternative chosen by overload resolution for converting construction.
	template<typename Source, typename... Ts>
	using construct_select_tag_t = decltype(std::declval<construct_select_overload_set<Source, Ts...>>()(std::declval<Source&&>()));

	/// @brief Selects the alternative index preferred by implicit overload resolution for constructing from `Source`.
	/// @details "Best" here means the alternative whose overload wins normal overload resolution
	///          among the viable non-narrowing implicit conversion candidates.
	template<typename Source, typename... Ts>
	struct implicit_best_construct_index {
	private:
		template<typename S, typename = void>
		struct impl : std::integral_constant<std::size_t, std::variant_npos> {};

		template<typename S>
		struct impl<S, std::void_t<construct_select_tag_t<S, Ts...>>>
			: std::integral_constant<std::size_t, construct_select_tag_t<S, Ts...>::value> {};
	public:
		static constexpr std::size_t value = impl<Source>::value;
	};

	template<typename Source, typename... Ts>
	inline constexpr std::size_t implicit_best_construct_index_v = implicit_best_construct_index<Source, Ts...>::value;

	/// @brief Selects the alternative index used by the converting constructor from `Source`.
	/// @details "Best" here means:
	/// - prefer the unique exact type match when one exists and is constructible
	/// - otherwise fall back to the implicit non-narrowing overload-resolution winner
	/// - otherwise report `std::variant_npos`
	template<typename Source, typename... Ts>
	struct best_construct_index {
	private:
		using U = std::remove_cv_t<std::remove_reference_t<Source>>;
		static constexpr std::size_t exact_count = count_exact_v<U, Ts...>;
		static constexpr std::size_t exact_index = index_of_exact_v<U, Ts...>;
		static constexpr std::size_t implicit_index = implicit_best_construct_index_v<Source, Ts...>;
	public:
		static constexpr std::size_t value = []() constexpr {
			if constexpr (exact_count > 1) {
				return std::variant_npos;
			} else if constexpr (exact_count == 1) {
				if constexpr (std::is_constructible_v<U, Source&&>) {
					return exact_index;
				} else {
					return std::variant_npos;
				}
			} else {
				// Portable baseline (intentional): converting ctor/operator= selection is implicit-only and
				// non-narrowing today because stdlib implementations differ on explicit-only converting behavior.
				// We intend to converge to the standard libraries' consensus once that behavior is stable/clear.
				return implicit_index;
			}
		}();
	};

	template<typename Source, typename... Ts>
	inline constexpr std::size_t best_construct_index_v = best_construct_index<Source, Ts...>::value;

	/// @brief Selects the alternative index used by converting assignment from `Source`.
	/// @details "Best" here means the converting-construction winner, filtered further by whether
	///          the selected alternative is actually assignable from `Source`.
	template<typename Source, typename... Ts>
	struct best_assign_index {
	private:
		static constexpr std::size_t selected_index = best_construct_index_v<Source, Ts...>;
	public:
		static constexpr std::size_t value = []() constexpr {
			if constexpr (selected_index == std::variant_npos) {
				return std::variant_npos;
			} else if constexpr (std::is_assignable_v<type_at_t<selected_index, Ts...>&, Source&&>) {
				return selected_index;
			} else {
				return std::variant_npos;
			}
		}();
	};

	template<typename Source, typename... Ts>
	inline constexpr std::size_t best_assign_index_v = best_assign_index<Source, Ts...>::value;

	/// @brief Reports whether the selected converting-construction path should be implicit.
	/// @details This answers "is the best converting-constructor candidate implicitly convertible?",
	///          which is used to split implicit vs explicit converting constructors.
	template<typename Source, typename... Ts>
	struct best_construct_is_implicit {
		static constexpr std::size_t I = best_construct_index_v<Source, Ts...>;
		static constexpr bool value = []() constexpr {
			if constexpr (I == std::variant_npos) {
				return false;
			} else {
				return std::is_convertible_v<Source&&, type_at_t<I, Ts...>>;
			}
		}();
	};

	template<typename Source, typename... Ts>
	inline constexpr bool best_construct_is_implicit_v = best_construct_is_implicit<Source, Ts...>::value;

	/// @brief Computes the aligned raw storage block large enough for all alternatives.
	template<typename... Ts>
	struct storage_traits {
		static constexpr std::size_t max_size = (std::max)({sizeof(Ts)...});
		static constexpr std::size_t max_align = (std::max)({alignof(Ts)...});

		struct storage_t {
			alignas(max_align) std::byte buffer[max_size];
		};
	};

	template<typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

	/// @brief True when every alternative is copy-constructible.
	template<typename... Ts>
	struct all_copy_constructible : std::conjunction<std::is_copy_constructible<Ts>...> {};

	template<typename... Ts>
	struct all_move_constructible : std::conjunction<std::is_move_constructible<Ts>...> {};

	template<typename... Ts>
	struct all_copy_assignable    : std::conjunction<std::is_copy_assignable<Ts>...> {};

	template<typename... Ts>
	struct all_move_assignable    : std::conjunction<std::is_move_assignable<Ts>...> {};

	template<typename... Ts>
	struct all_swappable          : std::conjunction<std::is_swappable<Ts>...> {};

	template<typename... Ts>
	struct all_nothrow_move_constructible
	                              : std::conjunction<std::is_nothrow_move_constructible<Ts>...> {};

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

	/// @brief Detects whether an encoded-index type exposes sideband access.
	template<typename T, typename = void>
	struct has_sideband : std::false_type {};

	template<typename T>
	struct has_sideband<T, std::void_t<decltype(std::declval<T&>().sideband())>> : std::true_type {};

	template<typename T>
	inline constexpr bool has_sideband_v = has_sideband<T>::value;

	/// @brief Concept for encoded-index storage with the noexcept API required by the variant core.
	template<typename EncodedIndex>
	concept encoded_index_noexcept_api =
		std::default_initializable<EncodedIndex> &&
		requires(EncodedIndex idx, const EncodedIndex cidx, std::size_t i) {
			{ cidx.index() } noexcept -> std::convertible_to<std::size_t>;
			{ idx.set_index(i) } noexcept -> std::same_as<void>;
		};

	template<template<std::size_t NTypes> class EncodedIndex, std::size_t NTypes>
	concept encoded_index_template_noexcept_api =
		encoded_index_noexcept_api<EncodedIndex<NTypes>>;

} // namespace custom_indexed_variant_detail

namespace detail = custom_indexed_variant_detail;

/// @brief Encoded-index policy family parameterized by the integer type used for the encoded word.
/// @tparam IndexT Unsigned integral word used to store the active index and optional sideband bits.
template<typename IndexT = std::size_t>
struct for_index_type {

	static_assert(std::is_integral_v<IndexT>,
			"IndexT must be an integral type");
	static_assert(std::is_unsigned_v<IndexT>,
			"IndexT must be an unsigned integral type");
	static_assert(!std::is_same_v<IndexT, bool>,
			"IndexT must not be bool");
	static_assert(std::numeric_limits<IndexT>::radix == 2,
			"IndexT must be base-2 (binary) integral");

	/// @brief Minimal encoded-index policy storing only the active alternative index.
	/// @tparam NTypes Number of variant alternatives.
	template<std::size_t NTypes>
	struct simple_encoded_index {

		/// @brief Encoded representation for std::variant_npos.
		static constexpr IndexT npos_code = ~IndexT(0);
		static_assert(npos_code >= NTypes); // Ensures npos_code is distinct from all other valid indicies

		/// @brief Constructs an index in the valueless state (std::variant_npos).
		simple_encoded_index() noexcept = default;

		/// @brief Returns the currently stored active index.
		std::size_t index() const noexcept {
			assert((index_ == npos_code) || (index_ < NTypes) || !"index_ is a valid value.");
			return (index_ == npos_code) ? std::variant_npos : static_cast<std::size_t>(index_);
		}

		/// @brief Sets the active index or std::variant_npos.
		/// @param val New index value in [0, NTypes) or std::variant_npos.
		void set_index(std::size_t val) noexcept {
			assert((val == std::variant_npos) || (val < NTypes) || !"val is a valid value.");
			index_ = (val == std::variant_npos) ? npos_code : static_cast<IndexT>(val);
			assert((index() == val) || !"index was set to expected value.");
		}

	private:
		IndexT index_ = npos_code;
	};

	/// @brief Encoded-index policy that stores both the active index and extra caller-managed sideband bits.
	/// @tparam NTypes Number of variant alternatives.
	/// @details The low bits encode the variant index and valueless state; the remaining bits are
	///          preserved sideband payload that the variant core itself does not interpret.
	template <std::size_t NTypes>
	struct index_encoded_with_sideband_data {
		static_assert(std::unsigned_integral<IndexT>, "IndexT must be an unsigned integral type");

		/// @brief Bit width of IndexT.
		static constexpr std::size_t width = std::numeric_limits<IndexT>::digits;

		/// @brief Total index states including valueless state.
		static constexpr std::size_t total_index_states = NTypes + 1;

		/// @brief Highest required index value (0-based) to represent all required states.
		/// Since we need values [0..NTypes-1] plus valueless, we need to represent NTypes as well.
		static constexpr std::size_t max_required_index = total_index_states - 1; // == NTypes

		/// @brief Bits needed to represent max_required_index.
		static constexpr std::size_t index_bits = std::bit_width(max_required_index);

		static_assert(index_bits <= width, "IndexT too small to encode required index states");
		static_assert(index_bits < width, "No room left for sideband payload; disallowed");

	private:
		enum field_index : std::size_t { INDEX = 0, SIDEBAND = 1 };

	public:
		/// @brief Underlying layout: [ index_bits | sideband(remainder) ].
		using bits_t = bitfield_pack<IndexT, field_index, bitfield_field_spec<index_bits>, bitfield_remainder>;

		/// @brief Encoded representation for std::variant_npos: all ones in the index field.
		static constexpr IndexT npos_code = bits_t::template field_max_bits<INDEX>();
		static_assert(npos_code >= NTypes, "npos_code must be distinct from valid indices");

		/// @brief Mutable proxy for reading or updating sideband bits without exposing index internals.
		struct sideband_proxy {
			sideband_proxy() = delete;
			explicit sideband_proxy(index_encoded_with_sideband_data* src) noexcept : data_(src) {}

			/// @brief Returns the current sideband payload bits.
			IndexT val() const noexcept { return data_->sideband_val(); }
			/// @brief Stores a sideband payload value, preserving the encoded index bits.
			void set_val(IndexT v) noexcept { data_->set_sideband_val(v); }
			/// @brief Validates that a sideband value fits before masking.
			void validate_val(IndexT v) noexcept { data_->validate_sideband_val(v); }

		private:
			index_encoded_with_sideband_data* data_;
		};

		/// @brief Const proxy for observing sideband bits while preserving constness.
		struct const_sideband_proxy {
			const_sideband_proxy() = delete;
			explicit const_sideband_proxy(const index_encoded_with_sideband_data* src) noexcept : data_(src) {}
			IndexT val() const noexcept { return data_->sideband_val(); }
		private:
			const index_encoded_with_sideband_data* data_;
		};

		using sideband_t = sideband_proxy;
		using const_sideband_t = const_sideband_proxy;

		/// @brief Constructs encoded state with `std::variant_npos` and zero sideband payload.
		constexpr index_encoded_with_sideband_data() noexcept {
			bits_.set_raw_storage(0);
			bits_.template set_bits<INDEX>(npos_code);
		}

		/// @brief Returns decoded active index.
		constexpr std::size_t index() const BITFIELD_PACK_NOEXCEPT {
			const IndexT stored = bits_.template get_bits<INDEX>();
			if (stored == npos_code) return std::variant_npos;
			// Values in [NTypes .. npos_code-1] are invalid.
			// As internal machinery, we assert this invariant.
			BITFIELD_PACK_ASSERT(stored < NTypes);
			return static_cast<std::size_t>(stored);
		}

		/// @brief Writes encoded active index while preserving sideband.
		constexpr void set_index(std::size_t v) BITFIELD_PACK_NOEXCEPT {
			BITFIELD_PACK_ASSERT((v == std::variant_npos) || (v < NTypes));
#ifndef BITFIELD_PACK_NDEBUG
			const IndexT prev_sideband = sideband_val();
#endif

			const IndexT stored = (v == std::variant_npos) ? npos_code : IndexT(v);
			bits_.template set_bits<INDEX>(stored);

			BITFIELD_PACK_ASSERT(sideband_val() == prev_sideband);
			BITFIELD_PACK_ASSERT(index() == v);
		}

		/// @brief Returns an explicit sideband proxy.
		constexpr sideband_t sideband() noexcept { return sideband_t(this); }
		constexpr const_sideband_t sideband() const noexcept { return const_sideband_t(this); }

		/// @brief Exposes the complete encoded word for tests and low-level adapters.
		constexpr IndexT raw_storage() const noexcept { return bits_.raw_storage(); }
		constexpr void set_raw_storage(IndexT v) noexcept { bits_.set_raw_storage(v); }

	private:
		constexpr IndexT sideband_val() const noexcept { return bits_.template get_bits<SIDEBAND>(); }

		constexpr void set_sideband_val(IndexT v) noexcept {
			// setter masks; validity is optional
			bits_.template set_bits<SIDEBAND>(v);
		}

		constexpr void validate_sideband_val(IndexT v) noexcept {
			// If you care, validate that v fits in the remainder width.
			// (This is optional; setter masks anyway.)
			constexpr IndexT maxv = bits_t::template field_max_bits<SIDEBAND>();
			BITFIELD_PACK_ASSERT(v <= maxv);
		}

		bits_t bits_{};
	};
};

template<std::size_t NTypes>
using simple_encoded_index = for_index_type<std::size_t>::simple_encoded_index<NTypes>;

template<std::size_t NTypes>
using index_encoded_with_sideband_data = for_index_type<std::size_t>::index_encoded_with_sideband_data<NTypes>;

/// @brief Variant-like discriminated union with pluggable encoded-index storage.
/// @tparam EncodedIndex Template that chooses how the active index is encoded.
/// @tparam Types Alternative types stored in the variant.
/// @details The active alternative lives in raw aligned storage and is tracked by `encoded_index_t`.
///          Depending on the encoded-index policy, the index word may also carry sideband bits that
///          are preserved across ordinary variant operations.
/// @note This implementation intentionally prefers a unique exact-type match before falling back to
///       the implicit non-narrowing overload-resolution winner for converting construction/assignment.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
	requires detail::encoded_index_template_noexcept_api<EncodedIndex, sizeof...(Types)>
class custom_indexed_variant {
	static_assert(sizeof...(Types) > 0, "custom_indexed_variant must have at least one alternative");
	static_assert((std::is_nothrow_destructible_v<Types> && ...),
		"custom_indexed_variant requires nothrow-destructible alternatives");
	using storage_traits = detail::storage_traits<Types...>;
	static constexpr std::size_t ntypes = sizeof...(Types);

public:
	/// @brief Encoded index storage policy bound to this variant arity.
	using encoded_index_t = EncodedIndex<ntypes>;
	static_assert(detail::encoded_index_noexcept_api<encoded_index_t>,
		"EncodedIndex<NTypes> must provide noexcept default ctor, noexcept index() -> size_t, and noexcept set_index(size_t).");
	/// @brief Sentinel value indicating the valueless state.
	static constexpr std::size_t npos = std::variant_npos;

	/// @brief Public index type used by index().
	using index_type = std::size_t;

	/// @brief Default-constructs the first alternative.
	custom_indexed_variant() noexcept(std::is_nothrow_default_constructible_v<detail::type_at_t<0, Types...>>)
		requires std::is_default_constructible_v<detail::type_at_t<0, Types...>>
		: index_obj_() {
		construct_active_impl<0>();
	}

	/// @brief Copy-constructs from another variant.
	custom_indexed_variant(const custom_indexed_variant& other)
		requires detail::all_copy_constructible<Types...>::value
		: index_obj_() {
		copy_construct_from_impl(other);
	}

	/// @brief Move-constructs from another variant.
	custom_indexed_variant(custom_indexed_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
		requires detail::all_move_constructible<Types...>::value
		: index_obj_() {
		move_construct_from_impl(other);
	}

	/// @brief Constructs the specified alternative in-place by index.
	template<std::size_t I, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_index_t<I>, Args&&... args)
		: index_obj_() {
		construct_active_impl<I>(std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_index_t<I>, std::initializer_list<U> init, Args&&... args)
		: index_obj_() {
		construct_active_impl<I>(init, std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by type.
	template<typename T, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_type_t<T>, Args&&... args)
		: index_obj_() {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t index = detail::index_of_exact_v<T, Types...>;
		static_assert(index != npos, "Type not found in custom_indexed_variant");
		construct_active_impl<index>(std::forward<Args>(args)...);
	}

	/// @brief Constructs the specified alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	constexpr explicit custom_indexed_variant(std::in_place_type_t<T>, std::initializer_list<U> init, Args&&... args)
		: index_obj_() {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t index = detail::index_of_exact_v<T, Types...>;
		static_assert(index != npos, "Type not found in custom_indexed_variant");
		construct_active_impl<index>(init, std::forward<Args>(args)...);
	}

	/// @brief Converting constructor from a value (selected alternative; implicit when conversion is implicit).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::size_t I = detail::best_construct_index_v<T, Types...>,
			std::enable_if_t<!std::is_same_v<U, custom_indexed_variant> &&
							!detail::is_in_place_index_v<U> &&
							!detail::is_in_place_type_v<U> &&
							(I != npos) &&
							detail::best_construct_is_implicit_v<T, Types...>, int> = 0>
	constexpr custom_indexed_variant(T&& value)
		: index_obj_() {
		construct_active_impl<I>(std::forward<T>(value));
	}

	/// @brief Converting constructor from a value (selected alternative; explicit when needed).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::size_t I = detail::best_construct_index_v<T, Types...>,
			std::enable_if_t<!std::is_same_v<U, custom_indexed_variant> &&
							!detail::is_in_place_index_v<U> &&
							!detail::is_in_place_type_v<U> &&
							(I != npos) &&
							!detail::best_construct_is_implicit_v<T, Types...>, int> = 0>
	constexpr explicit custom_indexed_variant(T&& value)
		: index_obj_() {
		construct_active_impl<I>(std::forward<T>(value));
	}

	/// @brief Destroys the active alternative if engaged.
	~custom_indexed_variant() {
		destroy_active_impl();
	}

	/// @brief Copy-assigns from another variant.
	custom_indexed_variant& operator=(const custom_indexed_variant& other)
		requires detail::all_copy_constructible<Types...>::value && detail::all_copy_assignable<Types...>::value {
		if (this == &other) {
			return *this;
		}
		return assign_from_variant_impl(other);
	}

	/// @brief Move-assigns from another variant.
	custom_indexed_variant& operator=(custom_indexed_variant&& other) noexcept(detail::all_nothrow_move_constructible<Types...>::value)
		requires detail::all_move_constructible<Types...>::value && detail::all_move_assignable<Types...>::value {
		if (this == &other) {
			return *this;
		}
		return assign_from_variant_impl(std::move(other));
	}

	/// @brief Assigns from a value (selected alternative).
	template<typename T,
			typename U = detail::remove_cvref_t<T>,
			std::size_t I = detail::best_assign_index_v<T, Types...>,
			std::enable_if_t<!std::is_same_v<U, custom_indexed_variant> &&
							!detail::is_in_place_index_v<U> &&
							!detail::is_in_place_type_v<U> &&
							(I != npos), int> = 0>
	custom_indexed_variant& operator=(T&& value) {
		assign_value_impl<I>(std::forward<T>(value));
		return *this;
	}

	/// @brief Reports whether the variant currently has no active alternative.
	bool valueless_by_exception() const noexcept { return is_valueless_impl(); }

	/// @brief Returns the active alternative index or `std::variant_npos`.
	std::size_t index() const noexcept { return current_index_impl(); }

	/// @brief Exposes encoded-index sideband when the selected policy provides it.
	/// @details This is intentionally absent for index policies that do not carry sideband bits.
	template<typename EI = encoded_index_t, typename = std::enable_if_t<detail::has_sideband_v<EI>>>
	auto sideband() noexcept(noexcept(std::declval<EI&>().sideband())) {
		return sideband_impl();
	}

	template<typename EI = encoded_index_t, typename = std::enable_if_t<detail::has_sideband_v<EI>>>
	auto sideband() const noexcept(noexcept(std::declval<const EI&>().sideband())) {
		return sideband_impl();
	}

	/// @brief Constructs a new alternative in-place by index.
	template<std::size_t I, typename... Args>
	decltype(auto) emplace(Args&&... args) {
		emplace_impl<I>(std::forward<Args>(args)...);
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by index with initializer_list.
	template<std::size_t I, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		emplace_impl<I>(init, std::forward<Args>(args)...);
		return get<I>();
	}

	/// @brief Constructs a new alternative in-place by type.
	template<typename T, typename... Args>
	decltype(auto) emplace(Args&&... args) {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return emplace<I>(std::forward<Args>(args)...);
	}

	/// @brief Constructs a new alternative in-place by type with initializer_list.
	template<typename T, typename U, typename... Args>
	decltype(auto) emplace(std::initializer_list<U> init, Args&&... args) {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return emplace<I>(init, std::forward<Args>(args)...);
	}

	/// @brief Swaps this variant with another.
	void swap(custom_indexed_variant& other)
		requires detail::all_swappable<Types...>::value && detail::all_move_constructible<Types...>::value {
		swap_impl(other);
	}

	/// @brief Access the active alternative by index.
	template<std::size_t I>
		decltype(auto) get() & {
			return get_ref_impl<I>(*this);
		}

	/// @brief Access the active alternative by index (const).
	template<std::size_t I>
		decltype(auto) get() const & {
			return get_ref_impl<I>(*this);
		}

	/// @brief Access the active alternative by index (rvalue).
	template<std::size_t I>
		decltype(auto) get() && {
			return std::move(get_ref_impl<I>(*this));
		}

	/// @brief Access the active alternative by index (const rvalue).
	/// @note This is kept for std::variant API parity; moving from const usually performs a copy (or is ill-formed).
	template<std::size_t I>
		decltype(auto) get() const && {
			return std::move(get_ref_impl<I>(*this));
		}

	/// @brief Access the active alternative by type.
	template<typename T>
	decltype(auto) get() & {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (const).
	template<typename T>
	decltype(auto) get() const & {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get<I>();
	}

	/// @brief Access the active alternative by type (rvalue).
	template<typename T>
	decltype(auto) get() && {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return std::move(get<I>());
	}

	/// @brief Access the active alternative by type (const rvalue).
	/// @note This is kept for std::variant API parity; moving from const usually performs a copy (or is ill-formed).
	template<typename T>
	decltype(auto) get() const && {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return std::move(get<I>());
	}

	/// @brief Pointer access to alternative by index.
	template<std::size_t I>
		auto get_if() noexcept {
			return get_if_impl<I>(*this);
		}

	/// @brief Pointer access to alternative by index (const).
	template<std::size_t I>
		auto get_if() const noexcept {
			return get_if_impl<I>(*this);
		}

	/// @brief Pointer access to alternative by type.
	template<typename T>
	auto get_if() noexcept {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get_if<I>();
	}

	/// @brief Pointer access to alternative by type (const).
	template<typename T>
	auto get_if() const noexcept {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return get_if<I>();
	}

	/// @brief Returns true if the active alternative matches T.
	template<typename T>
	bool holds_alternative() const noexcept {
		static_assert(detail::count_exact_v<T, Types...> == 1, "Type must occur exactly once in custom_indexed_variant");
		constexpr std::size_t I = detail::index_of_exact_v<T, Types...>;
		static_assert(I != npos, "Type not found in custom_indexed_variant");
		return current_index_impl() == I;
	}

private:
	// Representation inspection.
	/// @brief Returns the encoded active index as interpreted by the index policy.
	std::size_t current_index_impl() const noexcept { return index_obj_.index(); }
	/// @brief Returns true when no alternative is currently engaged.
	bool is_valueless_impl() const noexcept { return current_index_impl() == npos; }

	/// @brief Forwards explicit sideband access to the encoded-index policy.
	template<typename Self>
	static decltype(auto) sideband_impl(Self&& self) noexcept(noexcept(std::forward<Self>(self).index_obj_.sideband())) {
		return std::forward<Self>(self).index_obj_.sideband();
	}

	decltype(auto) sideband_impl() noexcept(noexcept(index_obj_.sideband())) {
		return sideband_impl(*this);
	}

	decltype(auto) sideband_impl() const noexcept(noexcept(index_obj_.sideband())) {
		return sideband_impl(*this);
	}

	// Storage access.
	/// @brief Returns the raw storage buffer used for the active alternative.
	std::byte* storage_bytes_impl() noexcept { return storage_.buffer; }
	const std::byte* storage_bytes_impl() const noexcept { return storage_.buffer; }

	/// @brief Placement-constructs alternative `I` into raw storage and publishes its index.
	template<std::size_t I, typename... Args>
	void construct_active_impl(Args&&... args) {
		assert(is_valueless_impl() && "construct requires valueless state");
		using T = detail::type_at_t<I, Types...>;
		::new (static_cast<void*>(storage_bytes_impl())) T(std::forward<Args>(args)...);
		index_obj_.set_index(I);
		assert(current_index_impl() == I && "construct must set active index exactly once after full construction");
	}

	/// @brief Destroys the active alternative and transitions to valueless.
	/// @note This function is idempotent: calling it while already valueless is a no-op.
	void destroy_active_impl() noexcept {
		const std::size_t active = current_index_impl();
		if (active == npos) {
			return;
		}
		assert(active < ntypes && "active index out of bounds in destroy_active");
		destroy_active_case_impl<0>(active);
		assert(current_index_impl() == active && "destroy_active must not change index before final npos transition");
		index_obj_.set_index(npos);
		assert(is_valueless_impl() && "destroy_active must transition to npos");
	}

	/// @brief Returns a typed pointer to alternative `I` within a variant's raw storage.
	template<std::size_t I, typename Variant>
	static auto active_ptr_impl(Variant& variant) noexcept {
		using VariantNoRef = std::remove_reference_t<Variant>;
		using Base = detail::type_at_t<I, Types...>;
		using CvBase = std::conditional_t<std::is_const_v<VariantNoRef>, std::add_const_t<Base>, Base>;
		using CvVolBase = std::conditional_t<std::is_volatile_v<VariantNoRef>, std::add_volatile_t<CvBase>, CvBase>;
		return std::launder(reinterpret_cast<CvVolBase*>(variant.storage_.buffer));
	}

	// Active-alternative dispatch.
	/// @brief Invokes `fn(index_tag, value)` for the currently active alternative.
	template<typename Variant, typename Fn>
	static void visit_active_impl(Variant&& variant, Fn&& fn) {
		visit_active_case_impl<0>(std::forward<Variant>(variant), std::forward<Fn>(fn));
	}

	template<std::size_t I, typename Variant, typename Fn>
	static void visit_active_case_impl(Variant&& variant, Fn&& fn) {
		if constexpr (I < sizeof...(Types)) {
			switch (variant.current_index_impl()) {
			case I:
				fn(std::integral_constant<std::size_t, I>{}, *active_ptr_impl<I>(variant));
				return;
			default:
				visit_active_case_impl<I + 1>(std::forward<Variant>(variant), std::forward<Fn>(fn));
				return;
			}
		}
	}

	/// @brief Destroys whichever alternative matches the runtime active index.
	template<std::size_t I>
	void destroy_active_case_impl(std::size_t active) noexcept {
		if constexpr (I < sizeof...(Types)) {
			using T = detail::type_at_t<I, Types...>;
			switch (active) {
			case I:
				assert(current_index_impl() == I && "destroy_active_impl index mismatch");
				active_ptr_impl<I>(*this)->~T();
				return;
			default:
				destroy_active_case_impl<I + 1>(active);
				return;
			}
		}
	}

	/// @brief Returns the active alternative by reference, throwing on index mismatch.
	template<std::size_t I, typename Variant>
	static decltype(auto) get_ref_impl(Variant&& variant) {
		if (variant.current_index_impl() != I) {
			throw std::bad_variant_access{};
		}
		return *active_ptr_impl<I>(variant);
	}

	/// @brief Returns a pointer to alternative `I` when active, otherwise `nullptr`.
	template<std::size_t I, typename Variant>
	static auto get_if_impl(Variant& variant) noexcept {
		return (variant.current_index_impl() == I) ? active_ptr_impl<I>(variant) : nullptr;
	}

	// Transition helpers.
	/// @brief Replaces the active alternative with `I`, preserving strong behavior where construction permits.
	template<std::size_t I, typename... Args>
	void emplace_impl(Args&&... args) {
		using T = detail::type_at_t<I, Types...>;
		if constexpr (std::is_nothrow_move_constructible_v<T> || std::is_nothrow_copy_constructible_v<T>) {
			std::optional<T> temp;
			temp.emplace(std::forward<Args>(args)...);
			destroy_active_impl();
			if constexpr (std::is_nothrow_move_constructible_v<T>) {
				construct_active_impl<I>(std::move(*temp));
			} else {
				construct_active_impl<I>(*temp);
			}
		} else {
			destroy_active_impl();
			construct_active_impl<I>(std::forward<Args>(args)...);
		}
	}

	/// @brief Assigns from a non-variant value using the preselected target alternative `I`.
	template<std::size_t I, typename Value>
	void assign_value_impl(Value&& value) {
		using Target = detail::type_at_t<I, Types...>;
		if (current_index_impl() == I) {
			get_ref_impl<I>(*this) = std::forward<Value>(value);
			return;
		}
		if constexpr (std::is_nothrow_move_constructible_v<Target> || std::is_nothrow_copy_constructible_v<Target>) {
			std::optional<Target> temp;
			temp.emplace(std::forward<Value>(value));
			destroy_active_impl();
			if constexpr (std::is_nothrow_move_constructible_v<Target>) {
				construct_active_impl<I>(std::move(*temp));
			} else {
				construct_active_impl<I>(*temp);
			}
		} else {
			destroy_active_impl();
			construct_active_impl<I>(std::forward<Value>(value));
		}
	}

	/// @brief Copy-constructs from another engaged variant, preserving valueless state otherwise.
	void copy_construct_from_impl(const custom_indexed_variant& other) {
		if (!other.valueless_by_exception()) {
			visit_active_impl(other, [&](auto index_tag, const auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct_active_impl<I>(value);
			});
		}
	}

	/// @brief Move-constructs from another engaged variant, preserving valueless state otherwise.
	void move_construct_from_impl(custom_indexed_variant& other) {
		if (!other.valueless_by_exception()) {
			visit_active_impl(other, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct_active_impl<I>(std::move(value));
			});
		}
	}

	/// @brief Assigns from another variant, either reusing the current alternative or replacing it.
	template<typename Other>
	custom_indexed_variant& assign_from_variant_impl(Other&& other) {
		if (other.valueless_by_exception()) {
			destroy_active_impl();
			return *this;
		}
		if (current_index_impl() == other.current_index_impl()) {
			visit_active_impl(*this, [&](auto index_tag, auto& value) {
				value = std::forward<Other>(other).template get<decltype(index_tag)::value>();
			});
			return *this;
		}
		destroy_active_impl();
		visit_active_impl(other, [&](auto index_tag, auto&& value) {
			constexpr std::size_t I = decltype(index_tag)::value;
			construct_active_impl<I>(std::forward<decltype(value)>(value));
		});
		return *this;
	}

	/// @brief Swaps payloads when both variants hold the same alternative index.
	void swap_same_index_impl(custom_indexed_variant& other) {
		visit_active_impl(*this, [&](auto index_tag, auto& value) {
			using std::swap;
			swap(value, other.template get<decltype(index_tag)::value>());
		});
	}

	/// @brief Swaps payloads when the variants hold different alternatives.
	/// @details This route stages both values in temporaries so the raw storage can be torn down
	///          and reconstructed with the opposite active alternatives.
	template<std::size_t L, std::size_t R>
	void swap_different_index_impl(custom_indexed_variant& other) {
		using Left = detail::type_at_t<L, Types...>;
		using Right = detail::type_at_t<R, Types...>;
		std::optional<Left> left_value;
		std::optional<Right> right_value;
		left_value.emplace(std::move(*active_ptr_impl<L>(*this)));
		right_value.emplace(std::move(*active_ptr_impl<R>(other)));
		destroy_active_impl();
		other.destroy_active_impl();
		bool left_constructed = false;
		try {
			construct_active_impl<R>(std::move(*right_value));
			left_constructed = true;
		} catch (...) {
			throw;
		}
		try {
			other.template construct_active_impl<L>(std::move(*left_value));
		} catch (...) {
			if (left_constructed) {
				destroy_active_impl();
			}
			throw;
		}
	}

	/// @brief Top-level swap state machine handling valueless, same-index, and different-index cases.
	void swap_impl(custom_indexed_variant& other)
		requires detail::all_swappable<Types...>::value && detail::all_move_constructible<Types...>::value {
		if (this == &other) {
			return;
		}
		if (valueless_by_exception() && other.valueless_by_exception()) {
			return;
		}
		if (current_index_impl() == other.current_index_impl()) {
			swap_same_index_impl(other);
			return;
		}
		if (valueless_by_exception()) {
			bool constructed = false;
			visit_active_impl(other, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				construct_active_impl<I>(std::move(value));
				constructed = true;
			});
			if (constructed) {
				other.destroy_active_impl();
			}
			return;
		}
		if (other.valueless_by_exception()) {
			bool constructed = false;
			visit_active_impl(*this, [&](auto index_tag, auto& value) {
				constexpr std::size_t I = decltype(index_tag)::value;
				other.template construct_active_impl<I>(std::move(value));
				constructed = true;
			});
			if (constructed) {
				destroy_active_impl();
			}
			return;
		}
		visit_active_impl(*this, [&](auto left_tag, auto&) {
			constexpr std::size_t L = decltype(left_tag)::value;
			visit_active_impl(other, [&](auto right_tag, auto&) {
				constexpr std::size_t R = decltype(right_tag)::value;
				swap_different_index_impl<L, R>(other);
			});
		});
	}

	[[no_unique_address]] encoded_index_t index_obj_;
	typename storage_traits::storage_t storage_{};
};

/// @brief Variant size trait for custom_indexed_variant.
template<typename Variant>
struct variant_size;

/// @brief variant_size specialization for custom_indexed_variant.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

/// @brief variant_size specialization for const custom_indexed_variant.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<const custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

/// @brief variant_size specialization for volatile custom_indexed_variant.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<volatile custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

/// @brief variant_size specialization for const volatile custom_indexed_variant.
template<template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_size<const volatile custom_indexed_variant<EncodedIndex, Types...>> : std::integral_constant<std::size_t, sizeof...(Types)> {};

/// @brief Convenience value wrapper for variant_size<Variant>::value.
template<typename Variant>
inline constexpr std::size_t variant_size_v = variant_size<Variant>::value;

/// @brief Variant alternative trait for custom_indexed_variant.
template<std::size_t I, typename Variant>
struct variant_alternative;

/// @brief variant_alternative specialization for custom_indexed_variant.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>> {
	static_assert(I < sizeof...(Types), "variant alternative index out of bounds");
	using type = custom_indexed_variant_detail::type_at_t<I, Types...>;
};

/// @brief variant_alternative specialization for const custom_indexed_variant.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, const custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_const_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

/// @brief variant_alternative specialization for volatile custom_indexed_variant.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, volatile custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_volatile_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

/// @brief variant_alternative specialization for const volatile custom_indexed_variant.
template<std::size_t I, template<std::size_t NTypes> class EncodedIndex, typename... Types>
struct variant_alternative<I, const volatile custom_indexed_variant<EncodedIndex, Types...>> {
	using type = std::add_cv_t<typename variant_alternative<I, custom_indexed_variant<EncodedIndex, Types...>>::type>;
};

/// @brief Convenience alias for variant_alternative<I, Variant>::type.
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

	template<typename Visitor, typename CollectedTuple, typename... Variants>
	struct visit_invocable_impl;

	template<typename Visitor, typename... Collected>
	struct visit_invocable_impl<Visitor, std::tuple<Collected...>> : std::bool_constant<std::is_invocable_v<Visitor, Collected...>> {};

	template<typename Visitor, typename... Collected, typename Variant, typename... Rest>
	struct visit_invocable_impl<Visitor, std::tuple<Collected...>, Variant, Rest...> {
	private:
		using V = std::remove_reference_t<Variant>;

		template<std::size_t... Is>
		static consteval bool compute(std::index_sequence<Is...>) {
			return (visit_invocable_impl<
				Visitor,
				std::tuple<Collected..., decltype(get<Is>(std::declval<Variant>()))>,
				Rest...>::value && ...);
		}
	public:
		static constexpr bool value = compute(std::make_index_sequence<variant_size_v<V>>{});
	};

	template<typename Visitor, typename... Variants>
	inline constexpr bool visit_invocable_v = visit_invocable_impl<Visitor, std::tuple<>, Variants...>::value;

	/// @brief Dispatches one variant by runtime index and forwards the selected alternative to fn.
	template<std::size_t I = 0, typename Variant, typename Fn>
	auto dispatch_one(Variant&& variant, Fn&& fn)
		-> decltype(std::declval<Fn>()(get<0>(std::declval<Variant>()))) {
		using V = std::remove_reference_t<Variant>;
		if constexpr (I < variant_size_v<V>) {
			if (variant.index() == I) {
				return std::forward<Fn>(fn)(get<I>(std::forward<Variant>(variant)));
			}
			return dispatch_one<I + 1>(std::forward<Variant>(variant), std::forward<Fn>(fn));
		}
		throw std::bad_variant_access{};
	}

	/// @brief Collects active alternatives from all variants and invokes the visitor with preserved value categories.
	template<typename Visitor, typename Tuple, typename Variant, typename... Rest>
	decltype(auto) visit_collect(Visitor&& vis, Tuple&& collected, Variant&& variant, Rest&&... rest) {
		return dispatch_one(std::forward<Variant>(variant), [&](auto&& value) -> decltype(auto) {
			auto next = std::tuple_cat(
				std::forward<Tuple>(collected),
				std::forward_as_tuple(std::forward<decltype(value)>(value)));
			if constexpr (sizeof...(Rest) == 0) {
				return std::apply(
					[&](auto&&... args) -> decltype(auto) {
						return std::invoke(std::forward<Visitor>(vis), std::forward<decltype(args)>(args)...);
					},
					std::move(next));
			} else {
				return visit_collect(
					std::forward<Visitor>(vis),
					std::move(next),
					std::forward<Rest>(rest)...);
			}
		});
	}

} // namespace custom_indexed_variant_detail

/// @brief Visit the active alternative(s) with a callable.
template<typename Visitor, typename... Variants>
	requires (sizeof...(Variants) > 0) && custom_indexed_variant_detail::visit_invocable_v<Visitor&&, Variants&&...>
inline decltype(auto) visit(Visitor&& vis, Variants&&... variants) {
	if ((variants.valueless_by_exception() || ...)) {
		throw std::bad_variant_access{};
	}
	return custom_indexed_variant_detail::visit_collect(
		std::forward<Visitor>(vis),
		std::tuple<>{},
		std::forward<Variants>(variants)...);
}

} // namespace internal

}} // namespace sw::universal
