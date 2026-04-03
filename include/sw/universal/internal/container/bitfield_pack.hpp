// bitfield_pack.hpp
#pragma once
/// @file bitfield_pack.hpp
/// @brief Pack multiple bitfields into a single machine word with compile-time layout.
///
/// Design notes (locked down):
/// - Two entry points:
///   * bitfield_pack<Word, IndexingSpec, FieldSpecs...> : "power" form (spec types)
///   * bitfield_pack_bits<Word, Widths...> : ergonomic alias (pure widths)
/// - "Word" can be:
///   * an unsigned integral type (shorthand; `formatted_value()` returns the same type)
///   * bitfield_word_spec<UnderlyingValueT, FormattedValueT>
///     (`formatted_value()` returns `FormattedValueT`, bit math uses `UnderlyingValueT`)
/// - "IndexingSpec" can be:
///   * `std::size_t` for plain numeric field keys
///   * a raw enum type for named contiguous fields
///   * a custom descriptor with `field_key` and `to_index()` for sparse/reordered field naming
/// - Setters mask and store (silent truncation). Validity is separate:
///   * is_valid<I>(value)
///   * validate<I>(value) -> assertion hook
///   This is intentional: `set_masked()` is a masked store, not a checked transaction.
///   Callers that need a precondition check should use `is_valid()` or `validate()`
///   before storing, or `set_if_valid()` when a checked write is more convenient.
/// - Mutation APIs perform a whole-word load/modify/store through the word spec hooks.
///   They do not provide conditional publication or CAS-loop semantics.
/// - Remainder field supported, but if present it MUST be the final field.
/// - No "biased" codec is shipped as live code here; the spec protocol supports it later.

#include <bit>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <tuple>
#include <utility>


#ifndef BITFIELD_PACK_ASSERT
/// @brief Assertion hook.
/// @details Override before including this header if tests or embedding code need a different
///          assertion behavior. Normal builds inherit plain `assert(...)`.
#define BITFIELD_PACK_ASSERT(expr) assert(expr)
#endif

#ifndef BITFIELD_PACK_NOEXCEPT
/// @brief `noexcept` annotation hook.
/// @details Tests may override this to empty so assertion hooks can throw through
///          code paths that are `noexcept` in normal builds. If not overridden, the header
///          uses ordinary `noexcept`.
#define BITFIELD_PACK_NOEXCEPT noexcept
#endif // ndef BITFIELD_PACK_NOEXCEPT

namespace sw::universal {

/// @example
/// @code
/// // These are field indices:
/// enum class ieee754_f32_field : std::size_t {
///     mantissa       =    0,
///     exponent       =    1,
///     sign           =    2
/// };
/// // These are for interpreting exponent:
/// enum class ieee754_f32_exponent_detail : signed int {
///     exponent__bias =  127,
///     // These are only needed to compute...
///     exponent__bits =    8,
///     prebiased_min  =    0,
///     prebiased_max  =  (1 << exponent__bits) - 1,
///     // ...these:
///     subnormal      =  prebiased_min - exponent__bias,
///     inf_or_nan     =  prebiased_max - exponent__bias,
/// };
///
/// template <std::size_t Width, typename DecodedT, DecodedT Bias>
/// struct biased_bitfield_field_width {
///    template<typename StorageT>
///    struct for_storage_t : public bitfield_field_width<Width, DecodedT>::template for_storage_t<StorageT> {
///        static constexpr StorageT encode(DecodedT v) noexcept {
///            return static_cast<StorageT>(v + Bias);
///        }
///        static constexpr DecodedT decode(StorageT v) noexcept {
///            return (static_cast<DecodedT>(v) - Bias);
///        }
///    };
/// };
///
/// using ieee754_f32_bits = bitfield_pack<
///     bitfield_word_spec<std::uint32_t, float>,
///     ieee754_f32_field,
///     bitfield_remainder,            // mantissa (remainder == (32 - (8+1)) == 23)
///     biased_bitfield_field_width<
///         // The next line would normally just be explicitly `8,` except we
///         // already defined it and want a single source of truth:
///         static_cast<std::size_t>(ieee754_f32_exponent_detail::exponent__bits),
///         signed int,
///         static_cast<signed int>(ieee754_f32_exponent_detail::exponent__bias)
///     >,                             // exponent
///     bitfield_field_width<1, bool>  // sign
/// >;
///
/// ieee754_f32_bits bits;
/// bits.set_formatted_value(1.0f);
///
/// auto mantissa = bits.template get_bits<ieee754_f32_field::mantissa>();
/// auto exponent = bits.template get_bits<ieee754_f32_field::exponent>();
/// auto sign = bits.template get<ieee754_f32_field::sign>();
///
/// // for 1.0f:
/// // (0 ? -1 : 1) * (1.0 + (0 / 2^23)) * 2^0
/// //  ^ sign                ^ mantissa     ^ exponent
/// assert(mantissa == 0);
/// assert(exponent == 0);
/// assert(sign     == 0);
///
/// // for 3.1415927f:
/// // (0 ? -1 : 1) * (1.0 + (4788187 / 2^23)) * 2^1
/// //  ^ sign                ^ mantissa           ^ exponent
/// bits.template set_masked<ieee754_f32_field::mantissa>(4788187);
/// bits.template set_masked<ieee754_f32_field::exponent>(1);
/// bits.template set_masked<ieee754_f32_field::sign>(0);
///
/// float round_trip = bits.formatted_value();
/// assert(std::abs(round_trip - 3.1415927f) < 0.000001);
/// @endcode

namespace bitfield_pack_detail {

/// @brief Canonical description of how a bitfield word is stored, exposed, and loaded.
/// @tparam UnderlyingValueT Unsigned integral whole-value type used for masking, shifting, and layout math.
/// @tparam FormattedValueT Whole-pack API type exposed by `formatted_value()` / `set_formatted_value()`.
/// @tparam StorageT Actual contained backend object type stored by `bitfield_pack`.
/// @details `underlying_val_t` is always the canonical unsigned representation used for bit math.
///          `formatted_val_t` is the public whole-pack value type.
///          `storage_t` is the owned backend object and may differ from both when callers want
///          custom load/store hooks.
template <class UnderlyingValueT, class FormattedValueT = UnderlyingValueT, class StorageT = UnderlyingValueT>
struct bitfield_word_spec {
	static_assert(std::unsigned_integral<UnderlyingValueT>, "UnderlyingValueT must be unsigned integral");
	static_assert(std::is_trivially_copyable_v<FormattedValueT>, "FormattedValueT must be trivially copyable");
	static_assert(sizeof(FormattedValueT) == sizeof(UnderlyingValueT),
	              "FormattedValueT must have same size as UnderlyingValueT");

	using underlying_val_t = UnderlyingValueT;
	using formatted_val_t = FormattedValueT;
	using storage_t = StorageT;
	static constexpr bool directly_mutable = true;

	/// @brief Converts a public whole-pack value into the canonical underlying representation.
	static constexpr underlying_val_t to_underlying_value(formatted_val_t v) noexcept {
		if constexpr (std::is_same_v<formatted_val_t, underlying_val_t>) {
			return v;
		} else {
			return std::bit_cast<underlying_val_t>(v);
		}
	}

	/// @brief Converts the canonical underlying representation into the public whole-pack value.
	static constexpr formatted_val_t from_underlying_value(underlying_val_t v) noexcept {
		if constexpr (std::is_same_v<formatted_val_t, underlying_val_t>) {
			return v;
		} else {
			return std::bit_cast<formatted_val_t>(v);
		}
	}

	/// @brief Loads the canonical underlying representation from a storage object.
	/// @note The default implementation is direct when `storage_t == underlying_val_t`.
	static constexpr underlying_val_t load_underlying_value(const storage_t& storage) noexcept {
		static_assert(std::same_as<storage_t, underlying_val_t>,
		              "bitfield_word_spec default load_underlying_value requires storage_t == underlying_val_t; provide a custom word spec otherwise");
		return storage;
	}

	/// @brief Stores the canonical underlying representation back into a storage object.
	/// @note The default implementation is direct when `storage_t == underlying_val_t`.
	static constexpr void store_underlying_value(storage_t& storage, underlying_val_t v) noexcept {
		static_assert(std::same_as<storage_t, underlying_val_t>,
		              "bitfield_word_spec default store_underlying_value requires storage_t == underlying_val_t; provide a custom word spec otherwise");
		storage = v;
	}
};

/// @brief Concept describing the minimal "word spec" protocol accepted by `bitfield_pack`.
/// @details A word spec supplies:
/// - `storage_t`: owned backend representation type
/// - `underlying_val_t`: canonical unsigned whole value used for bit math
/// - `formatted_val_t`: public whole-pack API type
/// - `directly_mutable`: whether ordinary mutating member APIs may participate on the live pack
/// - load/store hooks between `storage_t` and `underlying_val_t`
/// - whole-pack conversions between `formatted_val_t` and `underlying_val_t`
template <class Word>
concept bitfield_word_spec_like =
	requires(const typename Word::storage_t& cstorage,
	         typename Word::storage_t& storage,
	         typename Word::formatted_val_t formatted_value,
	         typename Word::underlying_val_t underlying_value) {
		typename Word::storage_t;
		typename Word::underlying_val_t;
		typename Word::formatted_val_t;
		requires std::unsigned_integral<typename Word::underlying_val_t>;
		{ Word::directly_mutable } -> std::convertible_to<bool>;
		{ Word::to_underlying_value(formatted_value) } -> std::same_as<typename Word::underlying_val_t>;
		{ Word::from_underlying_value(underlying_value) } -> std::same_as<typename Word::formatted_val_t>;
		{ Word::load_underlying_value(cstorage) } -> std::same_as<typename Word::underlying_val_t>;
		{ Word::store_underlying_value(storage, underlying_value) } -> std::same_as<void>;
	};

/// @brief Scratch-copy word spec preserving whole-pack conversions while rebinding storage to the
///        canonical underlying value type and re-enabling direct mutability.
template <class WordSpec>
struct scratch_copy_word_spec {
	using underlying_val_t = typename WordSpec::underlying_val_t;
	using formatted_val_t = typename WordSpec::formatted_val_t;
	using storage_t = underlying_val_t;
	static constexpr bool directly_mutable = true;

	static constexpr underlying_val_t to_underlying_value(formatted_val_t v) noexcept {
		return WordSpec::to_underlying_value(v);
	}

	static constexpr formatted_val_t from_underlying_value(underlying_val_t v) noexcept {
		return WordSpec::from_underlying_value(v);
	}

	static constexpr underlying_val_t load_underlying_value(const storage_t& storage) noexcept {
		return storage;
	}

	static constexpr void store_underlying_value(storage_t& storage, underlying_val_t v) noexcept {
		storage = v;
	}
};

/// @brief Normalizes a shorthand word type into a full word-spec type.
template <class Word>
struct normalize_word;

template <class Word>
	requires std::unsigned_integral<Word>
struct normalize_word<Word> {
	using type = bitfield_word_spec<Word, Word>;
};

template <bitfield_word_spec_like Word>
struct normalize_word<Word> {
	using type = Word;
};

template <class Word>
using normalize_word_t = typename normalize_word<Word>::type;

/// @brief Acceptable plain field-key types for the cast-based indexing wrapper.
/// @details This is intentionally narrower than "anything castable to std::size_t":
///          plain indexing keys are either integral positional keys or raw enums.
///          Raw enums are most appropriate when their values already form the desired
///          contiguous zero-based field order. Sparse, reordered, or externally fixed
///          enum values should use a custom indexing descriptor instead of relying on
///          direct casting.
template <class Key>
concept bitfield_cast_index_key =
	(std::integral<Key> && !std::same_as<std::remove_cv_t<Key>, bool>) || std::is_enum_v<Key>;

/// @brief Cast-based indexing descriptor for plain integral or enum field keys.
/// @tparam Key Raw field-key type accepted at the `bitfield_pack` call site.
/// @details This is the normalization wrapper used when callers provide a plain key type rather than
///          a full descriptor. The mapping policy is a direct `static_cast<std::size_t>(key)`,
///          so enum inputs should already encode the intended field slot numbering.
template <class Key>
	requires bitfield_cast_index_key<Key>
struct bitfield_index_by_cast {
	using field_key = Key;

	static consteval std::size_t to_index(field_key key) noexcept {
		return static_cast<std::size_t>(key);
	}
};

/// @brief Concept describing the canonical indexing descriptor protocol used by `bitfield_pack`.
/// @details Indexing descriptors only map a field key to a zero-based field-spec slot index.
///          They do not participate in widths, offsets, masks, or storage policy.
template <class IndexingSpec>
concept bitfield_indexing_descriptor =
	requires(typename IndexingSpec::field_key key) {
		typename IndexingSpec::field_key;
		{ IndexingSpec::to_index(key) } -> std::same_as<std::size_t>;
	};

/// @brief Normalizes an indexing spec into the canonical descriptor protocol.
/// @details Normalization is:
/// - preserve the type as-is when it already satisfies the full descriptor protocol
/// - otherwise wrap the plain integral/enum key type in `bitfield_index_by_cast`
template <class IndexingSpec, bool = bitfield_indexing_descriptor<IndexingSpec>>
struct normalize_indexing {
	using type = bitfield_index_by_cast<IndexingSpec>;
};

template <class IndexingSpec>
struct normalize_indexing<IndexingSpec, true> {
	using type = IndexingSpec;
};

template <class IndexingSpec>
using normalize_indexing_t = typename normalize_indexing<IndexingSpec>::type;

template <class Tuple, class T>
struct tuple_append;

template <class... Ts, class T>
struct tuple_append<std::tuple<Ts...>, T> {
	using type = std::tuple<Ts..., T>;
};

template <class Tuple, class T>
using tuple_append_t = typename tuple_append<Tuple, T>::type;

/// @brief Concept describing one field's width and semantic encoding protocol.
/// @tparam FieldSpec Field descriptor type.
/// @tparam StorageUInt Canonical storage word type for the enclosing pack.
/// @details Field specs describe either:
/// - a fixed-width field with `width`, `encode`, `decode`, and `is_valid`, or
/// - the trailing `bitfield_remainder` marker.
/// Semantic encode/decode is always expressed in terms of the enclosing pack's `storage_t`.
/// `is_valid(decoded)` is interpreted in the encoded-storage sense used by the pack:
/// field-specific semantic checks happen here, while width fit is checked separately by
/// `bitfield_pack::is_valid()` after `encode(decoded)`. Custom codecs should keep that
/// split in mind, especially for signed decoded domains or non-identity encodings.
template <class FieldSpec, class StorageUInt>
concept bitfield_field_spec =
	std::unsigned_integral<StorageUInt> &&
	requires {
		typename FieldSpec::template for_storage_t<StorageUInt>;
	} &&
	requires(StorageUInt u, typename FieldSpec::template for_storage_t<StorageUInt>::decoded_type v) {
		{ FieldSpec::template for_storage_t<StorageUInt>::is_remainder } -> std::convertible_to<bool>;
		{ FieldSpec::template for_storage_t<StorageUInt>::width } -> std::convertible_to<std::size_t>;
		{ FieldSpec::template for_storage_t<StorageUInt>::encode(v) } -> std::same_as<StorageUInt>;
		{ FieldSpec::template for_storage_t<StorageUInt>::decode(u) } -> std::same_as<typename FieldSpec::template for_storage_t<StorageUInt>::decoded_type>;
		{ FieldSpec::template for_storage_t<StorageUInt>::is_valid(v) } -> std::convertible_to<bool>;
	};

/// @brief Fixed-width identity field spec.
/// @tparam Width Number of bits assigned to the field.
/// @tparam DecodedT Decoded semantic value type, or `void` to use the enclosing `StorageT`.
/// @details Encoding and decoding are identity-style casts, and semantic validity defaults to true.
///          Width-fit checking belongs to `bitfield_pack`.
template <std::size_t Width, typename DecodedT = void>
struct bitfield_field_width {
	static_assert(Width > 0, "bitfield_field_width<Width>: Width must be > 0");

	template <class StorageT>
	struct for_storage_t {
		static_assert(std::unsigned_integral<StorageT>, "StorageT must be unsigned integral");

		using decoded_type = std::conditional_t<std::is_void_v<DecodedT>, StorageT, DecodedT>;

		static constexpr bool is_remainder = false;
		static constexpr std::size_t width = Width;

		static constexpr StorageT encode(decoded_type v) noexcept {
			return static_cast<StorageT>(v);
		}

		static constexpr decoded_type decode(StorageT bits) noexcept {
			return static_cast<decoded_type>(bits);
		}

		static constexpr bool is_valid(decoded_type) noexcept {
			return true;
		}
	};
};

/// @brief Trailing field spec that consumes all remaining storage bits.
/// @details The remainder field must be the last field in the pack. Its semantic behavior is
///          identity-style; width is derived from the enclosing layout rather than declared here.
struct bitfield_remainder {
	template <class StorageT>
	struct for_storage_t {
		static_assert(std::unsigned_integral<StorageT>, "StorageT must be unsigned integral");

		using decoded_type = StorageT;

		static constexpr bool is_remainder = true;
		static constexpr std::size_t width = 0;

		static constexpr StorageT encode(decoded_type v) noexcept { return static_cast<StorageT>(v); }
		static constexpr decoded_type decode(StorageT bits) noexcept { return bits; }

		static constexpr bool is_valid(decoded_type) noexcept {
			return true;
		}
	};
};

template <class StorageUInt>
constexpr std::size_t storage_bits_v = std::numeric_limits<StorageUInt>::digits;

/// @brief All-ones value in the canonical storage domain.
template <class StorageUInt>
constexpr StorageUInt all_ones_v = ~StorageUInt(0);

/// @brief Unshifted mask for a `W`-bit field in `StorageUInt`.
template <std::size_t W, class StorageUInt>
constexpr StorageUInt value_mask_unshifted() noexcept {
	static_assert(std::unsigned_integral<StorageUInt>);
	if constexpr (W == 0) {
		return StorageUInt(0);
	} else if constexpr (W >= storage_bits_v<StorageUInt>) {
		return all_ones_v<StorageUInt>;
	} else {
		return (StorageUInt(1) << W) - 1;
	}
}

/// @brief Count how many remainder fields appear.
/// @details This probes `for_storage_t<std::uintmax_t>` only to read the storage-invariant
///          `is_remainder` marker. Field specs are expected to be storage-parametric over
///          unsigned integral types, so this is a protocol assumption rather than a layout
///          dependency on `std::uintmax_t` specifically.
template <class... Specs>
consteval std::size_t remainder_count() {
	return (std::size_t(Specs::template for_storage_t<std::uintmax_t>::is_remainder) + ... + 0u);
}

// ------------------------ layout computation ------------------------

template <std::size_t FieldCount>
struct bitfield_layout_data {
	std::array<std::size_t, FieldCount> widths{};
	std::array<std::size_t, FieldCount> offsets{};
};

/// @brief Compile-time layout traits for one `bitfield_pack` instantiation.
/// @details This centralizes field-count, width resolution, remainder handling, and offset
///          computation so the main class can stay focused on the public API and word access paths.
template <class StorageT, class... FieldSpecs>
struct bitfield_layout_traits {
	static_assert(std::unsigned_integral<StorageT>, "StorageT must be unsigned integral");

	using storage_type = StorageT;
	using field_specs = std::tuple<FieldSpecs...>;

	static constexpr std::size_t storage_bits = storage_bits_v<storage_type>;
	static constexpr std::size_t field_count = sizeof...(FieldSpecs);
	static constexpr std::size_t remainder_fields = remainder_count<FieldSpecs...>();
	static constexpr bool has_remainder = remainder_fields == 1;

	template <std::size_t I>
	using raw_field_spec_t = std::tuple_element_t<I, field_specs>;

	template <std::size_t I>
	using field_spec_t = typename raw_field_spec_t<I>::template for_storage_t<storage_type>;

	template <std::size_t I>
	static constexpr bool is_remainder_v = field_spec_t<I>::is_remainder;

	template <std::size_t I>
	static consteval std::size_t declared_width() {
		if constexpr (is_remainder_v<I>) {
			return 0u;
		} else {
			return std::size_t(field_spec_t<I>::width);
		}
	}

	static consteval auto declared_widths() {
		std::array<std::size_t, field_count> widths{};
		[&]<std::size_t... Is>(std::index_sequence<Is...>) {
			((widths[Is] = declared_width<Is>()), ...);
		}(std::make_index_sequence<field_count>{});
		return widths;
	}

	static consteval std::size_t sum_widths(const std::array<std::size_t, field_count>& widths) {
		std::size_t sum = 0;
		for (std::size_t i = 0; i < field_count; ++i) {
			sum += widths[i];
		}
		return sum;
	}

	static consteval bool remainder_is_final() {
		if constexpr (!has_remainder) {
			return true;
		} else if constexpr (!is_remainder_v<field_count - 1>) {
			return false;
		} else {
			return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
				return ((!is_remainder_v<Is>) && ...);
			}(std::make_index_sequence<field_count - 1>{});
		}
	}

	static consteval bitfield_layout_data<field_count> make_layout() {
		constexpr auto fixed_widths = declared_widths();
		constexpr std::size_t fixed_width_sum = sum_widths(fixed_widths);
		static_assert(remainder_is_final(), "bitfield_pack: remainder field must be the final field");

		auto widths = fixed_widths;
		if constexpr (has_remainder) {
			static_assert(fixed_width_sum < storage_bits, "bitfield_pack: remainder would be zero bits wide; disallowed");
			widths[field_count - 1] = storage_bits - fixed_width_sum;
		} else {
			static_assert(fixed_width_sum <= storage_bits, "bitfield_pack: total fixed widths exceed storage bits");
		}

		std::array<std::size_t, field_count> offsets{};
		std::size_t current_offset = 0;
		for (std::size_t i = 0; i < field_count; ++i) {
			offsets[i] = current_offset;
			current_offset += widths[i];
		}

		return bitfield_layout_data<field_count>{widths, offsets};
	}

	static constexpr auto layout = make_layout();
	static constexpr std::size_t used_bits = sum_widths(layout.widths);
	static constexpr std::size_t extra_bits = storage_bits - used_bits;
	static constexpr bool has_extra_bits = extra_bits != 0;
	static constexpr std::size_t extra_bits_offset = used_bits;
	static constexpr storage_type extra_bits_mask = value_mask_unshifted<extra_bits, storage_type>();
};

} // namespace bitfield_pack_detail

/// @brief Packs multiple logical fields into one canonical storage word.
/// @tparam Word Unsigned integral word shorthand or a full word spec.
/// @tparam IndexingSpec Field-indexing specification. Use `std::size_t` for generic numeric keys,
///         a raw enum for the preferred named-field case, or a custom descriptor for sparse/reordered mapping.
/// @tparam FieldSpecs Field descriptors evaluated in least-significant-bit first order.
/// @details `bitfield_pack` owns a `storage_t`, performs all field math in `underlying_val_t`,
///          and exposes whole-pack conversion through `formatted_val_t`. Ordinary mutation APIs
///          are whole-value load/modify/store operations through the word-spec hooks; retry/CAS
///          policy, if desired, belongs to the caller using explicit backend access.
///          This is a typed field-access discipline and readability utility, not a sequence
///          container: callers define one fixed compile-time layout and then manipulate named
///          slices of that word.
template <class Word, class IndexingSpec, class... FieldSpecs>
class bitfield_pack {
private:
	using word_spec_t = bitfield_pack_detail::normalize_word_t<Word>;
	using indexing_spec_t = bitfield_pack_detail::normalize_indexing_t<IndexingSpec>;
	using storage_t = typename word_spec_t::storage_t;
	using underlying_val_t = typename word_spec_t::underlying_val_t;
	using formatted_val_t = typename word_spec_t::formatted_val_t;
	using field_key_t = typename indexing_spec_t::field_key;
	using layout_traits = bitfield_pack_detail::bitfield_layout_traits<underlying_val_t, FieldSpecs...>;
	static constexpr bool kDirectlyMutable = word_spec_t::directly_mutable;
	static constexpr bool kScratchCopyAvailable = !std::same_as<storage_t, underlying_val_t> || !kDirectlyMutable;
	using scratch_pack_t =
		bitfield_pack<bitfield_pack_detail::scratch_copy_word_spec<word_spec_t>, IndexingSpec, FieldSpecs...>;

	static_assert(std::unsigned_integral<underlying_val_t>, "bitfield_pack: underlying_val_t must be unsigned integral");
	static_assert(sizeof...(FieldSpecs) > 0, "bitfield_pack: must define at least one field");
	static_assert(bitfield_pack_detail::remainder_count<FieldSpecs...>() <= 1, "bitfield_pack: at most one remainder field is allowed");
	static_assert(bitfield_pack_detail::bitfield_indexing_descriptor<indexing_spec_t>,
	              "bitfield_pack: IndexingSpec must normalize to a descriptor with field_key and to_index()");

	// Validate that each FieldSpec conforms.
	static_assert((bitfield_pack_detail::bitfield_field_spec<FieldSpecs, underlying_val_t> && ...),
	              "bitfield_pack: all FieldSpecs must satisfy the bitfield_field_spec concept");

	static constexpr std::size_t kUnderlyingValueBits = layout_traits::storage_bits;
	static constexpr std::size_t kFieldCount = layout_traits::field_count;
	static constexpr std::size_t kExtraBits = layout_traits::extra_bits;
	static constexpr bool kHasExtraBits = layout_traits::has_extra_bits;
	static constexpr std::size_t kExtraBitsOffset = layout_traits::extra_bits_offset;
	static constexpr underlying_val_t kExtraBitsMask = layout_traits::extra_bits_mask;
	static constexpr auto kLayout = layout_traits::layout;

	template <std::size_t I>
	using field_spec_t = typename layout_traits::template field_spec_t<I>;

	template <std::size_t I>
	struct field_slot_traits {
		static constexpr std::size_t index = I;
		static constexpr std::size_t width = kLayout.widths[index];
		static constexpr std::size_t offset = kLayout.offsets[index];
		static constexpr underlying_val_t value_mask = bitfield_pack_detail::value_mask_unshifted<width, underlying_val_t>();
		static constexpr underlying_val_t mask = []() consteval {
			if constexpr (width == 0) {
				return underlying_val_t(0);
			} else if constexpr (width >= kUnderlyingValueBits) {
				return bitfield_pack_detail::all_ones_v<underlying_val_t>;
			} else {
				return underlying_val_t(value_mask << offset);
			}
		}();
		using value_type = typename field_spec_t<index>::decoded_type;
	};

	template <field_key_t Field>
	static consteval std::size_t field_index() noexcept {
		constexpr std::size_t index = indexing_spec_t::to_index(Field);
		static_assert(index < kFieldCount, "bitfield_pack: field key maps outside the defined field-spec range");
		return index;
	}

	template <field_key_t Field>
	struct field_traits {
		static constexpr std::size_t index = field_index<Field>();
		static constexpr std::size_t width = field_slot_traits<index>::width;
		static constexpr std::size_t offset = field_slot_traits<index>::offset;
		static constexpr underlying_val_t value_mask = field_slot_traits<index>::value_mask;
		static constexpr underlying_val_t mask = field_slot_traits<index>::mask;
		using value_type = typename field_slot_traits<index>::value_type;
	};

	template <std::size_t... Is>
	static auto make_field_values_type(std::index_sequence<Is...>) -> std::tuple<typename field_slot_traits<Is>::value_type...>;

	using field_values_type = decltype(make_field_values_type(std::make_index_sequence<kFieldCount>{}));

	template <std::size_t I>
	static constexpr underlying_val_t get_bits_by_index(underlying_val_t underlying_value) noexcept {
		return underlying_val_t((underlying_value >> field_slot_traits<I>::offset) & field_slot_traits<I>::value_mask);
	}

	template <std::size_t I>
	static constexpr typename field_slot_traits<I>::value_type get_value_by_index(underlying_val_t underlying_value) noexcept {
		return field_spec_t<I>::decode(get_bits_by_index<I>(underlying_value));
	}

	template <std::size_t I>
	static constexpr bool is_valid_by_index(typename field_slot_traits<I>::value_type v) noexcept {
		const underlying_val_t encoded = field_spec_t<I>::encode(v);
		bool width_fits = true;
		if constexpr (field_slot_traits<I>::width < kUnderlyingValueBits) {
			width_fits = (encoded & ~field_slot_traits<I>::value_mask) == 0;
		}
		return width_fits && field_spec_t<I>::is_valid(v);
	}

	template <class Tuple, std::size_t... Is>
	static constexpr bool validate_all_fields_impl(const Tuple& values, std::index_sequence<Is...>) noexcept {
		return (is_valid_by_index<Is>(static_cast<typename field_slot_traits<Is>::value_type>(std::get<Is>(values))) && ...);
	}

	template <class Tuple, std::size_t... Is>
	static constexpr underlying_val_t compose_underlying_value_impl(
		const Tuple& values,
		underlying_val_t extra_bits,
		std::index_sequence<Is...>) noexcept {
		underlying_val_t underlying_value = 0;
		((underlying_value |= underlying_val_t(
			(field_spec_t<Is>::encode(static_cast<typename field_slot_traits<Is>::value_type>(std::get<Is>(values))) &
			 field_slot_traits<Is>::value_mask) << field_slot_traits<Is>::offset)), ...);
		if constexpr (kHasExtraBits) {
			underlying_value |= underlying_val_t((extra_bits & kExtraBitsMask) << kExtraBitsOffset);
		}
		return underlying_value;
	}

	template <class Tuple, std::size_t... Is>
	static consteval bool values_match_fields_impl(std::index_sequence<Is...>) {
		return (std::constructible_from<typename field_slot_traits<Is>::value_type, std::tuple_element_t<Is, Tuple>> && ...);
	}

	template <class... Values>
	static consteval bool values_match_fields() {
		if constexpr (sizeof...(Values) != kFieldCount) {
			return false;
		} else {
			return values_match_fields_impl<std::tuple<Values...>>(std::make_index_sequence<kFieldCount>{});
		}
	}

	template <class... Values>
	static consteval bool values_match_fields_plus_extra_bits() {
		if constexpr (!kHasExtraBits || sizeof...(Values) != kFieldCount + 1) {
			return false;
		} else {
			using values_tuple = std::tuple<Values...>;
			return values_match_fields_impl<values_tuple>(std::make_index_sequence<kFieldCount>{}) &&
			       std::constructible_from<underlying_val_t, std::tuple_element_t<kFieldCount, values_tuple>>;
		}
	}

	constexpr void initialize_underlying_value(underlying_val_t v) BITFIELD_PACK_NOEXCEPT {
		word_spec_t::store_underlying_value(storage_, v);
	}

public:
	/// @brief Canonical normalized word spec used by this pack.
	using word_spec = word_spec_t;
	/// @brief Canonical normalized indexing descriptor used by this pack.
	using indexing_spec = indexing_spec_t;
	/// @brief Backend/storage object type owned by this pack instance.
	using storage_type = storage_t;
	/// @brief Canonical unsigned whole-value type used for masks, shifts, and layout.
	using underlying_val_type = underlying_val_t;
	/// @brief Whole-pack API type returned by `formatted_value()` and accepted by `set_formatted_value()`.
	using formatted_val_type = formatted_val_t;
	/// @brief Field-key type accepted by field-oriented APIs.
	using field_key_type = field_key_t;
	/// @brief Tuple returned by `get_all()`, in declared field order plus trailing extra bits when present.
	using all_values_type = std::conditional_t<kHasExtraBits,
	                                          bitfield_pack_detail::tuple_append_t<field_values_type, underlying_val_t>,
	                                          field_values_type>;
	/// @brief Whether the live pack supports ordinary mutating member APIs.
	static constexpr bool directly_mutable = kDirectlyMutable;
	/// @brief Mutable scratch pack rebound to plain underlying-value storage.
	/// @details This alias participates only when scratch copies are actually useful for the live pack.
	template <bool Enabled = kScratchCopyAvailable>
		requires(Enabled)
	using scratch_t = scratch_pack_t;

	/// @brief Tag type selecting direct storage construction.
	/// @details This allows callers with a pre-built `storage_t` to bypass the
	///          default-construct-then-store path used by the underlying-value constructor.
	struct from_backend_t {
		explicit constexpr from_backend_t() = default;
	};

	/// @brief Tag constant used with the direct-storage constructor.
	static constexpr from_backend_t from_backend{};

	/// @brief Number of fields.
	static consteval std::size_t size() noexcept { return kFieldCount; }

	/// @brief Number of residual high bits not claimed by declared fields.
	static consteval std::size_t extra_bits_width() noexcept { return kExtraBits; }

	/// @brief Construct with all bits zero.
	constexpr bitfield_pack() BITFIELD_PACK_NOEXCEPT : storage_{} {
		initialize_underlying_value(0);
	}

	/// @brief Construct from the canonical underlying whole value.
	explicit constexpr bitfield_pack(underlying_val_t underlying_value) BITFIELD_PACK_NOEXCEPT : storage_{} {
		initialize_underlying_value(underlying_value);
	}

	/// @brief Construct directly from a backend object.
	/// @details The backend is taken by value and then moved into place. This keeps the
	///          constructor simple for both movable backends and prvalue call sites while
	///          still bypassing the default-construct-then-store path.
	explicit constexpr bitfield_pack(from_backend_t, storage_t storage) BITFIELD_PACK_NOEXCEPT
		: storage_(std::move(storage)) {}

	/// @brief Get the canonical underlying whole value (always available).
	constexpr underlying_val_t underlying_value() const BITFIELD_PACK_NOEXCEPT { return load_underlying_value(); }

	/// @brief Set the canonical underlying whole value.
	/// @details This only participates for directly mutable packs. Atomic-backed or otherwise
	///          read/snapshot-oriented packs are expected to mutate via explicit storage/CAS workflows.
	constexpr void set_underlying_value(underlying_val_t v) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		store_underlying_value(v);
	}

	/// @brief Get the public whole-pack formatted value.
	/// @note For integral shorthand words this is the same as `underlying_value()`.
	constexpr formatted_val_t formatted_value() const BITFIELD_PACK_NOEXCEPT {
		return word_spec_t::from_underlying_value(load_underlying_value());
	}

	/// @brief Store the public whole-pack formatted value.
	/// @note For integral shorthand words this is the same as `set_underlying_value()`.
	constexpr void set_formatted_value(formatted_val_t v) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		store_underlying_value(word_spec_t::to_underlying_value(v));
	}

	/// @brief Returns the owned storage object by reference.
	constexpr storage_type& storage() BITFIELD_PACK_NOEXCEPT { return storage_; }

	/// @brief Returns the owned storage object by const reference.
	constexpr storage_type const& storage() const BITFIELD_PACK_NOEXCEPT { return storage_; }

	/// @brief Loads the canonical underlying value through the word-spec hook.
	constexpr underlying_val_t load_underlying_value() const BITFIELD_PACK_NOEXCEPT {
		return word_spec_t::load_underlying_value(storage_);
	}

	/// @brief Stores the canonical underlying value through the word-spec hook.
	constexpr void store_underlying_value(underlying_val_t v) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		word_spec_t::store_underlying_value(storage_, v);
	}

	/// @brief Returns an equivalent pack rebound to plain underlying-value storage.
	/// @details Live non-directly-mutable packs are for read/snapshot/storage/CAS workflows. Scratch
	///          copies are the directly mutable staging packs used to build before/after states.
	constexpr auto scratch_copy() const BITFIELD_PACK_NOEXCEPT
		requires(kScratchCopyAvailable) {
		return scratch_pack_t(load_underlying_value());
	}

	/// @brief Returns the compile-time bit width of a field key.
	template <field_key_t Field>
	static consteval std::size_t field_width() noexcept {
		return field_traits<Field>::width;
	}

	/// @brief Returns the compile-time starting bit offset of a field key.
	/// @details Offsets are computed in least-significant-bit first packing order.
	template <field_key_t Field>
	static consteval std::size_t field_offset() noexcept {
		return field_traits<Field>::offset;
	}

	/// @brief Returns the unshifted mask covering a field key's value bits.
	template <field_key_t Field>
	static consteval underlying_val_t field_value_mask() noexcept {
		return field_traits<Field>::value_mask;
	}

	/// @brief Returns the shifted mask for a field key within the full underlying value.
	template <field_key_t Field>
	static consteval underlying_val_t field_mask() noexcept {
		return field_traits<Field>::mask;
	}

	/// @brief Returns the maximum raw bit-pattern representable by a field key.
	template <field_key_t Field>
	static consteval underlying_val_t field_max_bits() noexcept {
		return field_traits<Field>::value_mask;
	}

	/// @brief Semantic value type decoded by a field key.
	template <field_key_t Field>
	using value_type = typename field_traits<Field>::value_type;

	/// @brief Extracts the raw bit-pattern stored in a field key.
	template <field_key_t Field>
	constexpr underlying_val_t get_bits() const BITFIELD_PACK_NOEXCEPT {
		return get_bits_by_index<field_traits<Field>::index>(load_underlying_value());
	}

	/// @brief Stores a raw bit-pattern into a field key.
	/// @note Oversized values are masked to the field width.
	/// @details This is the bit-pattern-oriented store path. It does not consult the
	///          field spec's semantic `is_valid()` predicate.
	/// @warning This is a whole-value load/modify/store through the backend hooks, not a CAS operation.
	template <field_key_t Field>
	constexpr void set_bits(underlying_val_t bits) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		constexpr underlying_val_t m = field_traits<Field>::mask;
		constexpr std::size_t off = field_traits<Field>::offset;
		const underlying_val_t current_underlying_value = load_underlying_value();
		store_underlying_value(underlying_val_t((current_underlying_value & ~m) | ((underlying_val_t(bits) << off) & m)));
	}

	/// @brief Decodes and returns the semantic value of a field key.
	template <field_key_t Field>
	constexpr value_type<Field> get() const BITFIELD_PACK_NOEXCEPT {
		return get_value_by_index<field_traits<Field>::index>(load_underlying_value());
	}

	/// @brief Encodes and stores the semantic value of a field key.
	/// @note The encoded bits are still masked to the field width after encoding.
	/// @details `set_masked()` intentionally performs a masked store, not a checked store. If the
	///          encoded representation does not fit, high bits are truncated just as they are
	///          for `set_bits()`. Call `is_valid()` or `validate()` first when silent
	///          truncation would be surprising or unacceptable, or use `set_if_valid()`
	///          for an in-place checked store.
	/// @warning This is a whole-value load/modify/store through the backend hooks, not a CAS operation.
	template <field_key_t Field>
	constexpr void set_masked(value_type<Field> v) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		const underlying_val_t enc = field_spec_t<field_traits<Field>::index>::encode(v);
		set_bits<Field>(enc);
	}

	/// @brief Validates and stores the semantic value of a field key.
	/// @details Invalid values do not modify the pack and return `false`. Valid values are
	///          stored through the same masked whole-value update path used by `set_masked()`.
	template <field_key_t Field>
	constexpr bool set_if_valid(value_type<Field> v) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		if (!is_valid<Field>(v)) {
			return false;
		}
		set_masked<Field>(v);
		return true;
	}

	/// @brief Checks whether a semantic value is valid for a field key before masking.
	/// @details Validity is judged in the encoded storage domain used by the pack:
	///          `encode(v)` must fit within the field width, and the field spec's own
	///          `is_valid(v)` predicate must also accept the decoded value. This matters for
	///          signed decoded types and custom codecs, where "domain-valid" and
	///          "encoded-width-valid" are related but distinct questions.
	template <field_key_t Field>
	static constexpr bool is_valid(value_type<Field> v) noexcept {
		return is_valid_by_index<field_traits<Field>::index>(v);
	}

	/// @brief Validates a semantic value for a field key using `BITFIELD_PACK_ASSERT`.
	/// @details This is an assertion hook only. It does not store the value, and `set_masked()`
	///          does not call it implicitly.
	template <field_key_t Field>
	static constexpr void validate(value_type<Field> v) BITFIELD_PACK_NOEXCEPT {
		BITFIELD_PACK_ASSERT(is_valid<Field>(v));
	}

	template <std::size_t... Is>
	constexpr all_values_type get_all_impl(std::index_sequence<Is...>) const BITFIELD_PACK_NOEXCEPT {
		const underlying_val_t current_underlying_value = load_underlying_value();
		if constexpr (kHasExtraBits) {
			return all_values_type{get_value_by_index<Is>(current_underlying_value)..., get_extra_bits()};
		} else {
			return all_values_type{get_value_by_index<Is>(current_underlying_value)...};
		}
	}

	/// @brief Returns all decoded field values in declared order.
	/// @details When declared fields do not consume the full underlying-value width,
	///          `get_all()` appends one final trailing element containing the residual
	///          unclaimed bits packed down into the low bits of `underlying_val_t`.
	constexpr all_values_type get_all() const BITFIELD_PACK_NOEXCEPT {
		return get_all_impl(std::make_index_sequence<kFieldCount>{});
	}

	/// @brief Returns residual unclaimed bits packed into the low bits.
	template <bool Enabled = kHasExtraBits>
		requires(Enabled)
	constexpr underlying_val_t get_extra_bits() const BITFIELD_PACK_NOEXCEPT {
		return underlying_val_t((load_underlying_value() >> kExtraBitsOffset) & kExtraBitsMask);
	}

	/// @brief Stores all decoded field values in declared order with silent masking.
	/// @details This computes a fresh underlying value from scratch, zero-filling any
	///          residual bits when they are not supplied explicitly.
	template <class... Values>
		requires(values_match_fields<Values...>())
	constexpr void set_all_masked(Values... values) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		const auto field_values = std::tuple<Values...>(std::move(values)...);
		store_underlying_value(compose_underlying_value_impl(field_values, underlying_val_t{0}, std::make_index_sequence<kFieldCount>{}));
	}

	template <class... Values>
		requires(values_match_fields_plus_extra_bits<Values...>())
	constexpr void set_all_masked(Values... values) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		auto values_tuple = std::tuple<Values...>(std::move(values)...);
		const underlying_val_t extra_bits = static_cast<underlying_val_t>(std::get<kFieldCount>(values_tuple));
		store_underlying_value(compose_underlying_value_impl(values_tuple, extra_bits, std::make_index_sequence<kFieldCount>{}));
	}

	/// @brief Validates and stores all decoded field values in declared order.
	/// @details Invalid field values, or out-of-range extra bits when present explicitly,
	///          leave the pack unchanged and return `false`.
	template <class... Values>
		requires(values_match_fields<Values...>())
	constexpr bool set_all_if_valid(Values... values) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		const auto field_values = std::tuple<Values...>(std::move(values)...);
		if (!validate_all_fields_impl(field_values, std::make_index_sequence<kFieldCount>{})) {
			return false;
		}
		store_underlying_value(compose_underlying_value_impl(field_values, underlying_val_t{0}, std::make_index_sequence<kFieldCount>{}));
		return true;
	}

	template <class... Values>
		requires(values_match_fields_plus_extra_bits<Values...>())
	constexpr bool set_all_if_valid(Values... values) BITFIELD_PACK_NOEXCEPT
		requires(kDirectlyMutable) {
		auto values_tuple = std::tuple<Values...>(std::move(values)...);
		const underlying_val_t extra_bits = static_cast<underlying_val_t>(std::get<kFieldCount>(values_tuple));
		if (!validate_all_fields_impl(values_tuple, std::make_index_sequence<kFieldCount>{})) {
			return false;
		}
		if ((extra_bits & ~kExtraBitsMask) != 0) {
			return false;
		}
		store_underlying_value(compose_underlying_value_impl(values_tuple, extra_bits, std::make_index_sequence<kFieldCount>{}));
		return true;
	}

private:
	// Owned storage object that ultimately stores the packed word. All field-oriented operations go
	// through the canonical `underlying_val_t` representation regardless of the formatted value type.
	storage_t storage_{};
};

/// @brief Widths-only convenience alias for identity field specs.
template <class Word, std::size_t... Widths>
using bitfield_pack_bits = bitfield_pack<Word, std::size_t, bitfield_pack_detail::bitfield_field_width<Widths>...>;

/// @brief Convenience alias for the explicit word-spec form.
template <class UnderlyingValueT, class FormattedValueT = UnderlyingValueT>
using bitfield_word_spec = bitfield_pack_detail::bitfield_word_spec<UnderlyingValueT, FormattedValueT>;

/// @brief Convenience alias for the fixed-width field spec with optional decoded type override.
template <std::size_t Width, typename DecodedT = void>
using bitfield_field_width = bitfield_pack_detail::bitfield_field_width<Width, DecodedT>;

/// @brief Convenience alias for the widths-only fixed-width identity field spec.
template <std::size_t Width>
using bitfield_field_spec = bitfield_field_width<Width>;

/// @brief Convenience alias for the trailing remainder field marker.
using bitfield_remainder = bitfield_pack_detail::bitfield_remainder;

} // namespace sw::universal
