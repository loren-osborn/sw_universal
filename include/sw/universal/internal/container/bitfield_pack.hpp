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
///   * an unsigned integral type (shorthand; raw() returns the same type)
///   * bitfield_word_spec<StorageUInt, RawIface> (raw() returns RawIface, storage is StorageUInt)
/// - "IndexingSpec" can be:
///   * `std::size_t` for plain numeric field keys
///   * a raw enum type for named contiguous fields
///   * a custom descriptor with `field_key` and `to_index()` for sparse/reordered field naming
/// - Setters mask and store (silent truncation). Validity is separate:
///   * is_valid<I>(value)
///   * validate<I>(value) -> assertion hook
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
/// Override in unit tests to throw, ignore, etc.
#define BITFIELD_PACK_ASSERT(expr) assert(expr)
#endif

#ifndef BITFIELD_PACK_NOEXCEPT
/// @brief noexcept hook.
/// In unit tests, you can define this to empty to allow thrown-assert paths.
#define BITFIELD_PACK_NOEXCEPT noexcept
#ifdef NDEBUG
#define BITFIELD_PACK_NDEBUG
#endif // def NDEBUG
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
/// bits.set_raw(1.0f);
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
/// bits.template set<ieee754_f32_field::mantissa>(4788187);
/// bits.template set<ieee754_f32_field::exponent>(1);
/// bits.template set<ieee754_f32_field::sign>(0);
///
/// float round_trip = bits.raw();
/// assert(std::abs(round_trip - 3.1415927f) < 0.000001);
/// @endcode

namespace bitfield_pack_detail {

/// @brief Convenience false value for dependent static_assert branches.
template <class T>
inline constexpr bool always_false_v = false;

/// @brief Canonical description of how a bitfield word is stored, exposed, and loaded.
/// @tparam StorageUInt Unsigned integral word used for all masking, shifting, and layout math.
/// @tparam RawIface Whole-word API type exposed by `raw()` / `set_raw()`.
/// @tparam Backend Actual contained backend object type stored by `bitfield_pack`.
/// @details `storage_t` is always the canonical unsigned representation used for bitfield math.
///          `raw_iface_t` is the public whole-word type. `backend_t` is the owned representation
///          inside the pack and may differ from both when callers want custom load/store hooks.
template <class StorageUInt, class RawIface = StorageUInt, class Backend = StorageUInt>
struct bitfield_word_spec {
	static_assert(std::unsigned_integral<StorageUInt>, "StorageUInt must be unsigned integral");
	static_assert(std::is_trivially_copyable_v<RawIface>, "RawIface must be trivially copyable");
	static_assert(sizeof(RawIface) == sizeof(StorageUInt), "RawIface must have same size as StorageUInt");

	using backend_t = Backend;
	using storage_t = StorageUInt;
	using raw_iface_t = RawIface;

	/// @brief Converts a public whole-word value into canonical storage bits.
	static constexpr storage_t to_storage(raw_iface_t v) noexcept {
		if constexpr (std::is_same_v<raw_iface_t, storage_t>) {
			return v;
		} else {
			return std::bit_cast<storage_t>(v);
		}
	}

	/// @brief Converts canonical storage bits into the public whole-word type.
	static constexpr raw_iface_t from_storage(storage_t v) noexcept {
		if constexpr (std::is_same_v<raw_iface_t, storage_t>) {
			return v;
		} else {
			return std::bit_cast<raw_iface_t>(v);
		}
	}

	/// @brief Loads canonical storage bits from a backend object.
	/// @note The default implementation is direct when `backend_t == storage_t`.
	static constexpr storage_t load_storage(const backend_t& backend) noexcept {
		static_assert(std::same_as<backend_t, storage_t>,
		              "bitfield_word_spec default load_storage requires backend_t == storage_t; provide a custom word spec otherwise");
		return backend;
	}

	/// @brief Stores canonical storage bits back into a backend object.
	/// @note The default implementation is direct when `backend_t == storage_t`.
	static constexpr void store_storage(backend_t& backend, storage_t v) noexcept {
		static_assert(std::same_as<backend_t, storage_t>,
		              "bitfield_word_spec default store_storage requires backend_t == storage_t; provide a custom word spec otherwise");
		backend = v;
	}
};

/// @brief Concept describing the minimal "word spec" protocol accepted by `bitfield_pack`.
/// @details A word spec supplies:
/// - `backend_t`: owned representation type
/// - `storage_t`: canonical unsigned word used for bit math
/// - `raw_iface_t`: public whole-word API type
/// - load/store hooks between `backend_t` and `storage_t`
/// - raw whole-word conversions between `raw_iface_t` and `storage_t`
template <class Word>
concept bitfield_word_spec_like =
	requires(const typename Word::backend_t& cbackend,
	         typename Word::backend_t& backend,
	         typename Word::raw_iface_t raw,
	         typename Word::storage_t storage) {
		typename Word::backend_t;
		typename Word::storage_t;
		typename Word::raw_iface_t;
		requires std::unsigned_integral<typename Word::storage_t>;
		{ Word::to_storage(raw) } -> std::same_as<typename Word::storage_t>;
		{ Word::from_storage(storage) } -> std::same_as<typename Word::raw_iface_t>;
		{ Word::load_storage(cbackend) } -> std::same_as<typename Word::storage_t>;
		{ Word::store_storage(backend, storage) } -> std::same_as<void>;
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

/// @brief Identity indexing descriptor for the generic numeric-keyed case.
struct bitfield_index_by_position {
	using field_key = std::size_t;

	static consteval std::size_t to_index(field_key key) noexcept { return key; }
};

/// @brief Identity indexing descriptor for the common raw-enum case.
/// @tparam Enum Field enum whose values already match the field-spec slot order.
template <class Enum>
	requires std::is_enum_v<Enum>
struct bitfield_index_by_enum {
	using field_key = Enum;

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
/// @details Supported inputs are:
/// - `std::size_t` for generic numeric indexing
/// - a raw enum type, which maps by `static_cast<std::size_t>(key)`
/// - a descriptor that already provides `field_key` and `to_index()`
template <class IndexingSpec>
struct normalize_indexing;

template <>
struct normalize_indexing<std::size_t> {
	using type = bitfield_index_by_position;
};

template <class IndexingSpec>
	requires std::is_enum_v<IndexingSpec>
struct normalize_indexing<IndexingSpec> {
	using type = bitfield_index_by_enum<IndexingSpec>;
};

template <bitfield_indexing_descriptor IndexingSpec>
struct normalize_indexing<IndexingSpec> {
	using type = IndexingSpec;
};

template <class IndexingSpec>
using normalize_indexing_t = typename normalize_indexing<IndexingSpec>::type;

/// @brief Concept describing one field's width and semantic encoding protocol.
/// @tparam FieldSpec Field descriptor type.
/// @tparam StorageUInt Canonical storage word type for the enclosing pack.
/// @details Field specs describe either:
/// - a fixed-width field with `width`, `encode`, `decode`, and `is_valid`, or
/// - the trailing `bitfield_remainder` marker.
/// Semantic encode/decode is always expressed in terms of the enclosing pack's `storage_t`.
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
template <class... Specs>
consteval std::size_t remainder_count() {
	return (std::size_t(Specs::template for_storage_t<std::uintmax_t>::is_remainder) + ... + 0u);
}

} // namespace bitfield_pack_detail

/// @brief Packs multiple logical fields into one canonical storage word.
/// @tparam Word Unsigned integral word shorthand or a full word spec.
/// @tparam IndexingSpec Field-indexing specification. Use `std::size_t` for generic numeric keys,
///         a raw enum for the preferred named-field case, or a custom descriptor for sparse/reordered mapping.
/// @tparam FieldSpecs Field descriptors evaluated in least-significant-bit first order.
/// @details `bitfield_pack` owns a `backend_t`, performs all field math in `storage_t`,
///          and exposes whole-word conversion through `raw_iface_t`. Ordinary mutation APIs
///          are whole-word load/modify/store operations through the word-spec hooks; retry/CAS
///          policy, if desired, belongs to the caller using sideband access.
///          This is a typed field-access discipline and readability utility, not a sequence
///          container: callers define one fixed compile-time layout and then manipulate named
///          slices of that word.
template <class Word, class IndexingSpec, class... FieldSpecs>
class bitfield_pack {
private:
	using word_spec_t = bitfield_pack_detail::normalize_word_t<Word>;
	using indexing_spec_t = bitfield_pack_detail::normalize_indexing_t<IndexingSpec>;
	using backend_t = typename word_spec_t::backend_t;
	using storage_t = typename word_spec_t::storage_t;
	using field_key_t = typename indexing_spec_t::field_key;

	static_assert(std::unsigned_integral<storage_t>, "bitfield_pack: storage_t must be unsigned integral");
	static_assert(sizeof...(FieldSpecs) > 0, "bitfield_pack: must define at least one field");
	static_assert(bitfield_pack_detail::remainder_count<FieldSpecs...>() <= 1, "bitfield_pack: at most one remainder field is allowed");
	static_assert(bitfield_pack_detail::bitfield_indexing_descriptor<indexing_spec_t>,
	              "bitfield_pack: IndexingSpec must normalize to a descriptor with field_key and to_index()");

	// Validate that each FieldSpec conforms.
	static_assert((bitfield_pack_detail::bitfield_field_spec<FieldSpecs, storage_t> && ...),
	              "bitfield_pack: all FieldSpecs must satisfy the bitfield_field_spec concept");

	static constexpr std::size_t kStorageBits = bitfield_pack_detail::storage_bits_v<storage_t>;
	static constexpr std::size_t kFieldCount = sizeof...(FieldSpecs);

	template <std::size_t I>
	using spec_t = std::tuple_element_t<I, std::tuple<FieldSpecs...>>;

	template <std::size_t I>
	using instantiated_field_spec_t = typename spec_t<I>::template for_storage_t<storage_t>;

	template <std::size_t I>
	static constexpr bool is_remainder_v = instantiated_field_spec_t<I>::is_remainder;

	template <std::size_t I>
	static consteval std::size_t declared_field_width() {
		if constexpr (instantiated_field_spec_t<I>::is_remainder) {
			return 0u;
		} else {
			return std::size_t(instantiated_field_spec_t<I>::width);
		}
	}

	// Compile-time layout computation. Field specs are interpreted LSB-first; offsets are prefix sums
	// over the resolved widths, with an optional trailing remainder field consuming all spare bits.
	static consteval auto make_widths() {
		std::array<std::size_t, kFieldCount> w{};
		[&]<std::size_t... Is>(std::index_sequence<Is...>) {
			((w[Is] = declared_field_width<Is>()), ...);
		}(std::make_index_sequence<kFieldCount>{});
		return w;
	}

	static consteval bool remainder_is_final() {
		if constexpr (bitfield_pack_detail::remainder_count<FieldSpecs...>() == 0) {
			return true;
		} else {
			if constexpr (!is_remainder_v<kFieldCount - 1>) {
				return false;
			} else {
				return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
					return ((!is_remainder_v<Is>) && ...);
				}(std::make_index_sequence<kFieldCount - 1>{});
			}
		}
	}

	static consteval std::size_t fixed_sum(const std::array<std::size_t, kFieldCount>& w) {
		std::size_t s = 0;
		for (std::size_t i = 0; i < kFieldCount; ++i) s += w[i];
		return s;
	}

	static consteval auto resolve_widths_and_offsets() {
		constexpr auto fixed_widths = make_widths();
		constexpr auto sum_fixed = fixed_sum(fixed_widths);
		auto w = fixed_widths;

		constexpr bool has_remainder = bitfield_pack_detail::remainder_count<FieldSpecs...>() == 1;
		static_assert(remainder_is_final(), "bitfield_pack: remainder field must be the final field");

		if constexpr (has_remainder) {
			static_assert(sum_fixed < kStorageBits, "bitfield_pack: remainder would be zero bits wide; disallowed");
			w[kFieldCount - 1] = kStorageBits - sum_fixed;
		} else {
			static_assert(sum_fixed <= kStorageBits, "bitfield_pack: total fixed widths exceed storage bits");
		}

		// offsets as prefix sums
		std::array<std::size_t, kFieldCount> off{};
		std::size_t cur = 0;
		for (std::size_t i = 0; i < kFieldCount; ++i) {
			off[i] = cur;
			cur += w[i];
		}

		struct result {
			std::array<std::size_t, kFieldCount> widths;
			std::array<std::size_t, kFieldCount> offsets;
		};
		return result{w, off};
	}

	static constexpr auto kLayout = resolve_widths_and_offsets();

	template <field_key_t Field>
	static consteval std::size_t field_index() noexcept {
		constexpr std::size_t index = indexing_spec_t::to_index(Field);
		static_assert(index < kFieldCount, "bitfield_pack: field key maps outside the defined field-spec range");
		return index;
	}

public:
	/// @brief Canonical normalized word spec used by this pack.
	using word_spec = word_spec_t;
	/// @brief Canonical normalized indexing descriptor used by this pack.
	using indexing_spec = indexing_spec_t;
	/// @brief Backend object type owned by this pack instance.
	using backend_type = backend_t;
	/// @brief Canonical unsigned storage word used for masks, shifts, and layout.
	using storage_type = storage_t;
	/// @brief Whole-word API type returned by `raw()` and accepted by `set_raw()`.
	using raw_iface_type = typename word_spec_t::raw_iface_t;
	/// @brief Field-key type accepted by field-oriented APIs.
	using field_key_type = field_key_t;

	/// @brief Tag type selecting direct backend construction.
	/// @details This allows callers with a pre-built `backend_t` to bypass the
	///          default-construct-then-store path used by the storage constructor.
	struct from_backend_t {
		explicit constexpr from_backend_t() = default;
	};

	/// @brief Tag constant used with the backend constructor.
	static constexpr from_backend_t from_backend{};

	/// @brief Mutable advanced-access proxy for backend-aware whole-word operations.
	struct sideband_proxy;
	/// @brief Const advanced-access proxy for backend-aware whole-word observation.
	struct const_sideband_proxy;

	/// @brief Number of fields.
	static consteval std::size_t size() noexcept { return kFieldCount; }

	/// @brief Construct with all bits zero.
	constexpr bitfield_pack() BITFIELD_PACK_NOEXCEPT : backend_{} {
		store_storage_word(0);
	}

	/// @brief Construct from raw storage bits.
	explicit constexpr bitfield_pack(storage_t raw_storage) BITFIELD_PACK_NOEXCEPT : backend_{} {
		store_storage_word(raw_storage);
	}

	/// @brief Construct directly from a backend object.
	/// This bypasses the default-construct-then-store path for backend-backed usage.
	explicit constexpr bitfield_pack(from_backend_t, backend_t backend) BITFIELD_PACK_NOEXCEPT
		: backend_(std::move(backend)) {}

	/// @brief Get raw storage bits (always available).
	constexpr storage_t raw_storage() const BITFIELD_PACK_NOEXCEPT { return load_storage_word(); }

	/// @brief Set raw storage bits (always available).
	constexpr void set_raw_storage(storage_t v) BITFIELD_PACK_NOEXCEPT { store_storage_word(v); }

	/// @brief Get the whole-word public interface value.
	/// @note For integral shorthand words this is the same as `raw_storage()`.
	constexpr raw_iface_type raw() const BITFIELD_PACK_NOEXCEPT { return word_spec_t::from_storage(load_storage_word()); }

	/// @brief Store the whole-word public interface value.
	/// @note For integral shorthand words this is the same as `set_raw_storage()`.
	constexpr void set_raw(raw_iface_type v) BITFIELD_PACK_NOEXCEPT { store_storage_word(word_spec_t::to_storage(v)); }

	/// @brief Returns the compile-time bit width of a field key.
	template <field_key_t Field>
	static consteval std::size_t field_width() noexcept {
		return kLayout.widths[field_index<Field>()];
	}

	/// @brief Returns the compile-time starting bit offset of a field key.
	/// @details Offsets are computed in least-significant-bit first packing order.
	template <field_key_t Field>
	static consteval std::size_t field_offset() noexcept {
		return kLayout.offsets[field_index<Field>()];
	}

	/// @brief Returns the unshifted mask covering a field key's value bits.
	template <field_key_t Field>
	static consteval storage_t field_value_mask() noexcept {
		return bitfield_pack_detail::value_mask_unshifted<field_width<Field>(), storage_t>();
	}

	/// @brief Returns the shifted mask for a field key within the full storage word.
	template <field_key_t Field>
	static consteval storage_t field_mask() noexcept {
		constexpr std::size_t off = field_offset<Field>();
		constexpr std::size_t w = field_width<Field>();
		if constexpr (w == 0) return storage_t(0);
		if constexpr (w >= kStorageBits) return bitfield_pack_detail::all_ones_v<storage_t>;
		return storage_t(field_value_mask<Field>() << off);
	}

	/// @brief Returns the maximum raw bit-pattern representable by a field key.
	template <field_key_t Field>
	static consteval storage_t field_max_bits() noexcept {
		return field_value_mask<Field>();
	}

	/// @brief Semantic value type decoded by a field key.
	template <field_key_t Field>
	using value_type = typename instantiated_field_spec_t<field_index<Field>()>::decoded_type;

	/// @brief Extracts the raw bit-pattern stored in a field key.
	template <field_key_t Field>
	constexpr storage_t get_bits() const BITFIELD_PACK_NOEXCEPT {
		constexpr std::size_t off = field_offset<Field>();
		return storage_t((load_storage_word() >> off) & field_value_mask<Field>());
	}

	/// @brief Stores a raw bit-pattern into a field key.
	/// @note Oversized values are masked to the field width.
	/// @warning This is a whole-word load/modify/store through the backend hooks, not a CAS operation.
	template <field_key_t Field>
	constexpr void set_bits(storage_t bits) BITFIELD_PACK_NOEXCEPT {
		constexpr storage_t m = field_mask<Field>();
		constexpr std::size_t off = field_offset<Field>();
		const storage_t raw = load_storage_word();
		store_storage_word(storage_t((raw & ~m) | ((storage_t(bits) << off) & m)));
	}

	/// @brief Decodes and returns the semantic value of a field key.
	template <field_key_t Field>
	constexpr value_type<Field> get() const BITFIELD_PACK_NOEXCEPT {
		return instantiated_field_spec_t<field_index<Field>()>::decode(get_bits<Field>());
	}

	/// @brief Encodes and stores the semantic value of a field key.
	/// @note The encoded bits are still masked to the field width after encoding.
	/// @warning This is a whole-word load/modify/store through the backend hooks, not a CAS operation.
	template <field_key_t Field>
	constexpr void set(value_type<Field> v) BITFIELD_PACK_NOEXCEPT {
		const storage_t enc = instantiated_field_spec_t<field_index<Field>()>::encode(v);
		set_bits<Field>(enc);
	}

	/// @brief Checks whether a semantic value is valid for a field key before masking.
	template <field_key_t Field>
	static constexpr bool is_valid(value_type<Field> v) noexcept {
		const storage_t enc = instantiated_field_spec_t<field_index<Field>()>::encode(v);
		bool width_fits = true;
		if constexpr (field_width<Field>() < kStorageBits) {
			width_fits = (enc & ~field_value_mask<Field>()) == 0;
		}
		return width_fits && instantiated_field_spec_t<field_index<Field>()>::is_valid(v);
	}

	/// @brief Validates a semantic value for a field key using `BITFIELD_PACK_ASSERT`.
	template <field_key_t Field>
	static constexpr void validate(value_type<Field> v) BITFIELD_PACK_NOEXCEPT {
		BITFIELD_PACK_ASSERT(is_valid<Field>(v));
	}

	/// @brief Returns an explicit advanced-access proxy for backend-aware operations.
	/// @details The main API stays backend-agnostic; callers that need backend access,
	///          manual synchronization, or explicit whole-word load/store can opt in here.
	constexpr sideband_proxy sideband() BITFIELD_PACK_NOEXCEPT { return sideband_proxy(this); }
	constexpr const_sideband_proxy sideband() const BITFIELD_PACK_NOEXCEPT { return const_sideband_proxy(this); }

	/// @brief Mutable proxy exposing backend-aware whole-word operations.
	struct sideband_proxy {
		sideband_proxy() = delete;
		explicit constexpr sideband_proxy(bitfield_pack* src) BITFIELD_PACK_NOEXCEPT : data_(src) {}

		/// @brief Returns the owned backend object by reference.
		constexpr backend_t& backend() BITFIELD_PACK_NOEXCEPT { return data_->backend_; }
		/// @brief Loads the canonical storage word through the word-spec hook.
		constexpr storage_t load_storage_word() const BITFIELD_PACK_NOEXCEPT { return data_->load_storage_word(); }
		/// @brief Stores the canonical storage word through the word-spec hook.
		constexpr void store_storage_word(storage_t v) BITFIELD_PACK_NOEXCEPT { data_->store_storage_word(v); }

	private:
		bitfield_pack* data_;
	};

	/// @brief Const proxy exposing backend-aware whole-word observation.
	struct const_sideband_proxy {
		const_sideband_proxy() = delete;
		explicit constexpr const_sideband_proxy(const bitfield_pack* src) BITFIELD_PACK_NOEXCEPT : data_(src) {}

		/// @brief Returns the owned backend object by const reference.
		constexpr const backend_t& backend() const BITFIELD_PACK_NOEXCEPT { return data_->backend_; }
		/// @brief Loads the canonical storage word through the word-spec hook.
		constexpr storage_t load_storage_word() const BITFIELD_PACK_NOEXCEPT { return data_->load_storage_word(); }

	private:
		const bitfield_pack* data_;
	};

private:
	// Backend hook access.
	constexpr storage_t load_storage_word() const BITFIELD_PACK_NOEXCEPT {
		return word_spec_t::load_storage(backend_);
	}

	constexpr void store_storage_word(storage_t v) BITFIELD_PACK_NOEXCEPT {
		word_spec_t::store_storage(backend_, v);
	}

	// Owned backend object that ultimately stores the packed word. All field-oriented operations go
	// through the canonical `storage_t` representation regardless of the raw whole-word interface type.
	backend_t backend_{};
};

/// @brief Widths-only convenience alias for identity field specs.
template <class Word, std::size_t... Widths>
using bitfield_pack_bits = bitfield_pack<Word, std::size_t, bitfield_pack_detail::bitfield_field_width<Widths>...>;

/// @brief Convenience alias for the explicit word-spec form.
template <class StorageUInt, class RawIface = StorageUInt>
using bitfield_word_spec = bitfield_pack_detail::bitfield_word_spec<StorageUInt, RawIface>;

/// @brief Convenience alias for the fixed-width field spec with optional decoded type override.
template <std::size_t Width, typename DecodedT = void>
using bitfield_field_width = bitfield_pack_detail::bitfield_field_width<Width, DecodedT>;

/// @brief Convenience alias for the widths-only fixed-width identity field spec.
template <std::size_t Width>
using bitfield_field_spec = bitfield_field_width<Width>;

/// @brief Convenience alias for the trailing remainder field marker.
using bitfield_remainder = bitfield_pack_detail::bitfield_remainder;

} // namespace sw::universal
