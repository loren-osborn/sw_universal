// bitfield_pack.hpp
#pragma once
/// @file bitfield_pack.hpp
/// @brief Pack multiple bitfields into a single machine word with compile-time layout.
///
/// Design notes (locked down):
/// - Two entry points:
///   * bitfield_pack<Word, FieldSpecs...> : "power" form (spec types)
///   * bitfield_pack_bits<Word, Widths...> : ergonomic alias (pure widths)
/// - "Word" can be:
///   * an unsigned integral type (shorthand; raw() returns the same type)
///   * bitfield_word_spec<StorageUInt, RawIface> (raw() returns RawIface, storage is StorageUInt)
/// - Setters mask and store (silent truncation). Validity is separate:
///   * is_valid<I>(value)
///   * validate<I>(value) -> assertion hook
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

namespace bitfield_pack_detail {

template <class T>
inline constexpr bool always_false_v = false;

/// @brief A word spec that binds internal storage to an unsigned integer, and raw() to a definite interface type.
/// @tparam StorageUInt unsigned integral storage word used for shifting/masking
/// @tparam RawIface trivially copyable type with same size as StorageUInt (e.g. float, double, a register struct)
template <class StorageUInt, class RawIface = StorageUInt>
struct bitfield_word_spec {
	static_assert(std::unsigned_integral<StorageUInt>, "StorageUInt must be unsigned integral");
	static_assert(std::is_trivially_copyable_v<RawIface>, "RawIface must be trivially copyable");
	static_assert(sizeof(RawIface) == sizeof(StorageUInt), "RawIface must have same size as StorageUInt");

	using storage_t = StorageUInt;
	using raw_iface_t = RawIface;

	static constexpr storage_t to_storage(raw_iface_t v) noexcept {
		if constexpr (std::is_same_v<raw_iface_t, storage_t>) {
			return v;
		} else {
			return std::bit_cast<storage_t>(v);
		}
	}

	static constexpr raw_iface_t from_storage(storage_t v) noexcept {
		if constexpr (std::is_same_v<raw_iface_t, storage_t>) {
			return v;
		} else {
			return std::bit_cast<raw_iface_t>(v);
		}
	}
};

/// @brief Normalize Word to a canonical word_spec type.
template <class Word>
struct normalize_word;

template <class Word>
	requires std::unsigned_integral<Word>
struct normalize_word<Word> {
	using type = bitfield_word_spec<Word, Word>;
};

template <class StorageUInt, class RawIface>
struct normalize_word<bitfield_word_spec<StorageUInt, RawIface>> {
	using type = bitfield_word_spec<StorageUInt, RawIface>;
};

template <class Word>
using normalize_word_t = typename normalize_word<Word>::type;

/// @brief Field spec concept: provides width (or remainder marker), and encode/decode/is_valid.
/// encode/decode/is_valid are in terms of the word's storage_t.
template <class FieldSpec, class StorageUInt>
concept bitfield_field_spec =
	std::unsigned_integral<StorageUInt> &&
	requires {
		{ FieldSpec::is_remainder } -> std::convertible_to<bool>;
	} &&
	(
		(FieldSpec::is_remainder) ||
		requires {
			{ FieldSpec::width } -> std::convertible_to<std::size_t>;
		}
	) &&
	requires(StorageUInt u) {
		// Value type for semantic access
		typename FieldSpec::template value_type<StorageUInt>;
		// Validate / encode / decode
		{ FieldSpec::template is_valid<StorageUInt>(typename FieldSpec::template value_type<StorageUInt>{}) } -> std::convertible_to<bool>;
		{ FieldSpec::template encode<StorageUInt>(typename FieldSpec::template value_type<StorageUInt>{}) } -> std::same_as<StorageUInt>;
		{ FieldSpec::template decode<StorageUInt>(u) } -> std::same_as<typename FieldSpec::template value_type<StorageUInt>>;
	};

/// @brief Identity-width field spec.
/// - width bits wide
/// - value_type defaults to StorageUInt
/// - encode/decode are identity (with cast)
/// - is_valid checks that the value fits in width bits (in the StorageUInt domain)
template <std::size_t Width>
struct bitfield_field_width {
	static_assert(Width > 0, "bitfield_field_width<Width>: Width must be > 0");
	static constexpr bool is_remainder = false;
	static constexpr std::size_t width = Width;

	template <class StorageUInt>
	using value_type = StorageUInt;

	template <class StorageUInt>
	static constexpr bool is_valid(value_type<StorageUInt> v) noexcept {
		static_assert(std::unsigned_integral<StorageUInt>, "StorageUInt must be unsigned integral");
		if constexpr (Width >= std::numeric_limits<StorageUInt>::digits) {
			return true;
		} else {
			const StorageUInt mask = (StorageUInt(1) << Width) - 1;
			return (StorageUInt(v) & ~mask) == 0;
		}
	}

	template <class StorageUInt>
	static constexpr StorageUInt encode(value_type<StorageUInt> v) noexcept {
		static_assert(std::unsigned_integral<StorageUInt>, "StorageUInt must be unsigned integral");
		return StorageUInt(v);
	}

	template <class StorageUInt>
	static constexpr value_type<StorageUInt> decode(StorageUInt bits) noexcept {
		static_assert(std::unsigned_integral<StorageUInt>, "StorageUInt must be unsigned integral");
		return value_type<StorageUInt>(bits);
	}
};

/// @brief Remainder field spec: width is computed as "storage_bits - sum(fixed widths)".
/// Remainder MUST be the final field.
/// encode/decode/is_valid behave like identity on the remainder bits.
struct bitfield_remainder {
	static constexpr bool is_remainder = true;

	template <class StorageUInt>
	using value_type = StorageUInt;

	template <class StorageUInt>
	static constexpr bool is_valid(value_type<StorageUInt>) noexcept {
		// Remainder's "validity" is layout-defined; in practice any value will be masked by the pack.
		return true;
	}

	template <class StorageUInt>
	static constexpr StorageUInt encode(value_type<StorageUInt> v) noexcept { return StorageUInt(v); }

	template <class StorageUInt>
	static constexpr value_type<StorageUInt> decode(StorageUInt bits) noexcept { return value_type<StorageUInt>(bits); }
};

template <class StorageUInt>
constexpr std::size_t storage_bits_v = std::numeric_limits<StorageUInt>::digits;

template <class StorageUInt>
constexpr StorageUInt all_ones_v = ~StorageUInt(0);

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
	return (std::size_t(Specs::is_remainder) + ... + 0u);
}

} // namespace bitfield_pack_detail

/// @brief Primary bitfield pack.
/// @tparam Word unsigned integral word OR bitfield_word_spec<StorageUInt, RawIface>
/// @tparam FieldSpecs field specifier types (bitfield_field_width<W>, bitfield_remainder, or future codecs)
template <class Word, class... FieldSpecs>
class bitfield_pack {
private:
	using word_spec_t = bitfield_pack_detail::normalize_word_t<Word>;
	using storage_t = typename word_spec_t::storage_t;

	static_assert(std::unsigned_integral<storage_t>, "bitfield_pack: storage_t must be unsigned integral");
	static_assert(sizeof...(FieldSpecs) > 0, "bitfield_pack: must define at least one field");
	static_assert(bitfield_pack_detail::remainder_count<FieldSpecs...>() <= 1, "bitfield_pack: at most one remainder field is allowed");

	// Validate that each FieldSpec conforms.
	static_assert((bitfield_pack_detail::bitfield_field_spec<FieldSpecs, storage_t> && ...),
	              "bitfield_pack: all FieldSpecs must satisfy the bitfield_field_spec concept");

	static constexpr std::size_t kStorageBits = bitfield_pack_detail::storage_bits_v<storage_t>;
	static constexpr std::size_t kFieldCount = sizeof...(FieldSpecs);

	template <std::size_t I>
	using spec_t = std::tuple_element_t<I, std::tuple<FieldSpecs...>>;

	template <std::size_t I>
	static constexpr bool is_remainder_v = spec_t<I>::is_remainder;

	template <std::size_t I>
	static consteval std::size_t declared_field_width() {
		if constexpr (is_remainder_v<I>) {
			return 0u;
		} else {
			return std::size_t(spec_t<I>::width);
		}
	}

	// Build widths[] and offsets[] at compile time.
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

public:
	using word_spec = word_spec_t;
	using storage_type = storage_t;
	using raw_iface_type = typename word_spec_t::raw_iface_t;

	/// @brief Number of fields.
	static consteval std::size_t size() noexcept { return kFieldCount; }

	/// @brief Construct with all bits zero.
	constexpr bitfield_pack() BITFIELD_PACK_NOEXCEPT : raw_(0) {}

	/// @brief Construct from raw storage bits.
	explicit constexpr bitfield_pack(storage_t raw_storage) BITFIELD_PACK_NOEXCEPT : raw_(raw_storage) {}

	/// @brief Get raw storage bits (always available).
	constexpr storage_t raw_storage() const BITFIELD_PACK_NOEXCEPT { return raw_; }

	/// @brief Set raw storage bits (always available).
	constexpr void set_raw_storage(storage_t v) BITFIELD_PACK_NOEXCEPT { raw_ = v; }

	/// @brief Get raw interface value (Word-dependent).
	/// For integral Word, this is the same as raw_storage().
	constexpr raw_iface_type raw() const BITFIELD_PACK_NOEXCEPT { return word_spec_t::from_storage(raw_); }

	/// @brief Set raw interface value (Word-dependent).
	/// For integral Word, this is the same as set_raw_storage().
	constexpr void set_raw(raw_iface_type v) BITFIELD_PACK_NOEXCEPT { raw_ = word_spec_t::to_storage(v); }

	/// @brief Field bit width at index I.
	template <std::size_t I>
	static consteval std::size_t field_width() noexcept {
		static_assert(I < kFieldCount, "field_width<I>: I out of range");
		return kLayout.widths[I];
	}

	/// @brief Field bit offset at index I (LSB-first packing).
	template <std::size_t I>
	static consteval std::size_t field_offset() noexcept {
		static_assert(I < kFieldCount, "field_offset<I>: I out of range");
		return kLayout.offsets[I];
	}

	/// @brief Unshifted value mask (width bits).
	template <std::size_t I>
	static consteval storage_t field_value_mask() noexcept {
		return bitfield_pack_detail::value_mask_unshifted<field_width<I>(), storage_t>();
	}

	/// @brief Shifted bit mask for the field within storage word.
	template <std::size_t I>
	static consteval storage_t field_mask() noexcept {
		constexpr std::size_t off = field_offset<I>();
		constexpr std::size_t w = field_width<I>();
		if constexpr (w == 0) return storage_t(0);
		if constexpr (w >= kStorageBits) return bitfield_pack_detail::all_ones_v<storage_t>;
		return storage_t(field_value_mask<I>() << off);
	}

	/// @brief Field's maximum encodable raw value (unshifted mask).
	template <std::size_t I>
	static consteval storage_t field_max_bits() noexcept {
		return field_value_mask<I>();
	}

	/// @brief Field spec's semantic value type.
	template <std::size_t I>
	using value_type = typename spec_t<I>::template value_type<storage_t>;

	/// @brief Extract raw bits (un-decoded) from field I.
	template <std::size_t I>
	constexpr storage_t get_bits() const BITFIELD_PACK_NOEXCEPT {
		static_assert(I < kFieldCount, "get_bits<I>: I out of range");
		constexpr std::size_t off = field_offset<I>();
		return storage_t((raw_ >> off) & field_value_mask<I>());
	}

	/// @brief Store raw bits into field I (masked to width; silent truncation).
	template <std::size_t I>
	constexpr void set_bits(storage_t bits) BITFIELD_PACK_NOEXCEPT {
		static_assert(I < kFieldCount, "set_bits<I>: I out of range");
		constexpr storage_t m = field_mask<I>();
		constexpr std::size_t off = field_offset<I>();
		raw_ = storage_t((raw_ & ~m) | ((storage_t(bits) << off) & m));
	}

	/// @brief Decode and return semantic value of field I.
	template <std::size_t I>
	constexpr value_type<I> get() const BITFIELD_PACK_NOEXCEPT {
		static_assert(I < kFieldCount, "get<I>: I out of range");
		return spec_t<I>::template decode<storage_t>(get_bits<I>());
	}

	/// @brief Encode and store semantic value of field I (masked; silent truncation).
	template <std::size_t I>
	constexpr void set(value_type<I> v) BITFIELD_PACK_NOEXCEPT {
		static_assert(I < kFieldCount, "set<I>: I out of range");
		const storage_t enc = spec_t<I>::template encode<storage_t>(v);
		set_bits<I>(enc);
	}

	/// @brief Check validity of semantic value for field I (does not consider masking).
	template <std::size_t I>
	static constexpr bool is_valid(value_type<I> v) noexcept {
		static_assert(I < kFieldCount, "is_valid<I>: I out of range");
		return spec_t<I>::template is_valid<storage_t>(v);
	}

	/// @brief Validate semantic value for field I using assertion hook.
	template <std::size_t I>
	static constexpr void validate(value_type<I> v) BITFIELD_PACK_NOEXCEPT {
		BITFIELD_PACK_ASSERT(is_valid<I>(v));
	}

private:
	storage_t raw_;
};

/// @brief Ergonomic alias: widths-only pack. Each width becomes an identity bitfield_field_width<W>.
template <class Word, std::size_t... Widths>
using bitfield_pack_bits = bitfield_pack<Word, bitfield_pack_detail::bitfield_field_width<Widths>...>;

/// Convenience aliases for users of the "power" form.
template <class StorageUInt, class RawIface = StorageUInt>
using bitfield_word_spec = bitfield_pack_detail::bitfield_word_spec<StorageUInt, RawIface>;

template <std::size_t Width>
using bitfield_field_spec = bitfield_pack_detail::bitfield_field_width<Width>;

using bitfield_remainder = bitfield_pack_detail::bitfield_remainder;

} // namespace sw::universal
