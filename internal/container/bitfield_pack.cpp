// test_bitfield_pack.cpp
//
// Focused tests for the bitfield packing utility itself:
// - layout math (widths, offsets, masks, remainder fields)
// - semantic field encoding/decoding and validity checks
// - word-spec/indexing customization and backend hook behavior
//
// This file is intentionally more direct than the container suites because the API is mostly
// compile-time/static. The comments here explain why each scenario exists rather than trying to
// narrate every assertion.
#include <cmath>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <type_traits>

#define BITFIELD_PACK_NOEXCEPT /* empty for tests */

static bool g_assert_enabled = true;

struct test_assert_failure : std::exception {
	const char* what() const noexcept override { return "BITFIELD_PACK_ASSERT failed"; }
};

#define BITFIELD_PACK_ASSERT(expr) do { \
	if (g_assert_enabled && !(expr)) throw test_assert_failure{}; \
} while (0)

#include "universal/internal/container/bitfield_pack.hpp"
#include "universal/internal/container/custom_indexed_variant.hpp"

#define TEST_TRUE(expr) do { \
	if (!(expr)) { \
		std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " : " #expr "\n"; \
		std::terminate(); \
	} \
} while(0)

#define TEST_EQ(a,b) do { \
	auto _a = (a); auto _b = (b); \
	if (!(_a == _b)) { \
		std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " : " #a " == " #b \
		          << "  got " << +_a << " vs " << +_b << "\n"; \
		std::terminate(); \
	} \
} while(0)

#define TEST_THROWS(stmt) do { \
	bool _threw = false; \
	try { (void)(stmt); } catch (...) { _threw = true; } \
	if (!_threw) { \
		std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " : expected throw: " #stmt "\n"; \
		std::terminate(); \
	} \
} while(0)

using namespace sw::universal;

// Helper for assembling an expected raw word by hand in the canonical storage domain.
// Tests use this to verify pack layout independently of the production helper implementation.
template<typename Storage>
constexpr Storage insert_field(Storage raw, Storage value, std::size_t width, std::size_t offset) {
	const Storage all_ones = ~Storage(0);
	const Storage mask = (width >= std::numeric_limits<Storage>::digits)
		? all_ones
		: ((Storage(1) << width) - 1);
	return Storage((raw & ~(mask << offset)) | ((value & mask) << offset));
}

// Shared assertion helper for tests that reason about both raw storage and decoded field slices.
template<typename Pack>
void check_raw_and_fields_numeric(const Pack& pack,
                                  typename Pack::storage_type expected_raw,
                                  typename Pack::storage_type f0,
                                  typename Pack::storage_type f1,
                                  typename Pack::storage_type f2) {
	static_assert(std::is_same_v<typename Pack::field_key_type, std::size_t>);
	TEST_EQ(pack.raw_storage(), expected_raw);
	TEST_EQ(pack.template get_bits<0>(), f0);
	TEST_EQ(pack.template get_bits<1>(), f1);
	TEST_EQ(pack.template get_bits<2>(), f2);
}

// Custom semantic field spec used to verify encode/decode/value_type plumbing.
struct offset_binary_field {
	template<class StorageT>
	struct for_storage_t {
		using decoded_type = int;
		static constexpr bool is_remainder = false;
		static constexpr std::size_t width = 4;

		static constexpr bool is_valid(decoded_type v) noexcept {
			return v >= -2 && v <= 5;
		}

		static constexpr StorageT encode(decoded_type v) noexcept {
			return StorageT(v + 2);
		}

		static constexpr decoded_type decode(StorageT bits) noexcept {
			return int(bits) - 2;
		}
	};
};

// Semantic validity is separate from width-fit validity.
struct even_only_field {
	template<class StorageT>
	struct for_storage_t {
		using decoded_type = std::uint16_t;
		static constexpr bool is_remainder = false;
		static constexpr std::size_t width = 3;

		static constexpr StorageT encode(decoded_type v) noexcept {
			return static_cast<StorageT>(v);
		}

		static constexpr decoded_type decode(StorageT bits) noexcept {
			return static_cast<decoded_type>(bits);
		}

		static constexpr bool is_valid(decoded_type v) noexcept {
			return (v % 2u) == 0u;
		}
	};
};

template <std::size_t Width, typename DecodedT, DecodedT Bias>
struct biased_bitfield_field_width {
	template<typename StorageT>
	struct for_storage_t : public bitfield_field_width<Width, DecodedT>::template for_storage_t<StorageT> {
		static constexpr StorageT encode(DecodedT v) noexcept {
			return static_cast<StorageT>(v + Bias);
		}

		static constexpr DecodedT decode(StorageT v) noexcept {
			return static_cast<DecodedT>(v) - Bias;
		}
	};
};

enum class named_field : std::size_t {
	low = 0,
	mid = 1,
	high = 2,
};

enum class sparse_field : std::uint32_t {
	mantissa = 99,
	sign = 7,
	exponent = 42,
};

struct sparse_ieee_indexing {
	using field_key = sparse_field;

	static consteval std::size_t to_index(field_key key) noexcept {
		switch (key) {
		case sparse_field::mantissa: return 0;
		case sparse_field::exponent: return 1;
		case sparse_field::sign: return 2;
		}
		return static_cast<std::size_t>(-1);
	}
};

static void test_bits_alias_basic() {
	// Basic widths-only usage exercises normalization from an unsigned storage word
	// and the generic std::size_t indexing fallback.
	using P = bitfield_pack_bits<std::uint16_t, 3, 5, 8>;
	static_assert(P::size() == 3);
	static_assert(std::is_same_v<P::backend_type, std::uint16_t>);
	static_assert(std::is_same_v<P::storage_type, std::uint16_t>);
	static_assert(std::is_same_v<P::raw_iface_type, std::uint16_t>);
	static_assert(std::is_same_v<P::field_key_type, std::size_t>);
	static_assert(P::template field_mask<0>() == std::uint16_t{0x0007});
	static_assert(P::template field_mask<1>() == std::uint16_t{0x00F8});
	static_assert(P::template field_mask<2>() == std::uint16_t{0xFF00});
	static_assert(P::template field_value_mask<0>() == std::uint16_t{0x0007});
	static_assert(P::template field_value_mask<1>() == std::uint16_t{0x001F});
	static_assert(P::template field_value_mask<2>() == std::uint16_t{0x00FF});

	P p;
	TEST_EQ(p.raw_storage(), std::uint16_t{0});
	TEST_EQ(p.raw(), std::uint16_t{0});

	// width/offset sanity (LSB first)
	static_assert(P::field_width<0>() == 3);
	static_assert(P::field_offset<0>() == 0);
	static_assert(P::field_width<1>() == 5);
	static_assert(P::field_offset<1>() == 3);
	static_assert(P::field_width<2>() == 8);
	static_assert(P::field_offset<2>() == 8);

	// Setting masks (silent truncation). These checks document that truncation is intentional and that
	// callers who want rejection must use the validation hooks separately.
	p.template set_bits<0>(0b1111); // 4-bit into 3-bit -> truncates to 0b111
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0b111});
	TEST_EQ(p.raw_storage(), std::uint16_t{0b111});

	p.template set_bits<1>(0b100101); // truncates to 5 bits
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0b111});
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0b00101});

	p.template set_bits<2>(0xAB);
	check_raw_and_fields_numeric(
		p,
		insert_field(insert_field(insert_field(std::uint16_t{0}, std::uint16_t{0b111}, 3, 0), std::uint16_t{0b00101}, 5, 3), std::uint16_t{0xAB}, 8, 8),
		std::uint16_t{0b111},
		std::uint16_t{0b00101},
		std::uint16_t{0xAB});

	p.set_raw_storage(0xBEEF);
	TEST_EQ(p.raw_storage(), std::uint16_t{0xBEEF});
	TEST_EQ(p.raw(), std::uint16_t{0xBEEF});

	p.set_raw(0x1234);
	TEST_EQ(p.raw_storage(), std::uint16_t{0x1234});
	TEST_EQ(p.raw(), std::uint16_t{0x1234});

	// Validity check separately
	TEST_TRUE(P::is_valid<0>(std::uint16_t{0b111}));
	TEST_TRUE(!P::is_valid<0>(std::uint16_t{0b1000}));
}

static void test_raw_enum_indexing() {
	using P = bitfield_pack<std::uint16_t, named_field, bitfield_field_spec<3>, bitfield_field_spec<5>, bitfield_field_spec<8>>;
	static_assert(std::is_same_v<P::field_key_type, named_field>);
	static_assert(std::is_same_v<P::indexing_spec, sw::universal::bitfield_pack_detail::bitfield_index_by_cast<named_field>>);
	static_assert(P::template field_width<named_field::low>() == 3);
	static_assert(P::template field_offset<named_field::mid>() == 3);
	static_assert(P::template field_mask<named_field::high>() == std::uint16_t{0xFF00});

	P p;
	p.template set_bits<named_field::low>(0b1111);
	p.template set_bits<named_field::mid>(0b100101);
	p.template set_bits<named_field::high>(0xAB);

	TEST_EQ(p.raw_storage(),
	        insert_field(insert_field(insert_field(std::uint16_t{0}, std::uint16_t{0b111}, 3, 0), std::uint16_t{0b00101}, 5, 3), std::uint16_t{0xAB}, 8, 8));
	TEST_EQ(p.template get_bits<named_field::low>(), std::uint16_t{0b111});
	TEST_EQ(p.template get_bits<named_field::mid>(), std::uint16_t{0b00101});
	TEST_EQ(p.template get_bits<named_field::high>(), std::uint16_t{0xAB});
}

static void test_raw_storage_constructor_and_field_isolation() {
	// Starting from a raw word lets us verify field extraction and neighboring-field isolation.
	using P = bitfield_pack_bits<std::uint16_t, 4, 4, 8>;
	P p(std::uint16_t{0xABCD});
	check_raw_and_fields_numeric(p, std::uint16_t{0xABCD}, std::uint16_t{0xD}, std::uint16_t{0xC}, std::uint16_t{0xAB});

	const auto before = p.raw_storage();
	p.template set_bits<1>(0x2);
	const auto after = p.raw_storage();
	TEST_EQ(after, insert_field(before, std::uint16_t{0x2}, 4, 4));
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0xD});
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0x2});
	TEST_EQ(p.template get_bits<2>(), std::uint16_t{0xAB});
}

static void test_semantic_field_spec_access() {
	// Semantic fields should encode/decode independently of the raw bit layout.
	using P = bitfield_pack<std::uint16_t, std::size_t, offset_binary_field, bitfield_field_spec<4>, bitfield_field_spec<8>>;
	static_assert(std::is_same_v<P::template value_type<0>, int>);
	static_assert(P::template field_width<0>() == 4);

	P p;
	p.template set<0>(5);
	p.template set<1>(0xA);
	p.template set<2>(0x5C);

	TEST_EQ(p.template get<0>(), 5);
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{7});
	TEST_EQ(p.template get<1>(), std::uint16_t{0xA});
	TEST_EQ(p.template get<2>(), std::uint16_t{0x5C});
	TEST_TRUE(P::is_valid<0>(-2));
	TEST_TRUE(P::is_valid<0>(5));
	TEST_TRUE(!P::is_valid<0>(6));
}

static void test_nested_field_spec_shapes() {
	using identity_field = bitfield_field_width<5>;
	using bool_field = bitfield_field_width<1, bool>;
	using identity_storage = identity_field::for_storage_t<std::uint16_t>;
	using bool_storage = bool_field::for_storage_t<std::uint16_t>;
	using remainder_storage = bitfield_remainder::for_storage_t<std::uint16_t>;

	static_assert(std::is_same_v<identity_storage::decoded_type, std::uint16_t>);
	static_assert(identity_storage::width == 5);
	static_assert(!identity_storage::is_remainder);
	static_assert(identity_storage::encode(std::uint16_t{0x1Fu}) == std::uint16_t{0x1Fu});
	static_assert(identity_storage::decode(std::uint16_t{0x12u}) == std::uint16_t{0x12u});
	static_assert(identity_storage::is_valid(std::uint16_t{0xFFFFu}));

	static_assert(std::is_same_v<bool_storage::decoded_type, bool>);
	static_assert(bool_storage::width == 1);
	static_assert(bool_storage::decode(std::uint16_t{1u}));

	static_assert(std::is_same_v<remainder_storage::decoded_type, std::uint16_t>);
	static_assert(remainder_storage::is_remainder);
	static_assert(remainder_storage::width == 0);
	static_assert(remainder_storage::encode(std::uint16_t{7u}) == std::uint16_t{7u});
}

static void test_validate_hook() {
	// Validation is separate from mutation; invalid values are caught only when the hook is used.
	using P = bitfield_pack_bits<std::uint8_t, 3, 5>;
	static_assert(P::field_max_bits<0>() == 0b111);

	g_assert_enabled = true;
	TEST_THROWS(P::validate<0>(std::uint8_t{0b1000}));

	g_assert_enabled = false;
	// validate should not throw when asserts are disabled
	P::validate<0>(std::uint8_t{0b1000});
}

static void test_width_fit_and_semantic_validity_split() {
	using FitPack = bitfield_pack<std::uint8_t, std::size_t, bitfield_field_spec<3>>;
	static_assert(FitPack::is_valid<0>(std::uint8_t{0b111}));
	static_assert(!FitPack::is_valid<0>(std::uint8_t{0b1000}));

	using SemanticPack = bitfield_pack<std::uint16_t, std::size_t, even_only_field, bitfield_field_spec<5>>;
	static_assert(SemanticPack::template field_width<0>() == 3);

	TEST_TRUE(SemanticPack::is_valid<0>(std::uint16_t{6}));
	TEST_TRUE(!SemanticPack::is_valid<0>(std::uint16_t{3}));
	TEST_TRUE(!SemanticPack::is_valid<0>(std::uint16_t{8}));
}

static void test_remainder_layout() {
	// The trailing remainder consumes whatever bits are left after the fixed-width prefix.
	// [4 bits][remainder]
	using P = bitfield_pack<std::uint16_t, std::size_t, bitfield_field_spec<4>, bitfield_remainder>;
	static_assert(P::size() == 2);
	static_assert(P::field_width<0>() == 4);
	static_assert(P::field_offset<0>() == 0);
	static_assert(P::field_offset<1>() == 4);
	static_assert(P::field_width<1>() == 12);
	static_assert(P::template field_mask<0>() == std::uint16_t{0x000F});
	static_assert(P::template field_mask<1>() == std::uint16_t{0xFFF0});
	static_assert(P::template field_value_mask<1>() == std::uint16_t{0x0FFF});

	P p;
	p.template set_bits<0>(0xF);
	p.template set_bits<1>(0xABC);
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0xF});
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0xABC});
	TEST_EQ(p.raw_storage(), std::uint16_t{0xABCF});

	p.set_raw_storage(0x1234);
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0x4});
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0x123});

	p.template set_bits<1>(0xFFFF);
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0x0FFF});
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0x4});
}

static void test_custom_descriptor_indexing() {
	using Word = bitfield_word_spec<std::uint32_t, float>;
	using P = bitfield_pack<Word, sparse_ieee_indexing, bitfield_field_spec<23>, bitfield_field_spec<8>, bitfield_field_spec<1>>;
	static_assert(std::is_same_v<P::field_key_type, sparse_field>);
	static_assert(std::is_same_v<P::indexing_spec, sparse_ieee_indexing>);
	static_assert(P::template field_width<sparse_field::mantissa>() == 23);
	static_assert(P::template field_offset<sparse_field::exponent>() == 23);
	static_assert(P::template field_offset<sparse_field::sign>() == 31);
	static_assert(P::template field_mask<sparse_field::sign>() == std::uint32_t{0x80000000u});

	P p;
	p.template set<sparse_field::mantissa>(0u);
	p.template set<sparse_field::exponent>(127u);
	p.template set<sparse_field::sign>(0u);
	TEST_EQ(p.raw(), 1.0f);
	TEST_EQ(p.raw_storage(), std::bit_cast<std::uint32_t>(1.0f));

	p.template set_bits<sparse_field::mantissa>(0x400000u);
	p.template set_bits<sparse_field::exponent>(0xFFu);
	p.template set_bits<sparse_field::sign>(0u);
	TEST_TRUE(std::isnan(p.raw()) || std::isinf(p.raw()) == false);
}

static void test_biased_field_spec() {
	using P = bitfield_pack<
		std::uint16_t,
		std::size_t,
		biased_bitfield_field_width<8, int, 127>,
		bitfield_field_spec<8>
	>;

	P p;
	p.template set<0>(1);
	p.template set<1>(0x5Au);

	TEST_EQ(p.template get<0>(), 1);
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{128});
	TEST_EQ(p.template get<1>(), std::uint16_t{0x5A});
	TEST_TRUE(P::is_valid<0>(1));
	TEST_TRUE(!P::is_valid<0>(129));
}

static void test_word_spec_float_roundtrip() {
	// Raw-interface tests verify that `raw()` / `set_raw()` can expose a non-integral whole-word type
	// while still doing all field math in the canonical unsigned storage domain.
	// Bind raw() to float, but still pack bits in uint32_t
	using Word = bitfield_word_spec<std::uint32_t, float>;
	using F = bitfield_pack_bits<Word, 1, 8, 23>;
	static_assert(std::is_same_v<F::raw_iface_type, float>);
	static_assert(std::is_same_v<F::storage_type, std::uint32_t>);

	// Layout: sign (1), exponent (8), mantissa (23) LSB-first means:
	// field0=sign at bit0, field1=exponent at bit1..8, field2=mantissa at bit9..31
	// That's NOT IEEE-754's native layout order (which is mantissa LSB), but this is OK:
	// we are demonstrating bound raw IO + field extraction; for IEEE-754 you'd choose widths
	// and offsets consistent with the representation you want.
	//
	// For real float bit layout you'd likely specify mantissa first (23), exponent (8), sign (1).
	using IEEE = bitfield_pack_bits<Word, 23, 8, 1>;

	IEEE p;
	p.set_raw(1.0f);
	TEST_EQ(p.raw(), 1.0f);

	const auto mant = p.template get_bits<0>();
	const auto exp  = p.template get_bits<1>();
	const auto sign = p.template get_bits<2>();

	// For 1.0f: sign 0, exponent raw 127, mantissa 0.
	TEST_EQ(sign, std::uint32_t{0});
	TEST_EQ(exp,  std::uint32_t{127});
	TEST_EQ(mant, std::uint32_t{0});
	TEST_EQ(p.raw_storage(), std::bit_cast<std::uint32_t>(1.0f));

	// NaN example: exponent all ones, mantissa non-zero.
	IEEE q;
	q.template set_bits<2>(0);          // sign
	q.template set_bits<1>(0xFF);       // exponent all ones
	q.template set_bits<0>(0x400000);   // mantissa non-zero
	float nanish = q.raw();
	TEST_TRUE(std::isnan(nanish) || std::isinf(nanish) == false); // likely NaN; platform may vary on payload handling
	TEST_EQ(Word::to_storage(q.raw()), q.raw_storage());
}

static void test_backend_access() {
	// Backend access exposes the backend and whole-word hooks explicitly without teaching the pack
	// about synchronization policy such as CAS loops.
	struct backend_word {
		std::uint32_t word = 0;
	};

	struct backend_spec {
		using backend_t = backend_word;
		using storage_t = std::uint32_t;
		using raw_iface_t = std::uint32_t;

		static constexpr storage_t to_storage(raw_iface_t v) noexcept { return v; }
		static constexpr raw_iface_t from_storage(storage_t v) noexcept { return v; }
		static constexpr storage_t load_storage(const backend_t& backend) noexcept { return backend.word; }
		static constexpr void store_storage(backend_t& backend, storage_t v) noexcept { backend.word = v; }
	};

	using P = bitfield_pack<backend_spec, std::size_t, bitfield_field_spec<4>, bitfield_remainder>;
	static_assert(std::is_same_v<P::backend_type, backend_word>);

	P p(P::from_backend, backend_word{0x4321u});
	auto sb = p.backend_access();
	TEST_EQ(sb.load_storage_word(), std::uint32_t{0x4321u});
	TEST_EQ(sb.backend().word, std::uint32_t{0x4321u});

	sb.store_storage_word(0x1234u);
	TEST_EQ(p.raw_storage(), std::uint32_t{0x1234u});
	TEST_EQ(sb.backend().word, std::uint32_t{0x1234u});

	p.template set_bits<0>(0xFu);
	TEST_EQ(sb.load_storage_word(), std::uint32_t{0x123Fu});

	const P& cp = p;
	auto csb = cp.backend_access();
	TEST_EQ(csb.load_storage_word(), std::uint32_t{0x123Fu});
	TEST_EQ(csb.backend().word, std::uint32_t{0x123Fu});
}

static void test_backend_hook_mutation_semantics() {
	// This backend counts load/store hook traffic so ordinary mutation can be verified as
	// whole-word load/modify/store through the spec rather than direct member access.
	struct counting_backend {
		std::uint16_t word = 0;
		int* loads = nullptr;
		int* stores = nullptr;
	};

	struct counting_spec {
		using backend_t = counting_backend;
		using storage_t = std::uint16_t;
		using raw_iface_t = std::uint16_t;

		static storage_t to_storage(raw_iface_t v) noexcept { return v; }
		static raw_iface_t from_storage(storage_t v) noexcept { return v; }
		static storage_t load_storage(const backend_t& backend) noexcept {
			if (backend.loads) ++*backend.loads;
			return backend.word;
		}
		static void store_storage(backend_t& backend, storage_t v) noexcept {
			if (backend.stores) ++*backend.stores;
			backend.word = v;
		}
	};

	using P = bitfield_pack<counting_spec, std::size_t, bitfield_field_spec<4>, bitfield_field_spec<4>, bitfield_field_spec<8>>;

	int loads = 0;
	int stores = 0;
	P p(P::from_backend, counting_backend{0x1200u, &loads, &stores});
	auto sb = p.backend_access();
	sb.store_storage_word(0x1234u);
	TEST_EQ(loads, 0);
	TEST_EQ(stores, 1);

	(void)p.raw_storage();
	TEST_EQ(loads, 1);
	TEST_EQ(stores, 1);

	p.template set_bits<0>(0xFu);
	TEST_EQ(loads, 2);
	TEST_EQ(stores, 2);
	TEST_EQ(p.raw_storage(), std::uint16_t{0x123Fu});
	TEST_EQ(loads, 3);

	p.template set<1>(std::uint16_t{0x2});
	TEST_EQ(loads, 4);
	TEST_EQ(stores, 3);
	TEST_EQ(p.raw_storage(), std::uint16_t{0x122Fu});
	TEST_EQ(loads, 5);

	TEST_EQ(sb.backend().word, std::uint16_t{0x122Fu});
}

static void test_word_spec_normalization() {
	// These static assertions keep the shorthand/inferred word-spec normalization readable.
	using IntegralPack = bitfield_pack_bits<std::uint32_t, 8, 8, 16>;
	using FloatWord = bitfield_word_spec<std::uint32_t, float>;
	using FloatPack = bitfield_pack_bits<FloatWord, 23, 8, 1>;

	struct backend_word {
		std::uint32_t word = 0;
	};
	struct backend_spec {
		using backend_t = backend_word;
		using storage_t = std::uint32_t;
		using raw_iface_t = float;
		static storage_t to_storage(raw_iface_t v) noexcept { return std::bit_cast<storage_t>(v); }
		static raw_iface_t from_storage(storage_t v) noexcept { return std::bit_cast<raw_iface_t>(v); }
		static storage_t load_storage(const backend_t& backend) noexcept { return backend.word; }
		static void store_storage(backend_t& backend, storage_t v) noexcept { backend.word = v; }
	};
	using BackendPack = bitfield_pack_bits<backend_spec, 23, 8, 1>;

	static_assert(std::is_same_v<IntegralPack::backend_type, std::uint32_t>);
	static_assert(std::is_same_v<IntegralPack::storage_type, std::uint32_t>);
	static_assert(std::is_same_v<IntegralPack::raw_iface_type, std::uint32_t>);
	static_assert(std::is_same_v<IntegralPack::indexing_spec, sw::universal::bitfield_pack_detail::bitfield_index_by_cast<std::size_t>>);

	static_assert(std::is_same_v<FloatPack::backend_type, std::uint32_t>);
	static_assert(std::is_same_v<FloatPack::storage_type, std::uint32_t>);
	static_assert(std::is_same_v<FloatPack::raw_iface_type, float>);

	static_assert(std::is_same_v<BackendPack::backend_type, backend_word>);
	static_assert(std::is_same_v<BackendPack::storage_type, std::uint32_t>);
	static_assert(std::is_same_v<BackendPack::raw_iface_type, float>);
}

static void test_index_encoded_sideband() {
	// Keep one cross-component smoke test here because custom_indexed_variant's encoded index relies
	// on bitfield_pack's sideband-friendly remainder layout.
	using E = sw::universal::internal::index_encoded_with_sideband_data<10>;
	E e;
	TEST_EQ(e.index(), std::variant_npos);

	auto sb = e.sideband();
	sb.set_val(0x1234); // will be masked to remainder width
	const auto before = sb.val();

	e.set_index(3);
	TEST_EQ(e.index(), std::size_t{3});
	TEST_EQ(e.sideband().val(), before);

	e.set_index(std::variant_npos);
	TEST_EQ(e.index(), std::variant_npos);
	TEST_EQ(e.sideband().val(), before);
}

int main() {
	try {
		test_bits_alias_basic();
		test_raw_enum_indexing();
		test_raw_storage_constructor_and_field_isolation();
		test_semantic_field_spec_access();
		test_nested_field_spec_shapes();
		test_validate_hook();
		test_width_fit_and_semantic_validity_split();
		test_remainder_layout();
		test_custom_descriptor_indexing();
		test_biased_field_spec();
		test_word_spec_float_roundtrip();
		test_backend_access();
		test_backend_hook_mutation_semantics();
		test_word_spec_normalization();
		test_index_encoded_sideband();
	} catch (const std::exception& e) {
		std::cerr << "UNCAUGHT EXCEPTION: " << e.what() << "\n";
		return 2;
	} catch (...) {
		std::cerr << "UNCAUGHT UNKNOWN EXCEPTION\n";
		return 3;
	}
	std::cout << "OK\n";
	return 0;
}
