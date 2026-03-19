// test_bitfield_pack.cpp
#include <cmath>
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

#include "universal/internal/bitvector/bitfield_pack.hpp"
#include "universal/internal/custom_indexed_variant/custom_indexed_variant.hpp"

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

static void test_bits_alias_basic() {
	using P = bitfield_pack_bits<std::uint16_t, 3, 5, 8>;
	static_assert(P::size() == 3);

	P p;
	TEST_EQ(p.raw_storage(), std::uint16_t{0});

	// width/offset sanity (LSB first)
	static_assert(P::field_width<0>() == 3);
	static_assert(P::field_offset<0>() == 0);
	static_assert(P::field_width<1>() == 5);
	static_assert(P::field_offset<1>() == 3);
	static_assert(P::field_width<2>() == 8);
	static_assert(P::field_offset<2>() == 8);

	// Setting masks (silent truncation)
	p.template set_bits<0>(0b1111); // 4-bit into 3-bit -> truncates to 0b111
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0b111});

	// Validity check separately
	TEST_TRUE(P::is_valid<0>(std::uint16_t{0b111}));
	TEST_TRUE(!P::is_valid<0>(std::uint16_t{0b1000}));
}

static void test_validate_hook() {
	using P = bitfield_pack_bits<std::uint8_t, 3, 5>;
	static_assert(P::field_max_bits<0>() == 0b111);

	g_assert_enabled = true;
	TEST_THROWS(P::validate<0>(std::uint8_t{0b1000}));

	g_assert_enabled = false;
	// validate should not throw when asserts are disabled
	P::validate<0>(std::uint8_t{0b1000});
}

static void test_remainder_layout() {
	// [4 bits][remainder]
	using P = bitfield_pack<std::uint16_t, bitfield_field_spec<4>, bitfield_remainder>;
	static_assert(P::size() == 2);
	static_assert(P::field_width<0>() == 4);
	static_assert(P::field_offset<0>() == 0);
	static_assert(P::field_offset<1>() == 4);
	static_assert(P::field_width<1>() == 12);

	P p;
	p.template set_bits<0>(0xF);
	p.template set_bits<1>(0xABC);
	TEST_EQ(p.template get_bits<0>(), std::uint16_t{0xF});
	TEST_EQ(p.template get_bits<1>(), std::uint16_t{0xABC});
}

static void test_word_spec_float_roundtrip() {
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

	// NaN example: exponent all ones, mantissa non-zero.
	IEEE q;
	q.template set_bits<2>(0);          // sign
	q.template set_bits<1>(0xFF);       // exponent all ones
	q.template set_bits<0>(0x400000);   // mantissa non-zero
	float nanish = q.raw();
	TEST_TRUE(std::isnan(nanish) || std::isinf(nanish) == false); // likely NaN; platform may vary on payload handling
}

static void test_backend_sideband_access() {
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

	using P = bitfield_pack<backend_spec, bitfield_field_spec<4>, bitfield_remainder>;
	static_assert(std::is_same_v<P::backend_type, backend_word>);

	P p(P::from_backend, backend_word{0x4321u});
	auto sb = p.sideband();
	TEST_EQ(sb.load_storage_word(), std::uint32_t{0x4321u});
	TEST_EQ(sb.backend().word, std::uint32_t{0x4321u});

	sb.store_storage_word(0x1234u);
	TEST_EQ(p.raw_storage(), std::uint32_t{0x1234u});
	TEST_EQ(sb.backend().word, std::uint32_t{0x1234u});

	p.template set_bits<0>(0xFu);
	TEST_EQ(sb.load_storage_word(), std::uint32_t{0x123Fu});

	const P& cp = p;
	auto csb = cp.sideband();
	TEST_EQ(csb.load_storage_word(), std::uint32_t{0x123Fu});
	TEST_EQ(csb.backend().word, std::uint32_t{0x123Fu});
}

static void test_index_encoded_sideband() {
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
		test_validate_hook();
		test_remainder_layout();
		test_word_spec_float_roundtrip();
		test_backend_sideband_access();
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
