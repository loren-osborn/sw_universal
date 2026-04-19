#pragma once
// sso_vector.hpp
//
// The `sso_vector` family shares one storage core with two public facades:
//  - `sso_vector`: small-size-optimized vector with ordinary value semantics
//  - `sso_cow_vector`: same storage strategy plus copy-on-write heap sharing
//
// Both variants keep small vectors in-object and use the same inline/heap representation switching.
// The CoW facade alone carries the atomic ownership header, detach-on-write rules, and proxy-based
// mutable access. The non-CoW facade structurally omits that heap-sharing state and exposes ordinary
// mutable references/iterators.
//
// Intentional API/semantic differences from std::vector:
//  - `sso_cow_vector` mutable element access uses proxy references and non-contiguous random-access
//    iterators so reads can preserve sharing and writes can detach only when needed
//  - `sso_vector` mutable access is ordinary `T&` / `T*`
//  - both variants keep logical size in variant sideband metadata rather than in the heap block
//
// Lifetime model:
//  - Both inline and heap representations hold raw storage, not always-live T objects.
//  - Live elements always occupy the contiguous prefix [0, size()).
//  - Middle insert/erase preserve that live-prefix invariant by assignment-based shifting of
//    already-live elements; only the tail grows or shrinks object lifetime.
//
// Threading model:
//  - Like std::vector, element operations are not thread-safe against concurrent mutation.
//  - `sso_cow_vector` uses an atomic ownership header only for shared heap bookkeeping.
//  - The bitfield hook layer publishes whole ownership-header words; it is not a general CAS
//    abstraction. Ownership-transition compare/exchange logic stays local to `sso_cow_vector`.
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "universal/internal/container/bitfield_pack.hpp"
#include "universal/internal/container/custom_indexed_variant.hpp"

namespace sw { namespace universal {
namespace internal {

enum class zero_inline_policy {
	disallow,
	// This exists mainly for deliberate tests and benchmarks. Ordinary production configurations
	// should normally keep a non-zero inline buffer and leave the policy at `disallow`.
	allow,
};

template<std::size_t InlineCount, zero_inline_policy ZeroInlinePolicy>
inline constexpr bool zero_inline_policy_matches_v =
	((InlineCount == 0) == (ZeroInlinePolicy == zero_inline_policy::allow));

namespace sso_vector_detail {

template<class T>
concept sso_vector_copy_constructible =
	std::is_copy_constructible_v<T>;

template<class T>
concept sso_vector_assignable_from_const_ref =
	std::is_assignable_v<T&, const T&>;

template<class T>
concept sso_vector_default_initializable =
	std::default_initializable<T>;

inline constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
	return (a + b - 1) / b;
}

template<class T, class Allocator>
inline constexpr std::size_t default_inline_bytes() noexcept {
	// The default inline payload budget is chosen so the production-build object size stays aligned
	// with std::vector<T, Allocator>. This is best-effort and ABI-sensitive: different standard
	// library layouts, allocators, or debug modes may legitimately change the heuristic result.
	// Debug-only invariant fields must not feed back into this policy.
	constexpr std::size_t vec_sz = sizeof(std::vector<T, Allocator>);
	if constexpr (vec_sz <= 2 * sizeof(void*)) {
		return 0;
	} else {
		return vec_sz - 2 * sizeof(void*);
	}
}

template<class T, class Allocator>
inline constexpr std::size_t default_inline_elems() noexcept {
	// This simply converts the ABI-sensitive byte budget above into a count of whole T objects.
	// The resulting inline capacity is therefore heuristic as well, not a standard-mandated parity point.
	constexpr std::size_t bytes = default_inline_bytes<T, Allocator>();
	if constexpr (bytes == 0) return 0;
	return bytes / sizeof(T);
}

template<class T, class Allocator>
inline constexpr zero_inline_policy default_zero_inline_policy() noexcept {
	if constexpr (default_inline_elems<T, Allocator>() == 0) {
		return zero_inline_policy::allow;
	} else {
		return zero_inline_policy::disallow;
	}
}

using header_underlying_t = std::uint64_t;
using header_storage_t = std::atomic<header_underlying_t>;

enum header_field : std::size_t {
	UNSHAREABLE = 0,
	REFCOUNT = 1,
};

struct ownership_header_word_spec {
	using underlying_val_t = header_underlying_t;
	using formatted_val_t = header_underlying_t;
	using storage_t = header_storage_t;
	static constexpr bool directly_mutable = false;

	static constexpr underlying_val_t to_underlying_value(formatted_val_t v) noexcept { return v; }
	static constexpr formatted_val_t from_underlying_value(underlying_val_t v) noexcept { return v; }

	static underlying_val_t load_underlying_value(const storage_t& storage) noexcept {
		return storage.load(std::memory_order_acquire);
	}

	static void store_underlying_value(storage_t& storage, underlying_val_t v) noexcept {
		storage.store(v, std::memory_order_release);
	}
};

using ownership_header_bits = bitfield_pack<
	ownership_header_word_spec,
	header_field,
	bitfield_field_spec<1>,
	bitfield_remainder
>;
using ownership_header_scratch_bits = typename ownership_header_bits::template scratch_t<>;

static_assert(ownership_header_bits::template field_width<REFCOUNT>() > 0,
	"sso_vector: header refcount remainder must be non-zero width");

inline constexpr header_underlying_t make_header_underlying_value(bool shareable, std::uint64_t rc) noexcept {
	ownership_header_scratch_bits header{};
	header.set_all_masked(shareable ? 0u : 1u, static_cast<header_underlying_t>(rc));
	return header.underlying_value();
}

inline constexpr bool ownership_header_is_shareable(const ownership_header_scratch_bits& header) noexcept {
	return header.template get<UNSHAREABLE>() == 0;
}

inline constexpr std::uint64_t ownership_header_share_count(const ownership_header_scratch_bits& header) noexcept {
	return static_cast<std::uint64_t>(header.template get<REFCOUNT>());
}

template<class T, bool EnableCow>
struct heap_block;

template<class T>
struct heap_block<T, true> {
	static_assert(alignof(T) <= alignof(std::max_align_t),
		"sso_vector requires T alignment compatible with byte-rebound allocator");
	mutable ownership_header_bits ownership_header;
	std::size_t capacity = 0;
#ifndef NDEBUG
	std::size_t live_count = 0;
#endif

	~heap_block() {
#ifndef NDEBUG
		assert(live_count == 0 && "heap_block should not be destroyed while it still owns live elements");
#endif
	}
};

template<class T>
struct heap_block<T, false> {
	static_assert(alignof(T) <= alignof(std::max_align_t),
		"sso_vector requires T alignment compatible with byte-rebound allocator");
	std::size_t capacity = 0;
#ifndef NDEBUG
	std::size_t live_count = 0;
#endif

	~heap_block() {
#ifndef NDEBUG
		assert(live_count == 0 && "heap_block should not be destroyed while it still owns live elements");
#endif
	}
};

template<class T, bool EnableCow>
inline constexpr std::size_t heap_block_payload_offset() noexcept {
	return ceil_div(sizeof(heap_block<T, EnableCow>), alignof(T)) * alignof(T);
}

template<class T, bool EnableCow>
inline T* block_data(heap_block<T, EnableCow>* b) noexcept {
	auto* payload = reinterpret_cast<std::byte*>(b) + heap_block_payload_offset<T, EnableCow>();
	return std::launder(reinterpret_cast<T*>(payload));
}

template<class T, bool EnableCow>
inline const T* block_data(const heap_block<T, EnableCow>* b) noexcept {
	auto* payload = reinterpret_cast<const std::byte*>(b) + heap_block_payload_offset<T, EnableCow>();
	return std::launder(reinterpret_cast<const T*>(payload));
}

template<class T, bool EnableCow>
inline constexpr std::size_t heap_block_bytes(std::size_t capacity) noexcept {
	const std::size_t capped_capacity = (capacity == 0 ? 1 : capacity);
	return heap_block_payload_offset<T, EnableCow>() + capped_capacity * sizeof(T);
}

template<class T, bool EnableCow>
inline constexpr bool heap_block_payload_is_aligned() noexcept {
	return (heap_block_payload_offset<T, EnableCow>() % alignof(T)) == 0;
}

template<class T, bool EnableCow>
inline constexpr bool heap_block_payload_follows_header() noexcept {
	return heap_block_payload_offset<T, EnableCow>() >= sizeof(heap_block<T, EnableCow>);
}

template<class T, bool EnableCow>
inline constexpr bool heap_block_payload_formula_is_consistent() noexcept {
	return heap_block_bytes<T, EnableCow>(1) == heap_block_payload_offset<T, EnableCow>() + sizeof(T);
}

static_assert(heap_block_payload_is_aligned<int, true>(), "sso_vector heap payload offset must preserve T alignment");
static_assert(heap_block_payload_is_aligned<int, false>(), "sso_vector heap payload offset must preserve T alignment");
static_assert(heap_block_payload_follows_header<int, true>(), "sso_vector heap payload must start after the header object");
static_assert(heap_block_payload_follows_header<int, false>(), "sso_vector heap payload must start after the header object");
static_assert(heap_block_payload_formula_is_consistent<int, true>(), "sso_vector heap byte formula must match header-plus-payload layout");
static_assert(heap_block_payload_formula_is_consistent<int, false>(), "sso_vector heap byte formula must match header-plus-payload layout");

template<class T, bool EnableCow, class Allocator>
inline heap_block<T, EnableCow>* allocate_block(std::size_t capacity, Allocator& alloc) {
	using byte_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>;
	using byte_traits = std::allocator_traits<byte_alloc>;
	byte_alloc bytes_alloc(alloc);
	const std::size_t bytes = heap_block_bytes<T, EnableCow>(capacity);
	std::byte* mem = byte_traits::allocate(bytes_alloc, bytes);
	auto* b = ::new (mem) heap_block<T, EnableCow>{};
	if constexpr (EnableCow) {
		ownership_header_word_spec::store_underlying_value(
			b->ownership_header.storage(),
			make_header_underlying_value(true, 1));
	}
	b->capacity = capacity;
	return b;
}

template<class T, bool EnableCow, class Allocator>
inline void deallocate_block(heap_block<T, EnableCow>* b, Allocator& alloc) noexcept {
	if (!b) return;
	using byte_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>;
	using byte_traits = std::allocator_traits<byte_alloc>;
	byte_alloc bytes_alloc(alloc);
	const std::size_t bytes = heap_block_bytes<T, EnableCow>(b->capacity);
	b->~heap_block<T, EnableCow>();
	byte_traits::deallocate(bytes_alloc, reinterpret_cast<std::byte*>(b), bytes);
}

template<class T>
inline ownership_header_scratch_bits load_ownership_header_snapshot(const heap_block<T, true>* b) noexcept {
	return b->ownership_header.scratch_copy();
}

template<class T>
inline bool ownership_header_is_unique(const heap_block<T, true>* b) noexcept {
	return ownership_header_share_count(load_ownership_header_snapshot(b)) == 1;
}

template<class T>
inline bool try_add_shared_owner(const heap_block<T, true>* b) noexcept {
	auto& storage = b->ownership_header.storage();
	for (;;) {
		auto scratch = load_ownership_header_snapshot(b);
		const std::uint64_t rc = ownership_header_share_count(scratch);
		if (!ownership_header_is_shareable(scratch)) return false;
		assert(rc >= 1);
		if (rc == std::numeric_limits<std::uint64_t>::max()) return false;
		const header_underlying_t before = scratch.underlying_value();
		scratch.template set_masked<REFCOUNT>(static_cast<header_underlying_t>(rc + 1));
		header_underlying_t expected = before;
		if (storage.compare_exchange_weak(expected, scratch.underlying_value(), std::memory_order_acq_rel, std::memory_order_acquire)) {
			return true;
		}
	}
}

template<class T>
inline bool release_shared_owner(const heap_block<T, true>* b) noexcept {
	auto& storage = b->ownership_header.storage();
	for (;;) {
		auto scratch = load_ownership_header_snapshot(b);
		const std::uint64_t rc = ownership_header_share_count(scratch);
		assert(rc >= 1);
		const std::uint64_t next_rc = rc - 1;
		const header_underlying_t before = scratch.underlying_value();
		scratch.template set_masked<REFCOUNT>(static_cast<header_underlying_t>(next_rc));
		header_underlying_t expected = before;
		if (storage.compare_exchange_weak(expected, scratch.underlying_value(), std::memory_order_acq_rel, std::memory_order_acquire)) {
			return next_rc == 0;
		}
	}
}

template<class T>
inline void mark_unshareable(heap_block<T, true>* b) noexcept {
	auto& storage = b->ownership_header.storage();
	auto scratch = load_ownership_header_snapshot(b);
	assert(ownership_header_share_count(scratch) == 1 && "sso_vector: mark_unshareable requires unique ownership");
	if (!ownership_header_is_shareable(scratch)) return;
	const header_underlying_t before = scratch.underlying_value();
	scratch.template set_masked<UNSHAREABLE>(1u);
	header_underlying_t expected = before;
	// Under the documented precondition (`refcount == 1` on a uniquely owned block), no other owner
	// may legally race a refcount/shareability transition on this header. The single strong CAS is
	// therefore an invariant check on the unique-owner assumption, not a retry-based state transition.
	const bool cas_ok = storage.compare_exchange_strong(
		expected,
		scratch.underlying_value(),
		std::memory_order_acq_rel,
		std::memory_order_acquire);
	(void)cas_ok;
	assert(cas_ok && "sso_vector: mark_unshareable expected unique-owner CAS to succeed");
}

template<typename T, std::size_t N, typename Allocator, bool EnableCow, zero_inline_policy ZeroInlinePolicy>
class basic_sso_vector_core {
	static constexpr bool zero_inline_allowed = (ZeroInlinePolicy == zero_inline_policy::allow);
	static_assert((N == 0) == zero_inline_allowed,
		"sso_vector zero_inline_policy must exactly match whether InlineCount is zero");
	static_assert(sso_vector_copy_constructible<T>,
		"sso_vector requires copy-constructible value_type");
	static_assert(sso_vector_assignable_from_const_ref<T>,
		"sso_vector requires value_type assignable from const value_type&");

public:
	using value_type = T;
	using allocator_type = Allocator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using const_reference = const T&;
	using pointer = T*;
	using const_pointer = const T*;

private:
	using self_type = basic_sso_vector_core;
	using heap_block_type = heap_block<T, EnableCow>;

	struct inline_storage {
		static constexpr std::size_t inline_bytes = (N == 0 ? 1 : sizeof(T) * N);
		alignas(T) std::byte buf[inline_bytes]{};
#ifndef NDEBUG
		size_type live_count = 0;
#endif

		T* data() noexcept { return std::launder(reinterpret_cast<T*>(buf)); }
		const T* data() const noexcept { return std::launder(reinterpret_cast<const T*>(buf)); }

		~inline_storage() {
#ifndef NDEBUG
			assert(live_count == 0 && "inline_storage should not be destroyed while it still owns live elements");
#endif
		}
	};

	struct heap_storage {
		heap_block_type* block = nullptr;
		T* data() noexcept { return sso_vector_detail::block_data<T, EnableCow>(block); }
		const T* data() const noexcept { return sso_vector_detail::block_data<T, EnableCow>(block); }
		size_type capacity() const noexcept { return block ? block->capacity : 0; }
	};

	using variant_t = custom_indexed_variant<sideband_encoded_index, inline_storage, heap_storage>;
	using size_sideband_encoded_index = sideband_encoded_index<2>;
	static constexpr size_type max_encoded_size =
		static_cast<size_type>(size_sideband_encoded_index::sideband_max);

	static_assert(size_sideband_encoded_index::sideband_bits > 0,
		"sso_vector requires at least one sideband bit for logical size");
	static_assert(max_encoded_size >= static_cast<size_type>(N),
		"sso_vector sideband size encoding must at least represent the full inline range");

	static size_type sideband_to_size(std::size_t v) noexcept { return static_cast<size_type>(v); }
	static std::size_t size_to_sideband(size_type v) noexcept {
		assert(v <= max_encoded_size && "sso_vector logical size exceeds representable sideband range");
		return static_cast<std::size_t>(v);
	}

	size_type size_impl() const noexcept {
		return sideband_to_size(static_cast<std::size_t>(state_.sideband().get()));
	}

	void set_size_impl(size_type n) noexcept {
		state_.sideband().set(size_to_sideband(n));
	}

	bool is_inline_impl() const noexcept { return state_.index() == 0; }
	bool is_heap_impl() const noexcept { return state_.index() == 1; }

	inline_storage& inline_storage_impl() noexcept { return state_.template get<0>(); }
	const inline_storage& inline_storage_impl() const noexcept { return state_.template get<0>(); }
	heap_storage& heap_storage_impl() noexcept { return state_.template get<1>(); }
	const heap_storage& heap_storage_impl() const noexcept { return state_.template get<1>(); }

	heap_block_type* heap_block_impl() noexcept {
		return is_heap_impl() ? heap_storage_impl().block : nullptr;
	}
	const heap_block_type* heap_block_impl() const noexcept {
		return is_heap_impl() ? heap_storage_impl().block : nullptr;
	}

	size_type capacity_impl() const noexcept {
		return is_inline_impl() ? N : heap_storage_impl().capacity();
	}

	const T* data_const_impl() const noexcept {
		return is_inline_impl() ? inline_storage_impl().data() : heap_storage_impl().data();
	}

	T* data_mut_no_cow_impl() noexcept {
		return is_inline_impl() ? inline_storage_impl().data() : heap_storage_impl().data();
	}

#ifndef NDEBUG
	template<typename StorageOwner>
	static size_type& debug_live_count_ref(StorageOwner& owner) noexcept {
		return owner.live_count;
	}

	template<typename StorageOwner>
	static const size_type& debug_live_count_ref(const StorageOwner& owner) noexcept {
		return owner.live_count;
	}

	template<typename StorageOwner>
	static void debug_assert_construct_at(const StorageOwner& owner, size_type index) noexcept {
		assert(index == debug_live_count_ref(owner) && "new lifetimes should extend the live prefix at the end");
	}

	template<typename StorageOwner>
	static void debug_note_construct_at(StorageOwner& owner) noexcept {
		++debug_live_count_ref(owner);
	}

	template<typename StorageOwner>
	static void debug_assert_destroy_at(const StorageOwner& owner, size_type index) noexcept {
		assert(debug_live_count_ref(owner) > 0 && "destroy requires at least one live element");
		assert(index + 1 == debug_live_count_ref(owner) && "lifetimes should be destroyed from the end of the live prefix");
	}

	template<typename StorageOwner>
	static void debug_note_destroy_at(StorageOwner& owner) noexcept {
		--debug_live_count_ref(owner);
	}

	template<typename StorageOwner>
	static void debug_assert_overwrite_at(const StorageOwner& owner, size_type index) noexcept {
		assert(index < debug_live_count_ref(owner) && "overwrite should target an already-live element");
	}

	template<typename StorageOwner>
	static void debug_assert_shift_right(const StorageOwner& owner, size_type first, size_type last) noexcept {
		assert(first <= last && "shift_right range should be ordered");
		assert(last < debug_live_count_ref(owner) && "shift_right should stay within the live prefix");
	}

	template<typename StorageOwner>
	static void debug_assert_shift_left(const StorageOwner& owner, size_type first, size_type last, size_type count) noexcept {
		assert(first <= last && "shift_left range should be ordered");
		assert(count <= last - first && "shift_left count should fit inside the live range");
		assert(last <= debug_live_count_ref(owner) && "shift_left should stay within the live prefix");
	}
#else
	template<typename StorageOwner>
	static void debug_assert_construct_at(const StorageOwner&, size_type) noexcept {}
	template<typename StorageOwner>
	static void debug_note_construct_at(StorageOwner&) noexcept {}
	template<typename StorageOwner>
	static void debug_assert_destroy_at(const StorageOwner&, size_type) noexcept {}
	template<typename StorageOwner>
	static void debug_note_destroy_at(StorageOwner&) noexcept {}
	template<typename StorageOwner>
	static void debug_assert_overwrite_at(const StorageOwner&, size_type) noexcept {}
	template<typename StorageOwner>
	static void debug_assert_shift_right(const StorageOwner&, size_type, size_type) noexcept {}
	template<typename StorageOwner>
	static void debug_assert_shift_left(const StorageOwner&, size_type, size_type, size_type) noexcept {}
#endif

	static bool debug_const_iterator_in_closed_range(const T* it, const T* first, const T* last) noexcept {
		std::less<const T*> less;
		return !less(it, first) && !less(last, it);
	}

	const T& cref_at_unchecked(size_type index) const noexcept {
		assert(index < size_impl() && "sso_vector index should be in range");
		return data_const_impl()[index];
	}

	T& ref_at_unchecked(size_type index) noexcept {
		assert(index < size_impl() && "sso_vector index should be in range");
		return data_mut_no_cow_impl()[index];
	}

	template<class U>
	void set_at(size_type index, U&& value) {
		assert(index < size_impl() && "sso_vector set_at index should be in range");
		ensure_mutable_heap_impl();
		if (is_inline_impl()) {
			overwrite_at_impl(inline_storage_impl(), inline_storage_impl().data(), index, std::forward<U>(value));
		} else {
			overwrite_at_impl(*heap_storage_impl().block, heap_storage_impl().data(), index, std::forward<U>(value));
		}
	}

	template<typename StorageOwner, class U>
	void overwrite_at_impl(StorageOwner& owner, T* data, size_type index, U&& value) {
		debug_assert_overwrite_at(owner, index);
		data[index] = std::forward<U>(value);
	}

	template<typename StorageOwner, class... Args>
	void construct_at_impl(StorageOwner& owner, T* data, size_type index, Args&&... args) {
		debug_assert_construct_at(owner, index);
		std::allocator_traits<Allocator>::construct(alloc_, data + index, std::forward<Args>(args)...);
		debug_note_construct_at(owner);
	}

	template<typename StorageOwner>
	void destroy_at_impl(StorageOwner& owner, T* data, size_type index) noexcept {
		debug_assert_destroy_at(owner, index);
		std::allocator_traits<Allocator>::destroy(alloc_, data + index);
		debug_note_destroy_at(owner);
	}

	template<typename StorageOwner>
	void shift_right_assign_impl(StorageOwner& owner, T* data, size_type first, size_type last) {
		debug_assert_shift_right(owner, first, last);
		for (size_type i = last; i > first + 1; --i) {
			overwrite_at_impl(owner, data, i - 1, std::move_if_noexcept(data[i - 2]));
		}
	}

	template<typename StorageOwner>
	void shift_left_assign_impl(StorageOwner& owner, T* data, size_type first, size_type last, size_type count) {
		debug_assert_shift_left(owner, first, last, count);
		for (size_type i = first; i + count < last; ++i) {
			overwrite_at_impl(owner, data, i, std::move_if_noexcept(data[i + count]));
		}
	}

	template<typename StorageOwner>
	void destroy_range_impl(StorageOwner& owner, T* data, size_type first, size_type last) noexcept {
		while (last > first) {
			--last;
			destroy_at_impl(owner, data, last);
		}
	}

	template<typename StorageOwner>
	void copy_construct_range_impl(StorageOwner& owner, T* dst, const T* src, size_type n) {
		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				construct_at_impl(owner, dst, constructed, src[constructed]);
			}
		} catch (...) {
			destroy_range_impl(owner, dst, 0, constructed);
			throw;
		}
	}

	template<typename StorageOwner>
	void move_construct_range_impl(StorageOwner& owner, T* dst, T* src, size_type n) {
		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				construct_at_impl(owner, dst, constructed, std::move_if_noexcept(src[constructed]));
			}
		} catch (...) {
			destroy_range_impl(owner, dst, 0, constructed);
			throw;
		}
	}

	template<typename StorageOwner>
	void default_construct_appended_impl(StorageOwner& owner, T* d, size_type from, size_type to) {
		size_type constructed = 0;
		try {
			for (size_type i = from; i < to; ++i, ++constructed) {
				construct_at_impl(owner, d, i);
			}
		} catch (...) {
			destroy_range_impl(owner, d, from, from + constructed);
			throw;
		}
	}

	template<typename StorageOwner>
	void fill_construct_appended_impl(StorageOwner& owner, T* d, size_type from, size_type to, const T& value) {
		size_type constructed = 0;
		try {
			for (size_type i = from; i < to; ++i, ++constructed) {
				construct_at_impl(owner, d, i, value);
			}
		} catch (...) {
			destroy_range_impl(owner, d, from, from + constructed);
			throw;
		}
	}

	void release_heap_block_impl(heap_block_type* b, size_type constructed) noexcept {
		if (!b) return;
		if constexpr (EnableCow) {
			if (sso_vector_detail::release_shared_owner(b)) {
				T* d = sso_vector_detail::block_data<T, EnableCow>(b);
				destroy_range_impl(*b, d, 0, constructed);
				sso_vector_detail::deallocate_block<T, EnableCow>(b, alloc_);
			}
		} else {
			T* d = sso_vector_detail::block_data<T, EnableCow>(b);
			destroy_range_impl(*b, d, 0, constructed);
			sso_vector_detail::deallocate_block<T, EnableCow>(b, alloc_);
		}
	}

	void release_heap_impl() noexcept {
		if (!is_heap_impl()) return;
		auto& h = heap_storage_impl();
		if (!h.block) return;
		const size_type n = size_impl();
		auto* block = h.block;
		h.block = nullptr;
		release_heap_block_impl(block, n);
	}

	void ensure_unique_heap_impl() {
		if constexpr (!EnableCow) {
			return;
		} else {
			if (!is_heap_impl()) return;
			auto& h = heap_storage_impl();
			if (!h.block) return;

			const auto cur = sso_vector_detail::load_ownership_header_snapshot(h.block);
			const std::uint64_t rc = sso_vector_detail::ownership_header_share_count(cur);
			assert(rc >= 1);
			if (rc == 1) return;

			const size_type n = size_impl();
			auto* old_block = h.block;
			const size_type cap = old_block->capacity;

			auto* new_block = sso_vector_detail::allocate_block<T, EnableCow>(cap, alloc_);
			T* dst = sso_vector_detail::block_data<T, EnableCow>(new_block);
			const T* src = sso_vector_detail::block_data<T, EnableCow>(old_block);
			try {
				copy_construct_range_impl(*new_block, dst, src, n);
			} catch (...) {
				sso_vector_detail::deallocate_block<T, EnableCow>(new_block, alloc_);
				throw;
			}

			h.block = new_block;
			release_heap_block_impl(old_block, n);
		}
	}

	void reset_storage_impl() noexcept {
		const size_type n = size_impl();
		if (is_inline_impl()) {
			destroy_range_impl(inline_storage_impl(), inline_storage_impl().data(), 0, n);
			set_size_impl(0);
			return;
		}

		release_heap_impl();
		set_size_impl(0);
	}

	bool can_adopt_heap_storage_from_impl(const self_type& other) const noexcept {
		if constexpr (std::allocator_traits<Allocator>::is_always_equal::value) {
			return true;
		} else {
			return alloc_ == other.alloc_;
		}
	}

	void move_rebuild_from_impl(self_type&& other) {
		const size_type n = other.size_impl();
		if (n == 0) {
			state_.template emplace<0>();
			set_size_impl(0);
			other.reset_to_inline_empty_impl();
			return;
		}

		bool can_move_from_source = other.is_inline_impl();
		if (!can_move_from_source && other.is_heap_impl() && other.heap_storage_impl().block) {
			if constexpr (EnableCow) {
				can_move_from_source =
					sso_vector_detail::ownership_header_is_unique(other.heap_storage_impl().block);
			} else {
				can_move_from_source = true;
			}
		}

		if (n <= N) {
			state_.template emplace<0>();
			T* dst = inline_storage_impl().data();
			try {
				if (can_move_from_source) {
					move_construct_range_impl(inline_storage_impl(), dst, other.data_mut_no_cow_impl(), n);
				} else {
					copy_construct_range_impl(inline_storage_impl(), dst, other.data_const_impl(), n);
				}
			} catch (...) {
				set_size_impl(0);
				throw;
			}
			set_size_impl(n);
			other.reset_to_inline_empty_impl();
			return;
		}

		const size_type new_cap = other.is_heap_impl() ? other.capacity_impl() : growth_capacity(n, N);
		auto* new_block = sso_vector_detail::allocate_block<T, EnableCow>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data<T, EnableCow>(new_block);
		try {
			if (can_move_from_source) {
				move_construct_range_impl(*new_block, dst, other.data_mut_no_cow_impl(), n);
			} else {
				copy_construct_range_impl(*new_block, dst, other.data_const_impl(), n);
			}
		} catch (...) {
			sso_vector_detail::deallocate_block<T, EnableCow>(new_block, alloc_);
			throw;
		}

		state_.template emplace<1>();
		heap_storage_impl().block = new_block;
		set_size_impl(n);
		other.reset_to_inline_empty_impl();
	}

	void move_from_impl(self_type&& other) {
		const size_type n = other.size_impl();
		if (other.is_inline_impl()) {
			state_.template emplace<0>();
			T* dst = inline_storage_impl().data();
			T* src = other.inline_storage_impl().data();
			try {
				move_construct_range_impl(inline_storage_impl(), dst, src, n);
			} catch (...) {
				set_size_impl(0);
				throw;
			}
			set_size_impl(n);
			destroy_range_impl(other.inline_storage_impl(), src, 0, n);
			other.set_size_impl(0);
			return;
		}

		auto* block = other.heap_storage_impl().block;
		other.heap_storage_impl().block = nullptr;
		state_.template emplace<1>();
		heap_storage_impl().block = block;
		set_size_impl(n);
		other.state_.template emplace<0>();
		other.set_size_impl(0);
	}

	void promote_inline_to_heap_impl(size_type new_cap) {
		assert(is_inline_impl());
		if (new_cap < 1) new_cap = 1;

		auto* b = sso_vector_detail::allocate_block<T, EnableCow>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data<T, EnableCow>(b);
		T* src = inline_storage_impl().data();
		const size_type n = size_impl();
		try {
			move_construct_range_impl(*b, dst, src, n);
		} catch (...) {
			sso_vector_detail::deallocate_block<T, EnableCow>(b, alloc_);
			throw;
		}

		destroy_range_impl(inline_storage_impl(), src, 0, n);

		heap_storage h{};
		h.block = b;
		state_.template emplace<1>(h);
	}

	void reallocate_heap_impl(size_type new_cap) {
		assert(is_heap_impl());
		auto& h = heap_storage_impl();
		assert(h.block);

		ensure_unique_heap_impl();
		auto* old_block = h.block;
		const size_type n = size_impl();
		assert(new_cap >= n);

		auto* new_block = sso_vector_detail::allocate_block<T, EnableCow>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data<T, EnableCow>(new_block);
		T* src = sso_vector_detail::block_data<T, EnableCow>(old_block);
		try {
			move_construct_range_impl(*new_block, dst, src, n);
		} catch (...) {
			sso_vector_detail::deallocate_block<T, EnableCow>(new_block, alloc_);
			throw;
		}

		destroy_range_impl(*old_block, src, 0, n);
		sso_vector_detail::deallocate_block<T, EnableCow>(old_block, alloc_);

		h.block = new_block;
	}

	static size_type growth_capacity(size_type desired, size_type current) noexcept {
		const size_type doubled = current ? current * 2 : 1;
		return (std::max)(desired, doubled);
	}

	void ensure_capacity_impl(size_type desired) {
		const size_type cap = capacity_impl();
		if (desired <= cap) return;

		if (desired <= N && is_inline_impl()) return;

		if (is_inline_impl()) {
			promote_inline_to_heap_impl(growth_capacity(desired, N));
			return;
		}

		reallocate_heap_impl(growth_capacity(desired, cap));
	}

	void ensure_mutable_heap_impl() {
		if constexpr (EnableCow) {
			if (is_heap_impl()) ensure_unique_heap_impl();
		}
	}

	void reset_to_inline_empty_impl() noexcept {
		reset_storage_impl();
		state_.template emplace<0>();
		set_size_impl(0);
	}

	void copy_from_impl(const self_type& other) {
		if (other.is_inline_impl()) {
			state_.template emplace<0>();
			T* dst = inline_storage_impl().data();
			const T* src = other.inline_storage_impl().data();
			try {
				copy_construct_range_impl(inline_storage_impl(), dst, src, other.size());
			} catch (...) {
				set_size_impl(0);
				throw;
			}
			set_size_impl(other.size());
			return;
		}

		state_.template emplace<1>();
		auto* b = other.heap_storage_impl().block;
		assert(b);

		if constexpr (EnableCow) {
			if (sso_vector_detail::try_add_shared_owner(b)) {
				heap_storage_impl().block = b;
				set_size_impl(other.size());
				return;
			}
		}

		auto* nb = sso_vector_detail::allocate_block<T, EnableCow>(b->capacity, alloc_);
		T* dst = sso_vector_detail::block_data<T, EnableCow>(nb);
		const T* src = sso_vector_detail::block_data<T, EnableCow>(b);
		try {
			copy_construct_range_impl(*nb, dst, src, other.size());
		} catch (...) {
			sso_vector_detail::deallocate_block<T, EnableCow>(nb, alloc_);
			throw;
		}
		heap_storage_impl().block = nb;
		set_size_impl(other.size());
	}

	void clear_impl() noexcept {
		const size_type n = size_impl();
		if (n == 0) {
			set_size_impl(0);
			return;
		}

		if (is_inline_impl()) {
			destroy_range_impl(inline_storage_impl(), inline_storage_impl().data(), 0, n);
			set_size_impl(0);
			return;
		}

		if constexpr (EnableCow) {
			const auto hdr = sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block);
			if (sso_vector_detail::ownership_header_share_count(hdr) > 1) {
				release_heap_impl();
				state_.template emplace<0>();
				set_size_impl(0);
				return;
			}
		}

		destroy_range_impl(*heap_storage_impl().block, heap_storage_impl().data(), 0, n);
		set_size_impl(0);
	}

	void privatize_heap_impl() {
		if constexpr (EnableCow) {
			if (auto* block = heap_block_impl()) {
				if (!sso_vector_detail::ownership_header_is_unique(block)) {
					ensure_unique_heap_impl();
					block = heap_block_impl();
				}
				sso_vector_detail::mark_unshareable(block);
			}
		}
	}

	template<class... Args>
	void emplace_back_impl(Args&&... args) {
		const size_type n = size_impl();
		ensure_capacity_impl(n + 1);
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (is_inline_impl()) {
			construct_at_impl(inline_storage_impl(), d, n, std::forward<Args>(args)...);
		} else {
			construct_at_impl(*heap_storage_impl().block, d, n, std::forward<Args>(args)...);
		}
		set_size_impl(n + 1);
	}

	void pop_back_impl() {
		const size_type n = size_impl();
		if (n == 0) return;
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (is_inline_impl()) {
			destroy_at_impl(inline_storage_impl(), d, n - 1);
		} else {
			destroy_at_impl(*heap_storage_impl().block, d, n - 1);
		}
		set_size_impl(n - 1);
	}

	void resize_impl(size_type count) {
		const size_type n = size_impl();
		if (count == n) return;

		if (count < n) {
			ensure_mutable_heap_impl();
			T* d = data_mut_no_cow_impl();
			if (is_inline_impl()) destroy_range_impl(inline_storage_impl(), d, count, n);
			else destroy_range_impl(*heap_storage_impl().block, d, count, n);
			set_size_impl(count);
			return;
		}

		ensure_capacity_impl(count);
		ensure_mutable_heap_impl();
		if (is_inline_impl()) default_construct_appended_impl(inline_storage_impl(), data_mut_no_cow_impl(), n, count);
		else default_construct_appended_impl(*heap_storage_impl().block, data_mut_no_cow_impl(), n, count);
		set_size_impl(count);
	}

	void resize_impl(size_type count, const T& value) {
		const size_type n = size_impl();
		if (count == n) return;

		if (count < n) {
			ensure_mutable_heap_impl();
			T* d = data_mut_no_cow_impl();
			if (is_inline_impl()) destroy_range_impl(inline_storage_impl(), d, count, n);
			else destroy_range_impl(*heap_storage_impl().block, d, count, n);
			set_size_impl(count);
			return;
		}

		ensure_capacity_impl(count);
		ensure_mutable_heap_impl();
		if (is_inline_impl()) fill_construct_appended_impl(inline_storage_impl(), data_mut_no_cow_impl(), n, count, value);
		else fill_construct_appended_impl(*heap_storage_impl().block, data_mut_no_cow_impl(), n, count, value);
		set_size_impl(count);
	}

	void assign_fill_impl(size_type count, const T& value) {
		clear_impl();
		ensure_capacity_impl(count);
		if (count == 0) return;
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		size_type constructed = 0;
		try {
			for (; constructed < count; ++constructed) {
				if (is_inline_impl()) construct_at_impl(inline_storage_impl(), d, constructed, value);
				else construct_at_impl(*heap_storage_impl().block, d, constructed, value);
			}
		} catch (...) {
			if (is_inline_impl()) destroy_range_impl(inline_storage_impl(), d, 0, constructed);
			else destroy_range_impl(*heap_storage_impl().block, d, 0, constructed);
			throw;
		}
		set_size_impl(count);
	}

	template<typename InputIt>
	void assign_range_impl(InputIt first, InputIt last) {
		self_type rebuilt(alloc_);
		using category = typename std::iterator_traits<InputIt>::iterator_category;
		if constexpr (std::derived_from<category, std::forward_iterator_tag>) {
			rebuilt.reserve(static_cast<size_type>(std::distance(first, last)));
		}
		for (; first != last; ++first) {
			rebuilt.emplace_back(*first);
		}
		reset_storage_impl();
		move_from_impl(std::move(rebuilt));
	}

	void shrink_to_fit_impl() {
		const size_type n = size_impl();
		if (n == 0) {
			reset_to_inline_empty_impl();
			return;
		}
		if (is_inline_impl()) return;

		if (n <= N) {
			ensure_unique_heap_impl();
			inline_storage ni{};
			T* dst = ni.data();
			T* src = heap_storage_impl().data();
			move_construct_range_impl(ni, dst, src, n);
			clear_impl();
			release_heap_impl();
			state_.template emplace<0>();
			move_construct_range_impl(inline_storage_impl(), inline_storage_impl().data(), dst, n);
			destroy_range_impl(ni, dst, 0, n);
			set_size_impl(n);
			return;
		}

		reallocate_heap_impl(n);
	}

	template<class... Args>
	size_type emplace_at_impl(size_type idx, Args&&... args) {
		const size_type n = size_impl();
		assert(idx <= n);
		ensure_capacity_impl(n + 1);
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (idx == n) {
			if (is_inline_impl()) construct_at_impl(inline_storage_impl(), d, n, std::forward<Args>(args)...);
			else construct_at_impl(*heap_storage_impl().block, d, n, std::forward<Args>(args)...);
		} else {
			if (is_inline_impl()) construct_at_impl(inline_storage_impl(), d, n, std::move_if_noexcept(d[n - 1]));
			else construct_at_impl(*heap_storage_impl().block, d, n, std::move_if_noexcept(d[n - 1]));
			try {
				if (is_inline_impl()) shift_right_assign_impl(inline_storage_impl(), d, idx, n);
				else shift_right_assign_impl(*heap_storage_impl().block, d, idx, n);
				T tmp(std::forward<Args>(args)...);
				if (is_inline_impl()) overwrite_at_impl(inline_storage_impl(), d, idx, std::move(tmp));
				else overwrite_at_impl(*heap_storage_impl().block, d, idx, std::move(tmp));
			} catch (...) {
				if (is_inline_impl()) destroy_at_impl(inline_storage_impl(), d, n);
				else destroy_at_impl(*heap_storage_impl().block, d, n);
				throw;
			}
		}
		set_size_impl(n + 1);
		return idx;
	}

	size_type erase_one_impl(size_type idx) {
		const size_type n = size_impl();
		assert(idx < n && "sso_vector::erase position must be dereferenceable and belong to this vector");
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (is_inline_impl()) {
			shift_left_assign_impl(inline_storage_impl(), d, idx, n, 1);
			destroy_at_impl(inline_storage_impl(), d, n - 1);
		} else {
			shift_left_assign_impl(*heap_storage_impl().block, d, idx, n, 1);
			destroy_at_impl(*heap_storage_impl().block, d, n - 1);
		}
		set_size_impl(n - 1);
		return idx;
	}

	size_type erase_range_impl(size_type idx_first, size_type idx_last) {
		const size_type n = size_impl();
		assert(idx_first <= idx_last && "sso_vector::erase range must be ordered");
		assert(idx_last <= n && "sso_vector::erase range must lie within this vector");
		if (idx_first == idx_last) return idx_first;
		const size_type count = idx_last - idx_first;
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (is_inline_impl()) shift_left_assign_impl(inline_storage_impl(), d, idx_first, n, count);
		else shift_left_assign_impl(*heap_storage_impl().block, d, idx_first, n, count);
		if (is_inline_impl()) destroy_range_impl(inline_storage_impl(), d, n - count, n);
		else destroy_range_impl(*heap_storage_impl().block, d, n - count, n);
		set_size_impl(n - count);
		return idx_first;
	}

	template<typename AppendInserted>
	size_type rebuild_with_inserted_impl(size_type idx, size_type inserted_count, AppendInserted&& append_inserted) {
		const size_type n = size_impl();
		assert(idx <= n);

		self_type rebuilt(alloc_);
		rebuilt.reserve(n + inserted_count);
		for (size_type i = 0; i < idx; ++i) {
			rebuilt.emplace_back(cref_at_unchecked(i));
		}
		append_inserted(rebuilt);
		for (size_type i = idx; i < n; ++i) {
			rebuilt.emplace_back(cref_at_unchecked(i));
		}

		reset_storage_impl();
		move_from_impl(std::move(rebuilt));
		return idx;
	}

	size_type insert_fill_impl(size_type idx, size_type count, const T& value) {
		if (count == 0) return idx;
		return rebuild_with_inserted_impl(idx, count, [&](self_type& rebuilt) {
			for (size_type i = 0; i < count; ++i) {
				rebuilt.emplace_back(value);
			}
		});
	}

	template<typename InputIt>
	size_type insert_range_impl(size_type idx, InputIt first, InputIt last) {
		using category = typename std::iterator_traits<InputIt>::iterator_category;
		if constexpr (std::derived_from<category, std::forward_iterator_tag>) {
			const size_type count = static_cast<size_type>(std::distance(first, last));
			return rebuild_with_inserted_impl(idx, count, [&](self_type& rebuilt) {
				for (; first != last; ++first) {
					rebuilt.emplace_back(*first);
				}
			});
		} else {
			std::vector<T, Allocator> buffered(alloc_);
			for (; first != last; ++first) {
				buffered.push_back(*first);
			}
			return rebuild_with_inserted_impl(idx, static_cast<size_type>(buffered.size()), [&](self_type& rebuilt) {
				for (auto& value : buffered) {
					rebuilt.emplace_back(std::move(value));
				}
			});
		}
	}

	decltype(auto) mutable_reference_at_impl(size_type index) noexcept {
		if constexpr (EnableCow) {
			return reference_proxy(this, index);
		} else {
			return ref_at_unchecked(index);
		}
	}

	decltype(auto) mutable_data_impl() {
		if constexpr (EnableCow) {
			privatize_heap_impl();
		}
		return data_mut_no_cow_impl();
	}

public:
	class reference_proxy {
	public:
		reference_proxy() = delete;
		reference_proxy(self_type* owner, size_type index) noexcept
			: owner_(owner), index_(index) {}

		operator T() const { return owner_->cref_at_unchecked(index_); }

		reference_proxy& operator=(const T& v) {
			owner_->set_at(index_, v);
			return *this;
		}

		reference_proxy& operator=(T&& v) {
			owner_->set_at(index_, std::move(v));
			return *this;
		}

		reference_proxy& operator=(const reference_proxy& other) {
			T tmp = static_cast<T>(other);
			return (*this = std::move(tmp));
		}

		friend void swap(reference_proxy a, reference_proxy b) {
			T tmp = static_cast<T>(a);
			a = static_cast<T>(b);
			b = std::move(tmp);
		}

	private:
		self_type* owner_;
		size_type index_;
	};

	class iterator_proxy {
	public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using reference = reference_proxy;
		using pointer = void;

		iterator_proxy() = default;
		iterator_proxy(self_type* owner, size_type index) noexcept : owner_(owner), index_(index) {}

		size_type index() const noexcept { return index_; }
		self_type* owner() const noexcept { return owner_; }

		reference operator*() const noexcept { return reference(owner_, index_); }
		reference operator[](difference_type n) const noexcept { return reference(owner_, add_offset_impl(index_, n)); }

		iterator_proxy& operator++() noexcept { ++index_; return *this; }
		iterator_proxy operator++(int) noexcept { auto tmp = *this; ++(*this); return tmp; }
		iterator_proxy& operator--() noexcept { --index_; return *this; }
		iterator_proxy operator--(int) noexcept { auto tmp = *this; --(*this); return tmp; }

		iterator_proxy& operator+=(difference_type n) noexcept { index_ = add_offset_impl(index_, n); return *this; }
		iterator_proxy& operator-=(difference_type n) noexcept { index_ = subtract_offset_impl(index_, n); return *this; }

		friend iterator_proxy operator+(iterator_proxy it, difference_type n) noexcept { it += n; return it; }
		friend iterator_proxy operator+(difference_type n, iterator_proxy it) noexcept { it += n; return it; }
		friend iterator_proxy operator-(iterator_proxy it, difference_type n) noexcept { it -= n; return it; }
		friend difference_type operator-(const iterator_proxy& a, const iterator_proxy& b) noexcept {
			assert(a.owner_ == b.owner_ && "sso_vector iterator subtraction requires iterators into the same vector");
			return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
		}

		friend bool operator==(const iterator_proxy& a, const iterator_proxy& b) noexcept {
			return a.owner_ == b.owner_ && a.index_ == b.index_;
		}
		friend bool operator!=(const iterator_proxy& a, const iterator_proxy& b) noexcept { return !(a == b); }
		friend bool operator<(const iterator_proxy& a, const iterator_proxy& b) noexcept {
			assert(a.owner_ == b.owner_ && "sso_vector iterator ordering requires iterators into the same vector");
			return a.index_ < b.index_;
		}
		friend bool operator>(const iterator_proxy& a, const iterator_proxy& b) noexcept { return b < a; }
		friend bool operator<=(const iterator_proxy& a, const iterator_proxy& b) noexcept { return !(b < a); }
		friend bool operator>=(const iterator_proxy& a, const iterator_proxy& b) noexcept { return !(a < b); }

	private:
		static size_type add_offset_impl(size_type index, difference_type delta) noexcept {
			const difference_type next = static_cast<difference_type>(index) + delta;
			assert(next >= 0 && "sso_vector iterator advanced before begin()");
			return static_cast<size_type>(next);
		}

		static size_type subtract_offset_impl(size_type index, difference_type delta) noexcept {
			const difference_type next = static_cast<difference_type>(index) - delta;
			assert(next >= 0 && "sso_vector iterator decremented before begin()");
			return static_cast<size_type>(next);
		}

		self_type* owner_ = nullptr;
		size_type index_ = 0;
	};

	using reference = std::conditional_t<EnableCow, reference_proxy, T&>;
	using const_iterator = const T*;
	using iterator = std::conditional_t<EnableCow, iterator_proxy, T*>;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
	void debug_assert_valid_iterator_position(const iterator_proxy& pos, const char* message) const noexcept {
		assert(pos.owner() == this && message);
		assert(pos.index() <= size_impl() && message);
	}

	void debug_assert_valid_iterator_range(const iterator_proxy& first, const iterator_proxy& last, const char* message) const noexcept {
		assert(first.owner() == this && message);
		assert(last.owner() == this && message);
		assert(first.index() <= last.index() && message);
		assert(last.index() <= size_impl() && message);
	}

	void debug_assert_valid_const_iterator_position(const T* pos, const char* message) const noexcept {
		assert(debug_const_iterator_in_closed_range(pos, cbegin(), cend()) && message);
	}

	void debug_assert_valid_const_iterator_range(const T* first, const T* last, const char* message) const noexcept {
		assert(debug_const_iterator_in_closed_range(first, cbegin(), cend()) && message);
		assert(debug_const_iterator_in_closed_range(last, cbegin(), cend()) && message);
		const std::less<const T*> less{};
		(void)less;
		assert(!std::less<const T*>{}(last, first) && message);
	}

	template<typename InputIt>
	void debug_assert_not_self_range(InputIt, InputIt, const char*) const noexcept {}

	size_type const_iterator_offset_impl(const T* pos) const noexcept {
		return static_cast<size_type>(pos - cbegin());
	}

	size_type iterator_index_impl(const iterator_proxy& pos, const char* message) const noexcept {
		debug_assert_valid_iterator_position(pos, message);
		return pos.index();
	}

	size_type const_iterator_index_impl(const T* pos, const char* message) const noexcept {
		debug_assert_valid_const_iterator_position(pos, message);
		return const_iterator_offset_impl(pos);
	}

	std::pair<size_type, size_type> const_iterator_range_indices_impl(const T* first, const T* last, const char* message) const noexcept {
		debug_assert_valid_const_iterator_range(first, last, message);
		return {const_iterator_offset_impl(first), const_iterator_offset_impl(last)};
	}

	template<typename It>
	size_type mutable_iterator_index_impl(It pos, const char* message) const noexcept {
		if constexpr (EnableCow) {
			return iterator_index_impl(pos, message);
		} else {
			return const_iterator_index_impl(pos, message);
		}
	}

	template<typename It>
	std::pair<size_type, size_type> mutable_iterator_range_indices_impl(It first, It last, const char* message) const noexcept {
		if constexpr (EnableCow) {
			debug_assert_valid_iterator_range(first, last, message);
			return {first.index(), last.index()};
		} else {
			return const_iterator_range_indices_impl(first, last, message);
		}
	}

	iterator make_iterator_at(size_type index) noexcept {
		if constexpr (EnableCow) {
			return iterator(this, index);
		} else {
			return data_mut_no_cow_impl() + index;
		}
	}

	void debug_assert_not_self_range(const iterator_proxy& first, const iterator_proxy& last, const char* message) const noexcept {
		if (first.owner() == this || last.owner() == this) {
			assert(first.owner() == this && last.owner() == this && message);
			assert(false && message);
		}
	}

	void debug_assert_not_self_range(const T* first, const T* last, const char* message) const noexcept {
		if (debug_const_iterator_in_closed_range(first, cbegin(), cend()) ||
		    debug_const_iterator_in_closed_range(last, cbegin(), cend())) {
			debug_assert_valid_const_iterator_range(first, last, message);
			assert(false && message);
		}
	}

public:
	basic_sso_vector_core() noexcept(std::is_nothrow_default_constructible_v<Allocator> &&
	                                 std::is_nothrow_copy_constructible_v<Allocator>)
		: basic_sso_vector_core(Allocator()) {}

	explicit basic_sso_vector_core(const Allocator& alloc) noexcept(std::is_nothrow_copy_constructible_v<Allocator>)
		: alloc_(alloc), state_(std::in_place_index<0>) {
		set_size_impl(0);
	}

	basic_sso_vector_core(size_type count, const Allocator& alloc = Allocator())
		requires sso_vector_default_initializable<T>
		: basic_sso_vector_core(alloc) {
		resize(count);
	}

	basic_sso_vector_core(size_type count, const T& value, const Allocator& alloc = Allocator())
		: basic_sso_vector_core(alloc) {
		assign(count, value);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	basic_sso_vector_core(InputIt first, InputIt last, const Allocator& alloc = Allocator())
		: basic_sso_vector_core(alloc) {
		assign(first, last);
	}

	basic_sso_vector_core(std::initializer_list<T> init, const Allocator& alloc = Allocator())
		: basic_sso_vector_core(init.begin(), init.end(), alloc) {}

	basic_sso_vector_core(const basic_sso_vector_core& other)
		: alloc_(std::allocator_traits<Allocator>::select_on_container_copy_construction(other.alloc_))
		, state_(std::in_place_index<0>) {
		copy_from_impl(other);
	}

	basic_sso_vector_core(basic_sso_vector_core&& other)
		: alloc_(std::move(other.alloc_)), state_(std::in_place_index<0>) {
		set_size_impl(0);
		move_from_impl(std::move(other));
	}

	~basic_sso_vector_core() {
		reset_storage_impl();
	}

	basic_sso_vector_core& operator=(const basic_sso_vector_core& other) {
		if (this == &other) return *this;

		Allocator target_alloc = alloc_;
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			target_alloc = other.alloc_;
		}

		basic_sso_vector_core temp(target_alloc);
		temp.copy_from_impl(other);

		reset_storage_impl();
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			alloc_ = other.alloc_;
		}
		move_from_impl(std::move(temp));
		return *this;
	}

	basic_sso_vector_core& operator=(basic_sso_vector_core&& other) {
		if (this == &other) return *this;

		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
			reset_storage_impl();
			alloc_ = std::move(other.alloc_);
			move_from_impl(std::move(other));
			return *this;
		}

		if (can_adopt_heap_storage_from_impl(other)) {
			reset_storage_impl();
			move_from_impl(std::move(other));
			return *this;
		}

		basic_sso_vector_core temp(alloc_);
		temp.move_rebuild_from_impl(std::move(other));
		reset_storage_impl();
		move_from_impl(std::move(temp));
		return *this;
	}

	const_reference at(size_type pos) const {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return data_const_impl()[pos];
	}

	decltype(auto) at(size_type pos) {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return (*this)[pos];
	}

	decltype(auto) operator[](size_type pos) noexcept {
		assert(pos < size() && "sso_vector::operator[] should be in range");
		return mutable_reference_at_impl(pos);
	}

	const_reference operator[](size_type pos) const noexcept {
		assert(pos < size() && "sso_vector::operator[] should be in range");
		return cref_at_unchecked(pos);
	}

	const_reference front() const noexcept { return data_const_impl()[0]; }
	decltype(auto) front() noexcept { return (*this)[0]; }

	const_reference back() const noexcept { return data_const_impl()[size() - 1]; }
	decltype(auto) back() noexcept { return (*this)[size() - 1]; }

	const_pointer data() const noexcept { return data_const_impl(); }

	decltype(auto) data() {
		return mutable_data_impl();
	}

	iterator begin() noexcept {
		if constexpr (EnableCow) {
			return iterator(this, 0);
		} else {
			return data_mut_no_cow_impl();
		}
	}

	iterator end() noexcept {
		if constexpr (EnableCow) {
			return iterator(this, size());
		} else {
			return data_mut_no_cow_impl() + size();
		}
	}

	const_iterator begin() const noexcept { return data_const_impl(); }
	const_iterator end() const noexcept { return data_const_impl() + size(); }
	const_iterator cbegin() const noexcept { return begin(); }
	const_iterator cend() const noexcept { return end(); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	bool empty() const noexcept { return size() == 0; }
	size_type size() const noexcept { return size_impl(); }
	bool is_inline() const noexcept { return is_inline_impl(); }

	bool is_shared() const noexcept {
		if constexpr (EnableCow) {
			return share_count() > 1;
		} else {
			return false;
		}
	}

	bool is_shareable() const noexcept {
		if constexpr (EnableCow) {
			if (!is_heap_impl()) return true;
			return sso_vector_detail::ownership_header_is_shareable(
				sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block));
		} else {
			return true;
		}
	}

	size_type share_count() const noexcept {
		if constexpr (EnableCow) {
			if (!is_heap_impl()) return 1;
			return static_cast<size_type>(sso_vector_detail::ownership_header_share_count(
				sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block)));
		} else {
			return 1;
		}
	}

	size_type capacity() const noexcept { return capacity_impl(); }

	void reserve(size_type new_cap) {
		if (new_cap <= capacity()) return;
		ensure_capacity_impl(new_cap);
	}

	void shrink_to_fit() {
		shrink_to_fit_impl();
	}

	void clear() noexcept { clear_impl(); }
	void push_back(const T& value) { emplace_back(value); }
	void push_back(T&& value) { emplace_back(std::move(value)); }

	template<class... Args>
	decltype(auto) emplace_back(Args&&... args) {
		const size_type index = size();
		emplace_back_impl(std::forward<Args>(args)...);
		return mutable_reference_at_impl(index);
	}

	void pop_back() { pop_back_impl(); }

	void resize(size_type count)
		requires sso_vector_default_initializable<T> {
		resize_impl(count);
	}

	void resize(size_type count, const T& value) { resize_impl(count, value); }
	void assign(size_type count, const T& value) { assign_fill_impl(count, value); }

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	void assign(InputIt first, InputIt last) {
		debug_assert_not_self_range(first, last, "sso_vector::assign(range) does not accept iterators into *this");
		assign_range_impl(first, last);
	}

	void assign(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

	template<class... Args>
	iterator emplace(iterator pos, Args&&... args) {
		const size_type index = mutable_iterator_index_impl(pos, "sso_vector::emplace position must belong to this vector");
		return make_iterator_at(emplace_at_impl(index, std::forward<Args>(args)...));
	}

	iterator insert(iterator pos, const T& value) { return emplace(pos, value); }
	iterator insert(iterator pos, T&& value) { return emplace(pos, std::move(value)); }
	iterator insert(iterator pos, size_type count, const T& value) {
		const size_type index = mutable_iterator_index_impl(pos, "sso_vector::insert position must belong to this vector");
		return make_iterator_at(insert_fill_impl(index, count, value));
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(iterator pos, InputIt first, InputIt last) {
		debug_assert_not_self_range(first, last, "sso_vector::insert(range) does not accept iterators into *this");
		const size_type index = mutable_iterator_index_impl(pos, "sso_vector::insert position must belong to this vector");
		return make_iterator_at(insert_range_impl(index, first, last));
	}

	iterator insert(iterator pos, std::initializer_list<T> ilist) {
		return insert(pos, ilist.begin(), ilist.end());
	}

	iterator insert(const_iterator pos, const T& value) {
		return insert(make_iterator_at(const_iterator_index_impl(pos, "sso_vector::insert const_iterator position must belong to this vector")), value);
	}

	iterator insert(const_iterator pos, T&& value) {
		return insert(make_iterator_at(const_iterator_index_impl(pos, "sso_vector::insert const_iterator position must belong to this vector")), std::move(value));
	}

	iterator insert(const_iterator pos, size_type count, const T& value) {
		return insert(make_iterator_at(const_iterator_index_impl(pos, "sso_vector::insert const_iterator position must belong to this vector")), count, value);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(const_iterator pos, InputIt first, InputIt last) {
		debug_assert_not_self_range(first, last, "sso_vector::insert(range) does not accept iterators into *this");
		return insert(make_iterator_at(const_iterator_index_impl(pos, "sso_vector::insert const_iterator position must belong to this vector")), first, last);
	}

	iterator insert(const_iterator pos, std::initializer_list<T> ilist) {
		return insert(make_iterator_at(const_iterator_index_impl(pos, "sso_vector::insert const_iterator position must belong to this vector")), ilist);
	}

	iterator erase(iterator pos) {
		const size_type index = mutable_iterator_index_impl(pos, "sso_vector::erase position must belong to this vector");
		return make_iterator_at(erase_one_impl(index));
	}

	iterator erase(iterator first, iterator last) {
		const auto [first_index, last_index] =
			mutable_iterator_range_indices_impl(first, last, "sso_vector::erase range must belong to this vector and be ordered");
		return make_iterator_at(erase_range_impl(first_index, last_index));
	}

	iterator erase(const_iterator pos) {
		return make_iterator_at(erase_one_impl(const_iterator_index_impl(pos, "sso_vector::erase const_iterator position must belong to this vector")));
	}

	iterator erase(const_iterator first, const_iterator last) {
		const auto [first_index, last_index] =
			const_iterator_range_indices_impl(first, last, "sso_vector::erase const_iterator range must belong to this vector and be ordered");
		return make_iterator_at(erase_range_impl(first_index, last_index));
	}

	void swap(basic_sso_vector_core& other) {
		if (this == &other) return;
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value) {
			std::swap(alloc_, other.alloc_);
		} else if (!can_adopt_heap_storage_from_impl(other) && (is_heap_impl() || other.is_heap_impl())) {
			basic_sso_vector_core temp(alloc_);
			temp.move_rebuild_from_impl(std::move(*this));
			*this = std::move(other);
			other = std::move(temp);
			return;
		}
		if (is_heap_impl() && other.is_heap_impl()) {
			std::swap(heap_storage_impl().block, other.heap_storage_impl().block);
			const size_type this_size = size_impl();
			set_size_impl(other.size_impl());
			other.set_size_impl(this_size);
			return;
		}
		if (is_inline_impl() && other.is_inline_impl()) {
			basic_sso_vector_core tmp(std::move(other));
			other.move_from_impl(std::move(*this));
			reset_storage_impl();
			move_from_impl(std::move(tmp));
			return;
		}
		if (is_inline_impl() && other.is_heap_impl()) {
			inline_storage tmp{};
			const size_type this_size = size_impl();
			move_construct_range_impl(tmp, tmp.data(), inline_storage_impl().data(), this_size);
			destroy_range_impl(inline_storage_impl(), inline_storage_impl().data(), 0, this_size);

			auto* other_block = other.heap_storage_impl().block;
			const size_type other_size = other.size_impl();
			other.heap_storage_impl().block = nullptr;
			state_.template emplace<1>();
			heap_storage_impl().block = other_block;
			set_size_impl(other_size);

			other.state_.template emplace<0>();
			move_construct_range_impl(other.inline_storage_impl(), other.inline_storage_impl().data(), tmp.data(), this_size);
			destroy_range_impl(tmp, tmp.data(), 0, this_size);
			other.set_size_impl(this_size);
			return;
		}
		other.swap(*this);
	}

	allocator_type get_allocator() const noexcept { return alloc_; }

private:
	Allocator alloc_{};
	variant_t state_{std::in_place_index<0>};
};

} // namespace sso_vector_detail

template<typename T, std::size_t N, typename Allocator = std::allocator<T>,
         zero_inline_policy ZeroInlinePolicy = zero_inline_policy::disallow>
class sso_vector final
	: public sso_vector_detail::basic_sso_vector_core<T, N, Allocator, false, ZeroInlinePolicy> {
	using base = sso_vector_detail::basic_sso_vector_core<T, N, Allocator, false, ZeroInlinePolicy>;
public:
	using base::base;
	using base::operator=;
};

template<typename T, std::size_t N, typename Allocator = std::allocator<T>,
         zero_inline_policy ZeroInlinePolicy = zero_inline_policy::disallow>
class sso_cow_vector final
	: public sso_vector_detail::basic_sso_vector_core<T, N, Allocator, true, ZeroInlinePolicy> {
	using base = sso_vector_detail::basic_sso_vector_core<T, N, Allocator, true, ZeroInlinePolicy>;
public:
	using base::base;
	using base::operator=;
};

template<typename T, typename Allocator = std::allocator<T>>
using sso_vector_default =
	sso_vector<
		T,
		sso_vector_detail::default_inline_elems<T, Allocator>(),
		Allocator,
		sso_vector_detail::default_zero_inline_policy<T, Allocator>()
	>;

template<typename T, typename Allocator = std::allocator<T>>
using sso_cow_vector_default =
	sso_cow_vector<
		T,
		sso_vector_detail::default_inline_elems<T, Allocator>(),
		Allocator,
		sso_vector_detail::default_zero_inline_policy<T, Allocator>()
	>;

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator==(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                       const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator!=(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                       const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return !(a == b);
}

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator<(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                      const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator>(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                      const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return b < a;
}

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator<=(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                       const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return !(b < a);
}

template<template<typename, std::size_t, typename, zero_inline_policy> class Vector,
         typename T, std::size_t N, typename Allocator, zero_inline_policy ZeroInlinePolicy>
inline bool operator>=(const Vector<T, N, Allocator, ZeroInlinePolicy>& a,
                       const Vector<T, N, Allocator, ZeroInlinePolicy>& b) {
	return !(a < b);
}

}}} // namespace sw::universal::internal
