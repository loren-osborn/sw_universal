#pragma once
// sso_vector.hpp
//
// A std::vector-like container with:
//  - Small-size optimization (inline storage for N elements)
//  - Copy-on-write heap storage (shared buffer with atomic ownership header)
//  - Proxy element + proxy non-const iterators (vector<bool>-style)
//  - "Shareable" state to prevent sharing when an external non-const pointer has been handed out.
//
// The core intent is to act as a practical near-drop-in replacement for std::vector for the workloads
// it was designed for, while deliberately differing internally to keep small vectors in-object and
// large vectors cheaply shareable until mutation needs unique ownership. The mutable API is
// intentionally not "ordinary std::vector<T>& everywhere": non-const element access goes through
// proxies so reads can preserve sharing and writes can detach only when needed.
//
// Intentional API/semantic differences from std::vector:
//  - non-const operator[] / at() / front() / back() return proxy references, not T&
//  - non-const iterators are random-access proxy iterators, not contiguous iterators
//  - mutable data() is the explicit "raw mutable pointer escapes" operation; it detaches first and
//    then disables future sharing for that heap block
//  - const access remains ordinary contiguous const T* / const T& observation
//  - some bulk modifiers rebuild through a temporary vector to keep allocator/COW rules simple and
//    exception behavior reviewable; this favors correctness and small-vector ergonomics over
//    squeezing every last std::vector-style insertion optimization out of every iterator category
//
// Key design points:
//  - Two-layout representation uses custom_indexed_variant with sideband size.
//  - Heap sharing state uses bitfield_pack directly over a single atomic ownership header.
//    Layout: [UNSHAREABLE:1][REFCOUNT:remainder].
//  - Heap capacity is stored in the heap block header.
//  - Heap ownership header is mutable so refcount/shareability can change through const references.
//  - N==0 degrades to std::vector.
//  - Default N computed to match sizeof(std::vector<T, Allocator>) where possible.
//
// Lifetime model:
//  - Both inline and heap representations hold raw storage, not always-live T objects.
//  - Live elements always occupy the contiguous prefix [0, size()).
//  - Middle insert/erase preserve that live-prefix invariant by assignment-based shifting of
//    already-live elements; only the tail grows or shrinks object lifetime.
//  - Future maintainers must not treat the buffers as trivially relocatable byte arrays unless T's
//    semantics truly permit it; bytewise movement would be wrong for many non-trivial element types.
//
// Threading model (matches std::vector intent):
//  - Like std::vector, element operations are not thread-safe against concurrent mutation.
//  - Atomic ownership header is for shared heap bookkeeping only.
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "universal/internal/container/custom_indexed_variant.hpp"
#include "universal/internal/container/bitfield_pack.hpp"

namespace sw { namespace universal {
namespace internal {

namespace sso_vector_detail {

// ------------------------ helpers ------------------------

inline constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
	return (a + b - 1) / b;
}

template<class T, class Allocator>
inline constexpr std::size_t default_inline_bytes() noexcept {
	// The default inline payload budget is chosen so the production-build object size stays aligned
	// with std::vector<T, Allocator>. Debug-only invariant fields must not feed back into this policy.
	constexpr std::size_t vec_sz = sizeof(std::vector<T, Allocator>);
	if constexpr (vec_sz <= 2 * sizeof(void*)) {
		return 0;
	} else {
		return vec_sz - 2 * sizeof(void*);
	}
}

template<class T, class Allocator>
inline constexpr std::size_t default_inline_elems() noexcept {
	constexpr std::size_t bytes = default_inline_bytes<T, Allocator>();
	if constexpr (bytes == 0) return 0;
	return bytes / sizeof(T);
}

// ------------------------ heap ownership header ------------------------

using header_underlying_t = std::uint64_t;
using header_storage_t = std::atomic<header_underlying_t>;

enum header_field : std::size_t {
	UNSHAREABLE = 0,
	REFCOUNT = 1,
};

/// @brief Local word spec binding the ownership header bits directly to atomic storage.
/// @details The pack still does all bit math in `header_underlying_t`; only load/store route through
///          the atomic storage object. CAS loops stay local to sso_vector ownership transitions.
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

static_assert(ownership_header_bits::template field_width<REFCOUNT>() > 0, "sso_vector: header refcount remainder must be non-zero width");

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

template<class T>
struct heap_block;

template<class T>
inline ownership_header_scratch_bits load_ownership_header_snapshot(const heap_block<T>* b) noexcept;

template<class T>
inline bool try_add_shared_owner(const heap_block<T>* b) noexcept;

template<class T>
inline bool release_shared_owner(const heap_block<T>* b) noexcept;

template<class T>
inline void mark_unshareable(heap_block<T>* b) noexcept;

template<class T>
inline bool ownership_header_is_unique(const heap_block<T>* b) noexcept;

// ------------------------ heap block ------------------------

// Heap blocks are reference-counted storage nodes shared by multiple vectors while the block remains
// marked shareable. The vector's logical size is not stored in the block; it remains per-container in
// the variant sideband so sharing does not require rewriting per-instance size metadata.
template<class T>
struct heap_block {
	static_assert(alignof(T) <= alignof(std::max_align_t), "sso_vector requires T alignment compatible with byte-rebound allocator");
	mutable ownership_header_bits ownership_header;
	std::size_t capacity;
#ifndef NDEBUG
	std::size_t live_count = 0;
#endif
	alignas(T) std::byte data[sizeof(T)];

	~heap_block() {
#ifndef NDEBUG
		assert(live_count == 0 && "heap_block should not be destroyed while it still owns live elements");
#endif
	}
};

template<class T>
inline constexpr std::size_t heap_block_align() noexcept {
	return (std::max)(alignof(heap_block<T>), alignof(T));
}

template<class T>
inline T* block_data(heap_block<T>* b) noexcept {
	// Heap payload objects are repeatedly constructed into raw storage. Re-enter typed access through
	// one laundered choke point so lifetime restarts stay visible to optimizers and provenance-aware ABIs.
	return std::launder(reinterpret_cast<T*>(b->data));
}
template<class T>
inline const T* block_data(const heap_block<T>* b) noexcept {
	return std::launder(reinterpret_cast<const T*>(b->data));
}

template<class T>
inline std::size_t heap_block_bytes(std::size_t capacity) noexcept {
	const std::size_t capped_capacity = (capacity == 0 ? 1 : capacity);
	const std::size_t header_bytes = sizeof(heap_block<T>);
	return header_bytes + (capped_capacity - 1) * sizeof(T);
}

template<class T, class Allocator>
inline heap_block<T>* allocate_block(std::size_t capacity, Allocator& alloc) {
	using byte_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>;
	using byte_traits = std::allocator_traits<byte_alloc>;
	byte_alloc bytes_alloc(alloc);
	const std::size_t bytes = heap_block_bytes<T>(capacity);
	std::byte* mem = byte_traits::allocate(bytes_alloc, bytes);
	auto* b = ::new (mem) heap_block<T>{};
	ownership_header_word_spec::store_underlying_value(b->ownership_header.storage(), make_header_underlying_value(true, 1));
	b->capacity = capacity;
	b->data[0] = std::byte{0};
	return b;
}

template<class T, class Allocator>
inline void deallocate_block(heap_block<T>* b, Allocator& alloc) noexcept {
	if (!b) return;
	using byte_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>;
	using byte_traits = std::allocator_traits<byte_alloc>;
	byte_alloc bytes_alloc(alloc);
	const std::size_t bytes = heap_block_bytes<T>(b->capacity);
	b->~heap_block<T>();
	byte_traits::deallocate(bytes_alloc, reinterpret_cast<std::byte*>(b), bytes);
}

template<class T>
inline ownership_header_scratch_bits load_ownership_header_snapshot(const heap_block<T>* b) noexcept {
	return b->ownership_header.scratch_copy();
}

template<class T>
inline bool ownership_header_is_unique(const heap_block<T>* b) noexcept {
	return ownership_header_share_count(load_ownership_header_snapshot(b)) == 1;
}

template<class T>
inline bool try_add_shared_owner(const heap_block<T>* b) noexcept {
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
inline bool release_shared_owner(const heap_block<T>* b) noexcept {
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
inline void mark_unshareable(heap_block<T>* b) noexcept {
	auto& storage = b->ownership_header.storage();
	auto scratch = load_ownership_header_snapshot(b);
	assert(ownership_header_share_count(scratch) == 1 && "sso_vector: mark_unshareable requires unique ownership");
	if (!ownership_header_is_shareable(scratch)) return;
	const header_underlying_t before = scratch.underlying_value();
	scratch.template set_masked<UNSHAREABLE>(1u);
	header_underlying_t expected = before;
	const bool cas_ok = storage.compare_exchange_strong(expected, scratch.underlying_value(),
	                                                    std::memory_order_acq_rel,
	                                                    std::memory_order_acquire);
	assert(cas_ok && "sso_vector: mark_unshareable expected unique-owner CAS to succeed");
}

} // namespace sso_vector_detail

template<typename T, std::size_t N, typename Allocator = std::allocator<T>>
class sso_vector;

/// @brief `N == 0` specialization that falls back directly to `std::vector`.
template<typename T, typename Allocator>
class sso_vector<T, 0, Allocator> : public std::vector<T, Allocator> {
	using base = std::vector<T, Allocator>;
public:
	using base::base;
};

/// @brief Vector-like container with inline storage plus copy-on-write heap storage.
/// @tparam T Element type.
/// @tparam N Number of elements that fit in the inline buffer before heap promotion.
/// @tparam Allocator Allocator used for heap storage.
/// @details The representation is a two-alternative `custom_indexed_variant`:
/// - alternative 0: inline storage for up to `N` elements
/// - alternative 1: heap block with a shared reference-counted header
///
/// The intent is "vector-like for intended workloads", not "internally identical to std::vector".
/// Non-const element access uses proxy references/iterators so ordinary indexing does not
/// immediately force a mutable raw pointer. Mutable `data()` is the explicit pointer-escape path:
/// it detaches first, then clears future shareability before returning `T*`.
template<typename T, std::size_t N, typename Allocator>
class sso_vector {
public:
	static_assert(N > 0, "N==0 specialization should have been selected");

	using value_type = T;
	using allocator_type = Allocator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using const_reference = const T&;
	using pointer = T*;
	using const_pointer = const T*;

	class reference_proxy;
	class iterator_proxy;

private:
	/// @brief Inline storage alternative used while `size() <= N`.
	struct inline_storage {
		alignas(T) std::byte buf[sizeof(T) * N]{};
#ifndef NDEBUG
		size_type live_count = 0;
#endif
		// Inline elements live in raw storage and may be destroyed/reconstructed in place as the vector
		// mutates, so typed access re-enters through `std::launder`.
		T* data() noexcept { return std::launder(reinterpret_cast<T*>(buf)); }
		const T* data() const noexcept { return std::launder(reinterpret_cast<const T*>(buf)); }

		~inline_storage() {
#ifndef NDEBUG
			assert(live_count == 0 && "inline_storage should not be destroyed while it still owns live elements");
#endif
		}
	};

	/// @brief Heap storage alternative pointing at the shared heap block.
	struct heap_storage {
		sso_vector_detail::heap_block<T>* block = nullptr;
		T* data() noexcept { return sso_vector_detail::block_data(block); }
		const T* data() const noexcept { return sso_vector_detail::block_data(block); }
		size_type capacity() const noexcept { return block ? block->capacity : 0; }
	};

	/// @brief Representation variant: inline storage plus sideband size, or heap storage plus sideband size.
	using variant_t = custom_indexed_variant<sideband_encoded_index, inline_storage, heap_storage>;

	/// @brief Converts encoded sideband bits into the public `size_type`.
	static size_type sideband_to_size(std::size_t v) noexcept { return static_cast<size_type>(v); }
	/// @brief Converts public `size_type` into encoded sideband bits.
	static std::size_t size_to_sideband(size_type v) noexcept { return static_cast<std::size_t>(v); }

	// ------------------------ representation inspection ------------------------

	/// @brief Reads the logical size stored in the variant sideband.
	size_type size_impl() const noexcept {
		return sideband_to_size(static_cast<std::size_t>(state_.sideband().get()));
	}
	/// @brief Writes the logical size into the variant sideband.
	void set_size_impl(size_type n) noexcept {
		state_.sideband().set(size_to_sideband(n));
	}

	/// @brief Returns true when the active representation is inline storage.
	bool is_inline_impl() const noexcept { return state_.index() == 0; }
	/// @brief Returns true when the active representation is heap storage.
	bool is_heap_impl() const noexcept { return state_.index() == 1; }

	/// @brief Accesses the inline-storage representation.
	inline_storage& inline_storage_impl() noexcept { return state_.template get<0>(); }
	const inline_storage& inline_storage_impl() const noexcept { return state_.template get<0>(); }
	/// @brief Accesses the heap-storage representation.
	heap_storage& heap_storage_impl() noexcept { return state_.template get<1>(); }
	const heap_storage& heap_storage_impl() const noexcept { return state_.template get<1>(); }

	/// @brief Returns the active heap block pointer, or `nullptr` when not on the heap.
	sso_vector_detail::heap_block<T>* heap_block_impl() noexcept {
		return is_heap_impl() ? heap_storage_impl().block : nullptr;
	}
	const sso_vector_detail::heap_block<T>* heap_block_impl() const noexcept {
		return is_heap_impl() ? heap_storage_impl().block : nullptr;
	}

	/// @brief Returns the current capacity exposed by the active representation.
	size_type capacity_impl() const noexcept {
		return is_inline_impl() ? N : heap_storage_impl().capacity();
	}

	/// @brief Returns the raw element pointer without forcing uniqueness.
	const T* data_const_impl() const noexcept {
		return is_inline_impl() ? inline_storage_impl().data() : heap_storage_impl().data();
	}
	/// @brief Returns the mutable raw element pointer without clearing shareability or detaching.
	/// @warning Callers must already have enforced any required uniqueness policy.
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

	/// @brief Internal mutation model.
	/// @details Live elements must always occupy the contiguous prefix `[0, live_count)`.
	///          New lifetimes extend that prefix only at the end; destruction shrinks it only from
	///          the end. Overwrites and shifts operate only on already-live elements after detach /
	///          uniqueness has been established. This keeps the COW/shareability policy centralized
	///          while still allowing bulk loops to use raw pointers once write-safety is known.
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

	// ------------------------ lifetime helpers ------------------------

	/// @brief Destroys a half-open element range `[first, last)`.
	template<typename StorageOwner>
	void destroy_range_impl(StorageOwner& owner, T* data, size_type first, size_type last) noexcept {
		while (last > first) {
			--last;
			destroy_at_impl(owner, data, last);
		}
	}

	/// @brief Copy-constructs `n` elements into uninitialized storage.
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

	/// @brief Move-constructs `n` elements into uninitialized storage using `move_if_noexcept`.
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

	/// @brief Default-constructs appended elements in the half-open range `[from, to)`.
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

	/// @brief Copy-constructs appended elements initialized from `value` in `[from, to)`.
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

	// ------------------------ heap/state transitions ------------------------

	/// @brief Releases one reference to a heap block, destroying elements and deallocating if it becomes unique-to-zero.
	/// @details This is a teardown path only. It must never clone or detach: destroying one sharer of
	///          a shared block should just decrement the refcount unless this was the last owner.
	void release_heap_block_impl(sso_vector_detail::heap_block<T>* b, size_type constructed) noexcept {
		if (!b) return;
		if (sso_vector_detail::release_shared_owner(b)) {
			T* d = sso_vector_detail::block_data(b);
			destroy_range_impl(*b, d, 0, constructed);
			sso_vector_detail::deallocate_block(b, alloc_);
		}
	}

	/// @brief Releases the active heap representation, if any.
	/// @details Like `release_heap_block_impl`, this is a release/reset helper, not a mutation helper.
	void release_heap_impl() noexcept {
		if (!is_heap_impl()) return;
		auto& h = heap_storage_impl();
		if (!h.block) return;
		const size_type n = size_impl();
		auto* block = h.block;
		h.block = nullptr;
		release_heap_block_impl(block, n);
	}

	/// @brief Detaches from shared heap storage when the active block has `refcount > 1`.
	/// @details Shareability controls whether future copies may share; it does not prevent detaching
	///          for mutation. This helper allocates a fresh heap block and copies the active payload.
	///          It is the "clone for mutation" path, in contrast to release/reset helpers which must
	///          never clone. Construction of the replacement block completes before publication.
	void ensure_unique_heap_impl() {
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

		auto* new_block = sso_vector_detail::allocate_block<T>(cap, alloc_);
		T* dst = sso_vector_detail::block_data(new_block);
		const T* src = sso_vector_detail::block_data(old_block);
		if constexpr (std::is_copy_constructible_v<T>) {
			try {
				copy_construct_range_impl(*new_block, dst, src, n);
			} catch (...) {
				sso_vector_detail::deallocate_block(new_block, alloc_);
				throw;
			}
		} else {
			sso_vector_detail::deallocate_block(new_block, alloc_);
			throw std::logic_error("sso_vector: detaching shared storage requires copy-constructible value_type");
		}

		h.block = new_block;
		release_heap_block_impl(old_block, n);
	}

	/// @brief Releases owned state without cloning shared heap storage.
	/// @details Destructor and assignment both rely on this staying a pure release path.
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

	/// @brief Returns true when this container may directly take ownership of another heap allocation.
	/// @details Direct heap-block transfer is only allocator-correct when allocators propagate, compare
	///          equal, or are always interchangeable by trait.
	bool can_adopt_heap_storage_from_impl(const sso_vector& other) const noexcept {
		if constexpr (std::allocator_traits<Allocator>::is_always_equal::value) {
			return true;
		} else {
			return alloc_ == other.alloc_;
		}
	}

	/// @brief Rebuilds the observable state from another vector using this container's allocator.
	/// @details This is the allocator-safe fallback when direct heap ownership transfer is not allowed.
	///          Shared heap sources are copied so sibling sharers are unaffected; unique sources are
	///          moved element-wise and then released through their original allocator.
	void move_rebuild_from_impl(sso_vector&& other) {
		const size_type n = other.size_impl();
		if (n == 0) {
			state_.template emplace<0>();
			set_size_impl(0);
			other.reset_to_inline_empty_impl();
			return;
		}

		const bool can_move_from_source =
			other.is_inline_impl() ||
			(other.is_heap_impl() && other.heap_storage_impl().block &&
			 sso_vector_detail::ownership_header_is_unique(other.heap_storage_impl().block));

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
		auto* new_block = sso_vector_detail::allocate_block<T>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data(new_block);
		try {
			if (can_move_from_source) {
				move_construct_range_impl(*new_block, dst, other.data_mut_no_cow_impl(), n);
			} else {
				copy_construct_range_impl(*new_block, dst, other.data_const_impl(), n);
			}
		} catch (...) {
			sso_vector_detail::deallocate_block(new_block, alloc_);
			throw;
		}

		state_.template emplace<1>();
		heap_storage_impl().block = new_block;
		set_size_impl(n);
		other.reset_to_inline_empty_impl();
	}

	/// @brief Moves the observable state out of another vector without byte-moving live inline elements.
	/// @warning Heap-block stealing is only allocator-correct when the caller has already established
	///          that direct ownership transfer is allowed.
	void move_from_impl(sso_vector&& other) {
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

	/// @brief Promotes inline storage to a fresh heap block with capacity `new_cap`.
	/// @details Inline elements are live objects in raw inline storage, so promotion must transfer them
	///          element-wise and then explicitly destroy the old inline lifetimes.
	void promote_inline_to_heap_impl(size_type new_cap) {
		assert(is_inline_impl());
		if (new_cap < 1) new_cap = 1;

		auto* b = sso_vector_detail::allocate_block<T>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data(b);
		T* src = inline_storage_impl().data();
		const size_type n = size_impl();
		try {
			move_construct_range_impl(*b, dst, src, n);
		} catch (...) {
			sso_vector_detail::deallocate_block(b, alloc_);
			throw;
		}

		destroy_range_impl(inline_storage_impl(), src, 0, n);

		heap_storage h{};
		h.block = b;
		state_.template emplace<1>(h);
	}

	/// @brief Reallocates the active heap representation into a fresh block with capacity `new_cap`.
	/// @details Reallocation first forces uniqueness so the old block can be treated as exclusively owned
	///          storage during transfer and teardown.
	void reallocate_heap_impl(size_type new_cap) {
		assert(is_heap_impl());
		auto& h = heap_storage_impl();
		assert(h.block);

		ensure_unique_heap_impl();
		auto* old_block = h.block;
		const size_type n = size_impl();
		assert(new_cap >= n);

		auto* new_block = sso_vector_detail::allocate_block<T>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data(new_block);
		T* src = sso_vector_detail::block_data(old_block);
		try {
			move_construct_range_impl(*new_block, dst, src, n);
		} catch (...) {
			sso_vector_detail::deallocate_block(new_block, alloc_);
			throw;
		}

		destroy_range_impl(*old_block, src, 0, n);
		sso_vector_detail::deallocate_block(old_block, alloc_);

		h.block = new_block;
	}

	/// @brief Computes conceptual geometric growth capacity for heap transitions.
	static size_type growth_capacity(size_type desired, size_type current) noexcept {
		const size_type doubled = current ? current * 2 : 1;
		return (std::max)(desired, doubled);
	}

	/// @brief Ensures capacity for `desired` elements, promoting or reallocating as needed.
	void ensure_capacity_impl(size_type desired) {
		const size_type cap = capacity_impl();
		if (desired <= cap) return;

		if (desired <= N && is_inline_impl()) {
			return;
		}

		if (is_inline_impl()) {
			promote_inline_to_heap_impl(growth_capacity(desired, N));
			return;
		}

		reallocate_heap_impl(growth_capacity(desired, cap));
	}

	/// @brief Ensures the active heap block is uniquely owned before mutation.
	void ensure_mutable_heap_impl() {
		if (is_heap_impl()) ensure_unique_heap_impl();
	}

	/// @brief Resets the container to an empty inline-storage state.
	void reset_to_inline_empty_impl() noexcept {
		reset_storage_impl();
		state_.template emplace<0>();
		set_size_impl(0);
	}

	/// @brief Copies the observable contents/state from another vector, sharing heap storage only when allowed.
	/// @details Heap-backed copies retain sharing only while the source block is still marked shareable.
	void copy_from_impl(const sso_vector& other) {
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

		if (sso_vector_detail::try_add_shared_owner(b)) {
			heap_storage_impl().block = b;
			set_size_impl(other.size());
			return;
		}

		auto* nb = sso_vector_detail::allocate_block<T>(b->capacity, alloc_);
		T* dst = sso_vector_detail::block_data(nb);
		const T* src = sso_vector_detail::block_data(b);
		try {
			copy_construct_range_impl(*nb, dst, src, other.size());
		} catch (...) {
			sso_vector_detail::deallocate_block(nb, alloc_);
			throw;
		}
		heap_storage_impl().block = nb;
		set_size_impl(other.size());
	}

	/// @brief Destroys all active elements while preserving the current representation shell.
	/// @details Clearing a shared heap-backed vector must not detach just to destroy "its own" elements.
	///          Instead it drops its reference and becomes empty, leaving surviving sharers in charge of
	///          eventual payload destruction.
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

		const auto hdr = sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block);
		if (sso_vector_detail::ownership_header_share_count(hdr) > 1) {
			release_heap_impl();
			state_.template emplace<0>();
			set_size_impl(0);
			return;
		}

		destroy_range_impl(*heap_storage_impl().block, heap_storage_impl().data(), 0, n);
		set_size_impl(0);
	}

	/// @brief Ensures the active heap block is private and then marks it unshareable.
	/// @details This is the policy behind non-const `data()`: once a raw `T*` escapes, future copies
	///          must not share the block because uncontrolled external mutation is now possible.
	///          Shared blocks detach first; the low-level mark step is a unique-owner-only one-shot CAS.
	void privatize_heap_impl() {
		if (auto* block = heap_block_impl()) {
			if (!sso_vector_detail::ownership_header_is_unique(block)) {
				ensure_unique_heap_impl();
				block = heap_block_impl();
			}
			sso_vector_detail::mark_unshareable(block);
		}
	}

	// Modifier primitives.
	/// @brief Appends one element at the end and returns a proxy reference to it.
	template<class... Args>
	reference_proxy emplace_back_impl(Args&&... args) {
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
		return reference(this, n);
	}

	/// @brief Removes the last element when present.
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

	/// @brief Resizes with value-initialized appended elements when growing.
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

	/// @brief Resizes with copies of `value` when growing.
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

	/// @brief Replaces the contents with `count` copies of `value`.
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
		// Build privately, then publish in one step. This intentionally keeps allocator/COW and failure
		// behavior easy to review. Forward-or-better iterators get a single reserve; input iterators
		// still work, but grow incrementally inside the temporary.
		sso_vector rebuilt(alloc_);
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

	/// @brief Attempts to discard excess capacity while preserving contents.
	/// @details For small sizes this may demote heap storage back into the inline buffer.
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

	/// @brief Inserts one element at index `idx`, shifting the tail as required.
	/// @note Middle insertion preserves the existing basic-exception-safety behavior.
	template<class... Args>
	iterator_proxy emplace_at_impl(size_type idx, Args&&... args) {
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
		return iterator(this, idx);
	}

	/// @brief Erases one element at index `idx`, preserving order of the remaining elements.
	iterator_proxy erase_one_impl(size_type idx) {
		const size_type n = size_impl();
		if (idx >= n) return end();
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
		return iterator(this, idx);
	}

	/// @brief Erases a half-open index range `[idx_first, idx_last)`, preserving order.
	iterator_proxy erase_range_impl(size_type idx_first, size_type idx_last) {
		const size_type n = size_impl();
		if (idx_first >= n || idx_first >= idx_last) return iterator(this, idx_first);
		const size_type count = idx_last > n ? (n - idx_first) : (idx_last - idx_first);
		ensure_mutable_heap_impl();
		T* d = data_mut_no_cow_impl();
		if (is_inline_impl()) shift_left_assign_impl(inline_storage_impl(), d, idx_first, n, count);
		else shift_left_assign_impl(*heap_storage_impl().block, d, idx_first, n, count);
		if (is_inline_impl()) destroy_range_impl(inline_storage_impl(), d, n - count, n);
		else destroy_range_impl(*heap_storage_impl().block, d, n - count, n);
		set_size_impl(n - count);
		return iterator(this, idx_first);
	}

	template<typename AppendInserted>
	iterator_proxy rebuild_with_inserted_impl(size_type idx, size_type inserted_count, AppendInserted&& append_inserted) {
		const size_type n = size_impl();
		assert(idx <= n);

		// Rebuild-through-temporary keeps the mutation path allocator-correct and COW-reviewable for the
		// more complex insert forms. The live-prefix invariant is maintained entirely within `rebuilt`
		// until publication.
		sso_vector rebuilt(alloc_);
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
		return iterator(this, idx);
	}

	iterator_proxy insert_fill_impl(size_type idx, size_type count, const T& value) {
		if (count == 0) return iterator(this, idx);
		// This path intentionally rebuilds through a temporary. That keeps allocator/COW publication easy
		// to reason about and avoids layering a second, more complex in-place bulk-insert algorithm on top
		// of the live-prefix helper model.
		return rebuild_with_inserted_impl(idx, count, [&](sso_vector& rebuilt) {
			for (size_type i = 0; i < count; ++i) {
				rebuilt.emplace_back(value);
			}
		});
	}

	template<typename InputIt>
	iterator_proxy insert_range_impl(size_type idx, InputIt first, InputIt last) {
		using category = typename std::iterator_traits<InputIt>::iterator_category;
		if constexpr (std::derived_from<category, std::forward_iterator_tag>) {
			const size_type count = static_cast<size_type>(std::distance(first, last));
			return rebuild_with_inserted_impl(idx, count, [&](sso_vector& rebuilt) {
				for (; first != last; ++first) {
					rebuilt.emplace_back(*first);
				}
			});
		} else {
			std::vector<T, Allocator> buffered(alloc_);
			for (; first != last; ++first) {
				buffered.push_back(*first);
			}
			return rebuild_with_inserted_impl(idx, static_cast<size_type>(buffered.size()), [&](sso_vector& rebuilt) {
				for (auto& value : buffered) {
					rebuilt.emplace_back(std::move(value));
				}
			});
		}
	}

public:
	// ------------------------ proxy reference ------------------------
	/**
	 * @brief Proxy reference used for non-const element access.
	 *
	 * Behaviors:
	 *  - Reading does NOT detach.
	 *  - Writing detaches if shared (refcount > 1), then writes.
	 *  - Does NOT clear SHAREABLE; only data() does that.
	 *
	 * This intentional split lets `v[i]` behave like a mutable element operation without immediately
	 * leaking a raw `T*`. The proxy is therefore closer to `vector<bool>::reference` than to `T&`.
	 */
	class reference_proxy {
	public:
		reference_proxy() = delete;

		reference_proxy(sso_vector* owner, size_type index) noexcept
			: owner_(owner), index_(index) {}

		// Read access (no detach)
		/// @brief Reads the referenced element by value without detaching shared storage.
		operator T() const { return owner_->cref_at_unchecked(index_); }

		// Write access detaches shared heap storage first, but does not clear shareability on its own.
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
			return (*this = tmp);
		}

	private:
		sso_vector* owner_;
		size_type index_;
	};

	// ------------------------ proxy iterator ------------------------
	/**
	 * @brief Random-access iterator whose dereference yields reference_proxy.
	 *
	 * This matches the "proxy iterator" requirement: non-const iteration supports
	 * element assignment via proxy without handing out raw non-const T*.
	 * It is intentionally not a contiguous iterator.
	 */
	class iterator_proxy {
	public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using reference = reference_proxy;
		using pointer = void;

		iterator_proxy() = default;
		iterator_proxy(sso_vector* owner, size_type index) noexcept : owner_(owner), index_(index) {}
		/// @brief Returns the current logical element index within the owning vector.
		size_type index() const noexcept { return index_; }

		reference operator*() const noexcept { return reference(owner_, index_); }
		reference operator[](difference_type n) const noexcept {
			return reference(owner_, add_offset_impl(index_, n));
		}

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
			return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
		}

		friend bool operator==(const iterator_proxy& a, const iterator_proxy& b) noexcept {
			return a.owner_ == b.owner_ && a.index_ == b.index_;
		}
		friend bool operator!=(const iterator_proxy& a, const iterator_proxy& b) noexcept { return !(a == b); }
		friend bool operator<(const iterator_proxy& a, const iterator_proxy& b) noexcept { return a.index_ < b.index_; }
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

		sso_vector* owner_ = nullptr;
		size_type index_ = 0;
	};

	// ------------------------ public types ------------------------
	using reference = reference_proxy;

	// const iterators are raw pointers (like std::vector)
	using const_iterator = const T*;
	// non-const iterators are proxy iterators: random-access but intentionally not contiguous
	using iterator = iterator_proxy;

	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	// ------------------------ ctors/dtor ------------------------

	/// @brief Constructs an empty vector using a default-constructed allocator.
	sso_vector() noexcept(std::is_nothrow_default_constructible_v<Allocator> &&
	                      std::is_nothrow_copy_constructible_v<Allocator>)
		: sso_vector(Allocator()) {}

	/// @brief Constructs an empty vector with the supplied allocator.
	explicit sso_vector(const Allocator& alloc) noexcept(std::is_nothrow_copy_constructible_v<Allocator>)
		: alloc_(alloc), state_(std::in_place_index<0>) {
		set_size_impl(0);
	}

	/// @brief Constructs `count` value-initialized elements.
	sso_vector(size_type count, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		resize(count);
	}

	/// @brief Constructs `count` copies of `value`.
	sso_vector(size_type count, const T& value, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		assign(count, value);
	}

	/// @brief Range constructor for non-integral iterator types.
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	sso_vector(InputIt first, InputIt last, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		assign(first, last);
	}

	/// @brief Initializer-list constructor.
	sso_vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
		: sso_vector(init.begin(), init.end(), alloc) {}

	/// @brief Copy constructor.
	/// @details Heap storage is shared only when the source block is still marked shareable.
	sso_vector(const sso_vector& other)
		: alloc_(std::allocator_traits<Allocator>::select_on_container_copy_construction(other.alloc_))
		, state_(std::in_place_index<0>) {
		copy_from_impl(other);
	}

	/// @brief Move constructor.
	sso_vector(sso_vector&& other)
		: alloc_(std::move(other.alloc_)), state_(std::in_place_index<0>) {
		set_size_impl(0);
		move_from_impl(std::move(other));
	}

	/// @brief Destroys all elements and releases heap ownership.
	~sso_vector() {
		reset_storage_impl();
	}

	// ------------------------ assignment ------------------------

	/// @brief Copy assignment with allocator propagation behavior matching allocator traits.
	sso_vector& operator=(const sso_vector& other) {
		if (this == &other) return *this;

		Allocator target_alloc = alloc_;
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			target_alloc = other.alloc_;
		}

		sso_vector temp(target_alloc);
		temp.copy_from_impl(other);

		reset_storage_impl();
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			alloc_ = other.alloc_;
		}
		move_from_impl(std::move(temp));
		return *this;
	}

	/// @brief Move assignment with allocator propagation behavior matching allocator traits.
	sso_vector& operator=(sso_vector&& other) {
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

		sso_vector temp(alloc_);
		temp.move_rebuild_from_impl(std::move(other));
		reset_storage_impl();
		move_from_impl(std::move(temp));
		return *this;
	}

	// ------------------------ element access ------------------------

	const_reference at(size_type pos) const {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return data_const_impl()[pos];
	}
	reference at(size_type pos) {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return (*this)[pos];
	}

	/// @brief Proxy-based subscript for mutable access.
	/// @details Reading through the proxy does not detach. Writing through it detaches shared heap storage.
	///          This differs intentionally from `std::vector<T>::operator[]`, whose mutable overload
	///          returns `T&` directly.
	reference operator[](size_type pos) noexcept {
		assert(pos < size() && "sso_vector::operator[] should be in range");
		return reference(this, pos);
	}
	const_reference operator[](size_type pos) const noexcept {
		assert(pos < size() && "sso_vector::operator[] should be in range");
		return cref_at_unchecked(pos);
	}

	const_reference front() const noexcept { return data_const_impl()[0]; }
	reference front() noexcept { return (*this)[0]; }

	const_reference back() const noexcept { return data_const_impl()[size() - 1]; }
	reference back() noexcept { return (*this)[size() - 1]; }

	const_pointer data() const noexcept { return data_const_impl(); }

	/// @brief Returns mutable contiguous storage.
	/// @warning This explicitly privatizes heap storage before handing out `T*`.
	///          Unlike proxy-based element mutation, raw mutable access also disables future sharing
	///          for the current heap block because external writes can no longer be tracked.
	pointer data() {
		privatize_heap_impl();
		return data_mut_no_cow_impl();
	}

	// ------------------------ iterators ------------------------

	iterator begin() noexcept { return iterator(this, 0); }
	iterator end() noexcept { return iterator(this, size()); }

	const_iterator begin() const noexcept { return data_const_impl(); }
	const_iterator end() const noexcept { return data_const_impl() + size(); }
	const_iterator cbegin() const noexcept { return begin(); }
	const_iterator cend() const noexcept { return end(); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	// ------------------------ capacity ------------------------

	bool empty() const noexcept { return size() == 0; }
	size_type size() const noexcept { return size_impl(); }
	/// @brief Returns true when the active representation is the inline/SSO buffer.
	bool is_inline() const noexcept { return is_inline_impl(); }
	/// @brief Returns true when the current heap payload has more than one owner.
	/// @note This is an instantaneous ownership snapshot, not a synchronization guarantee.
	bool is_shared() const noexcept { return share_count() > 1; }
	/// @brief Returns true when future sharing is currently permitted for the active state.
	/// @details Inline storage always reports shareable. Heap-backed state reflects the ownership header.
	bool is_shareable() const noexcept {
		if (!is_heap_impl()) return true;
		return sso_vector_detail::ownership_header_is_shareable(
			sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block));
	}
	/// @brief Returns the current effective owner count.
	/// @details Inline storage reports `1`. Heap-backed storage reports the current ownership-header count.
	size_type share_count() const noexcept {
		if (!is_heap_impl()) return 1;
		return static_cast<size_type>(sso_vector_detail::ownership_header_share_count(
			sso_vector_detail::load_ownership_header_snapshot(heap_storage_impl().block)));
	}

	size_type capacity() const noexcept { return capacity_impl(); }

	/// @brief Ensures capacity for at least `new_cap` elements.
	/// @details Reserving storage changes capacity/representation only; it must not construct elements.
	void reserve(size_type new_cap) {
		if (new_cap <= capacity()) return;
		ensure_capacity_impl(new_cap);
	}

	/// @brief Requests capacity reduction while preserving contents.
	void shrink_to_fit() {
		shrink_to_fit_impl();
	}

	// ------------------------ modifiers ------------------------

	/// @brief Destroys all elements while retaining the representation shell.
	void clear() noexcept { clear_impl(); }

	/// @brief Appends one copy of `value`.
	void push_back(const T& value) { emplace_back(value); }
	/// @brief Appends one moved-from `value`.
	void push_back(T&& value) { emplace_back(std::move(value)); }

	/// @brief Appends one element constructed in place.
	template<class... Args>
	reference emplace_back(Args&&... args) {
		return emplace_back_impl(std::forward<Args>(args)...);
	}

	/// @brief Removes the final element if present.
	void pop_back() { pop_back_impl(); }

	/// @brief Resizes using value-initialization when growing.
	void resize(size_type count) { resize_impl(count); }

	/// @brief Resizes using copies of `value` when growing.
	void resize(size_type count, const T& value) { resize_impl(count, value); }

	/// @brief Replaces the contents with `count` copies of `value`.
	void assign(size_type count, const T& value) { assign_fill_impl(count, value); }

	/// @brief Replaces the contents from an input range.
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	void assign(InputIt first, InputIt last) {
		assign_range_impl(first, last);
	}

	void assign(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

	/// @brief Insert by emplacement at position @p pos.
	/// @note For middle insertion this implementation provides the basic exception guarantee.
	/// If element move/assignment throws during shifting, the container remains valid but the
	/// exact prior element ordering is not guaranteed to be restored.
	template<class... Args>
	iterator emplace(iterator pos, Args&&... args) {
		return emplace_at_impl(pos.index(), std::forward<Args>(args)...);
	}

	iterator insert(iterator pos, const T& value) { return emplace(pos, value); }
	iterator insert(iterator pos, T&& value) { return emplace(pos, std::move(value)); }
	iterator insert(iterator pos, size_type count, const T& value) {
		return insert_fill_impl(pos.index(), count, value);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(iterator pos, InputIt first, InputIt last) {
		return insert_range_impl(pos.index(), first, last);
	}
	iterator insert(iterator pos, std::initializer_list<T> ilist) {
		return insert(pos, ilist.begin(), ilist.end());
	}

	iterator insert(const_iterator pos, const T& value) {
		return insert(iterator(this, static_cast<size_type>(pos - cbegin())), value);
	}
	iterator insert(const_iterator pos, T&& value) {
		return insert(iterator(this, static_cast<size_type>(pos - cbegin())), std::move(value));
	}
	iterator insert(const_iterator pos, size_type count, const T& value) {
		return insert(iterator(this, static_cast<size_type>(pos - cbegin())), count, value);
	}
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(const_iterator pos, InputIt first, InputIt last) {
		return insert(iterator(this, static_cast<size_type>(pos - cbegin())), first, last);
	}
	iterator insert(const_iterator pos, std::initializer_list<T> ilist) {
		return insert(iterator(this, static_cast<size_type>(pos - cbegin())), ilist);
	}

	iterator erase(iterator pos) {
		return erase_one_impl(pos.index());
	}
	iterator erase(iterator first, iterator last) {
		return erase_range_impl(first.index(), last.index());
	}
	iterator erase(const_iterator pos) {
		return erase(iterator(this, static_cast<size_type>(pos - cbegin())));
	}
	iterator erase(const_iterator first, const_iterator last) {
		return erase(iterator(this, static_cast<size_type>(first - cbegin())),
		             iterator(this, static_cast<size_type>(last - cbegin())));
	}

	void swap(sso_vector& other) {
		if (this == &other) return;
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value) {
			std::swap(alloc_, other.alloc_);
		} else if (!can_adopt_heap_storage_from_impl(other) && (is_heap_impl() || other.is_heap_impl())) {
			// When allocators do not propagate and heap ownership cannot be exchanged directly, simulate
			// swap through allocator-aware move assignment so each container keeps allocations compatible
			// with its own allocator.
			sso_vector temp(alloc_);
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
			sso_vector tmp(std::move(other));
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
	// Allocator used for explicit element construction/destruction and heap-block byte allocation.
	Allocator alloc_{};
	// Active representation plus sideband logical size. The sideband must always match the number of
	// live T objects in the active storage alternative.
	variant_t state_{std::in_place_index<0>};
};

// Default-N convenience alias
template<typename T, typename Allocator = std::allocator<T>>
using sso_vector_default =
	sso_vector<T, sso_vector_detail::default_inline_elems<T, Allocator>(), Allocator>;

// comparisons
template<typename T, std::size_t N, typename Allocator>
inline bool operator==(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
template<typename T, std::size_t N, typename Allocator>
inline bool operator!=(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return !(a == b);
}
template<typename T, std::size_t N, typename Allocator>
inline bool operator<(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}
template<typename T, std::size_t N, typename Allocator>
inline bool operator>(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return b < a;
}
template<typename T, std::size_t N, typename Allocator>
inline bool operator<=(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return !(b < a);
}
template<typename T, std::size_t N, typename Allocator>
inline bool operator>=(const sso_vector<T, N, Allocator>& a, const sso_vector<T, N, Allocator>& b) {
	return !(a < b);
}

}}} // namespace sw::universal::internal
