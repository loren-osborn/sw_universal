#pragma once
// sso_vector.hpp
//
// A std::vector-like container with:
//  - Small-size optimization (inline storage for N elements)
//  - Copy-on-write heap storage (shared buffer with atomic control word)
//  - Proxy element + proxy non-const iterators (vector<bool>-style)
//  - "Shareable" state to prevent sharing when an external non-const pointer has been handed out.
//
// Key design points:
//  - Two-layout representation uses custom_indexed_variant with sideband size.
//  - Heap sharing state uses bitfield_pack over a single atomic control word.
//    Layout: [SHAREABLE:1][REFCOUNT:remainder] (no SPARE, no arbitrary bit widths).
//  - Heap capacity is stored in the heap block header.
//  - Heap control word is mutable so refcount/shareable can change through const references.
//  - N==0 degrades to std::vector.
//  - Default N computed to match sizeof(std::vector<T, Allocator>) where possible.
//
// Threading model (matches std::vector intent):
//  - Like std::vector, element operations are not thread-safe against concurrent mutation.
//  - Atomic control word is for sharing bookkeeping only.
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

#include "universal/internal/custom_indexed_variant/custom_indexed_variant.hpp"
#include "universal/internal/bitvector/bitfield_pack.hpp"

namespace sw { namespace universal {
namespace internal {

namespace sso_vector_detail {

// ------------------------ helpers ------------------------

inline constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
	return (a + b - 1) / b;
}

template<class T, class Allocator>
inline constexpr std::size_t default_inline_bytes() noexcept {
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

// ------------------------ heap control word ------------------------

/// @brief Atomic header word type used for COW sharing state.
/// Bit layout (LSB-first):
/// - field 0: shareable flag, width = 1
/// - field 1: refcount, width = remainder (all higher bits)
using header_word = std::uint64_t;

using header_bits = bitfield_pack<
	header_word,
	bitfield_field_spec<1>,
	bitfield_remainder
>;

enum header_field : std::size_t {
	SHAREABLE = 0,
	REFCOUNT = 1,
};

static_assert(header_bits::template field_width<REFCOUNT>() > 0, "sso_vector: header refcount remainder must be non-zero width");

inline constexpr header_word hdr_pack(bool shareable, std::uint64_t rc) noexcept {
	header_bits b{};
	b.set_raw_storage(0);
	b.template set<SHAREABLE>(shareable ? 1u : 0u);
	b.template set<REFCOUNT>(static_cast<header_word>(rc));
	return b.raw_storage();
}

inline constexpr bool hdr_shareable(header_word w) noexcept {
	header_bits b{};
	b.set_raw_storage(w);
	return b.template get<SHAREABLE>() != 0;
}

inline constexpr std::uint64_t hdr_refcount(header_word w) noexcept {
	header_bits b{};
	b.set_raw_storage(w);
	return static_cast<std::uint64_t>(b.template get<REFCOUNT>());
}

inline constexpr header_word hdr_set_shareable(header_word w, bool shareable) noexcept {
	header_bits b{};
	b.set_raw_storage(w);
	b.template set<SHAREABLE>(shareable ? 1u : 0u);
	return b.raw_storage();
}

inline constexpr header_word hdr_set_refcount(header_word w, std::uint64_t rc) noexcept {
	header_bits b{};
	b.set_raw_storage(w);
	b.template set<REFCOUNT>(static_cast<header_word>(rc));
	return b.raw_storage();
}

// ------------------------ heap block ------------------------

template<class T>
struct heap_block {
	static_assert(alignof(T) <= alignof(std::max_align_t), "sso_vector requires T alignment compatible with byte-rebound allocator");
	mutable std::atomic<header_word> hdr_atomic;
	std::size_t capacity;
	alignas(T) std::byte data[sizeof(T)];
};

template<class T>
inline constexpr std::size_t heap_block_align() noexcept {
	return (std::max)(alignof(heap_block<T>), alignof(T));
}

template<class T>
inline T* block_data(heap_block<T>* b) noexcept {
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
	auto* b = ::new (mem) heap_block<T>{
		std::atomic<header_word>(hdr_pack(true, 1)),
		capacity,
		{std::byte{0}}
	};
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
inline header_word load_hdr(const heap_block<T>* b, std::memory_order order = std::memory_order_acquire) noexcept {
	return b->hdr_atomic.load(order);
}

template<class T>
inline void store_hdr(heap_block<T>* b, header_word v, std::memory_order order = std::memory_order_release) noexcept {
	b->hdr_atomic.store(v, order);
}

template<class T>
inline bool try_inc_ref_if_shareable(const heap_block<T>* b) noexcept {
	auto& a = b->hdr_atomic;
	header_word cur = a.load(std::memory_order_acquire);
	for (;;) {
		if (!hdr_shareable(cur)) return false;
		const std::uint64_t rc = hdr_refcount(cur);
		assert(rc >= 1);
		if (rc == std::numeric_limits<std::uint64_t>::max()) return false;
		header_word next = hdr_set_refcount(cur, rc + 1);
		if (a.compare_exchange_weak(cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
			return true;
		}
	}
}

template<class T>
inline bool dec_ref(const heap_block<T>* b) noexcept {
	auto& a = b->hdr_atomic;
	header_word cur = a.load(std::memory_order_acquire);
	for (;;) {
		const std::uint64_t rc = hdr_refcount(cur);
		assert(rc >= 1);
		const std::uint64_t next_rc = rc - 1;
		header_word next = hdr_set_refcount(cur, next_rc);
		if (a.compare_exchange_weak(cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
			return next_rc == 0;
		}
	}
}

template<class T>
inline void clear_shareable(heap_block<T>* b) noexcept {
	auto& a = b->hdr_atomic;
	header_word cur = a.load(std::memory_order_acquire);
	for (;;) {
		if (!hdr_shareable(cur)) return;
		header_word next = hdr_set_shareable(cur, false);
		if (a.compare_exchange_weak(cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) return;
	}
}

} // namespace sso_vector_detail

template<typename T, std::size_t N, typename Allocator = std::allocator<T>>
class sso_vector;

// N==0 specialization: degrade to std::vector.
template<typename T, typename Allocator>
class sso_vector<T, 0, Allocator> : public std::vector<T, Allocator> {
	using base = std::vector<T, Allocator>;
public:
	using base::base;
};

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

private:
	struct inline_storage {
		alignas(T) std::byte buf[sizeof(T) * N]{};
		T* data() noexcept { return std::launder(reinterpret_cast<T*>(buf)); }
		const T* data() const noexcept { return std::launder(reinterpret_cast<const T*>(buf)); }
	};

	struct heap_storage {
		sso_vector_detail::heap_block<T>* block = nullptr;
		T* data() noexcept { return sso_vector_detail::block_data(block); }
		const T* data() const noexcept { return sso_vector_detail::block_data(block); }
		size_type capacity() const noexcept { return block ? block->capacity : 0; }
	};

	using variant_t = custom_indexed_variant<index_encoded_with_sideband_data, inline_storage, heap_storage>;

	static size_type sideband_to_size(std::size_t v) noexcept { return static_cast<size_type>(v); }
	static std::size_t size_to_sideband(size_type v) noexcept { return static_cast<std::size_t>(v); }

	size_type size_unsafe() const noexcept {
		return sideband_to_size(static_cast<std::size_t>(state_.sideband().val()));
	}
	void set_size_unsafe(size_type n) noexcept {
		state_.sideband().set_val(size_to_sideband(n));
	}

	bool is_inline() const noexcept { return state_.index() == 0; }
	bool is_heap() const noexcept { return state_.index() == 1; }

	inline_storage& inl() noexcept { return state_.template get<0>(); }
	const inline_storage& inl() const noexcept { return state_.template get<0>(); }
	heap_storage& hep() noexcept { return state_.template get<1>(); }
	const heap_storage& hep() const noexcept { return state_.template get<1>(); }

	const T* data_const() const noexcept {
		return is_inline() ? inl().data() : hep().data();
	}
	T* data_mut_no_cow() noexcept {
		return is_inline() ? inl().data() : hep().data();
	}

	void destroy_range(T* first, T* last) noexcept {
		for (; first != last; ++first) {
			std::allocator_traits<Allocator>::destroy(alloc_, first);
		}
	}

	void release_heap_block(sso_vector_detail::heap_block<T>* b, size_type constructed) noexcept {
		if (!b) return;
		if (sso_vector_detail::dec_ref(b)) {
			T* d = sso_vector_detail::block_data(b);
			for (size_type i = 0; i < constructed; ++i) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + i);
			}
			sso_vector_detail::deallocate_block(b, alloc_);
		}
	}

	void release_heap() noexcept {
		if (!is_heap()) return;
		auto& h = hep();
		if (!h.block) return;
		const size_type n = size_unsafe();
		release_heap_block(h.block, n);
		h.block = nullptr;
	}

	// Detach if shared (refcount>1). Shareable only controls share-on-copy; it does not prevent detach.
	void ensure_unique_heap() {
		if (!is_heap()) return;
		auto& h = hep();
		if (!h.block) return;

		const auto cur = sso_vector_detail::load_hdr(h.block);
		const std::uint64_t rc = sso_vector_detail::hdr_refcount(cur);
		assert(rc >= 1);
		if (rc == 1) return;

		const size_type n = size_unsafe();
		auto* old_block = h.block;
		const size_type cap = old_block->capacity;

		auto* new_block = sso_vector_detail::allocate_block<T>(cap, alloc_);
		T* dst = sso_vector_detail::block_data(new_block);
		const T* src = sso_vector_detail::block_data(old_block);

		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				if constexpr (std::is_copy_constructible_v<T>) {
					std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, src[constructed]);
				} else {
					throw std::logic_error("sso_vector: detaching shared storage requires copy-constructible value_type");
				}
			}
		} catch (...) {
			destroy_range(dst, dst + constructed);
			sso_vector_detail::deallocate_block(new_block, alloc_);
			throw;
		}

		h.block = new_block;
		release_heap_block(old_block, n);
	}

	void promote_inline_to_heap(size_type new_cap) {
		assert(is_inline());
		if (new_cap < 1) new_cap = 1;

		auto* b = sso_vector_detail::allocate_block<T>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data(b);
		T* src = inl().data();
		const size_type n = size_unsafe();

		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				std::allocator_traits<Allocator>::construct(
					alloc_, dst + constructed, std::move_if_noexcept(src[constructed]));
			}
		} catch (...) {
			destroy_range(dst, dst + constructed);
			sso_vector_detail::deallocate_block(b, alloc_);
			throw;
		}

		destroy_range(src, src + n);

		heap_storage h{};
		h.block = b;
		state_.template emplace<1>(h);
	}

	void reallocate_heap(size_type new_cap) {
		assert(is_heap());
		auto& h = hep();
		assert(h.block);

		ensure_unique_heap();
		auto* old_block = h.block;
		const size_type n = size_unsafe();
		assert(new_cap >= n);

		auto* new_block = sso_vector_detail::allocate_block<T>(new_cap, alloc_);
		T* dst = sso_vector_detail::block_data(new_block);
		T* src = sso_vector_detail::block_data(old_block);

		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				std::allocator_traits<Allocator>::construct(
					alloc_, dst + constructed, std::move_if_noexcept(src[constructed]));
			}
		} catch (...) {
			destroy_range(dst, dst + constructed);
			sso_vector_detail::deallocate_block(new_block, alloc_);
			throw;
		}

		destroy_range(src, src + n);
		sso_vector_detail::deallocate_block(old_block, alloc_);

		h.block = new_block;
	}

	static size_type growth_capacity(size_type desired, size_type current) noexcept {
		const size_type doubled = current ? current * 2 : 1;
		return (std::max)(desired, doubled);
	}

	void ensure_capacity(size_type desired) {
		const size_type cap = capacity();
		if (desired <= cap) return;

		if (desired <= N && is_inline()) {
			return;
		}

		if (is_inline()) {
			promote_inline_to_heap(growth_capacity(desired, N));
			return;
		}

		reallocate_heap(growth_capacity(desired, cap));
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
	 */
	class reference_proxy {
	public:
		reference_proxy() = delete;

		reference_proxy(sso_vector* owner, size_type index) noexcept
			: owner_(owner), index_(index) {}

		// Read access (no detach)
		operator T() const { return owner_->data_const()[index_]; }

		// Write access (detach if needed, then write)
		reference_proxy& operator=(const T& v) {
			if (owner_->is_heap()) owner_->ensure_unique_heap();
			owner_->data_mut_no_cow()[index_] = v;
			return *this;
		}
		reference_proxy& operator=(T&& v) {
			if (owner_->is_heap()) owner_->ensure_unique_heap();
			owner_->data_mut_no_cow()[index_] = std::move(v);
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
		size_type index() const noexcept { return index_; }

		reference operator*() const noexcept { return reference(owner_, index_); }
		reference operator[](difference_type n) const noexcept {
			return reference(owner_, index_ + static_cast<size_type>(n));
		}

		iterator_proxy& operator++() noexcept { ++index_; return *this; }
		iterator_proxy operator++(int) noexcept { auto tmp = *this; ++(*this); return tmp; }
		iterator_proxy& operator--() noexcept { --index_; return *this; }
		iterator_proxy operator--(int) noexcept { auto tmp = *this; --(*this); return tmp; }

		iterator_proxy& operator+=(difference_type n) noexcept { index_ += static_cast<size_type>(n); return *this; }
		iterator_proxy& operator-=(difference_type n) noexcept { index_ -= static_cast<size_type>(n); return *this; }

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
		sso_vector* owner_ = nullptr;
		size_type index_ = 0;
	};

	// ------------------------ public types ------------------------
	using reference = reference_proxy;

	// const iterators are raw pointers (like std::vector)
	using const_iterator = const T*;
	// non-const iterators are proxy iterators
	using iterator = iterator_proxy;

	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	// ------------------------ ctors/dtor ------------------------

	sso_vector() noexcept(noexcept(Allocator()))
		: sso_vector(Allocator()) {}

	explicit sso_vector(const Allocator& alloc) noexcept
		: alloc_(alloc), state_(std::in_place_index<0>) {
		set_size_unsafe(0);
	}

	sso_vector(size_type count, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		resize(count);
	}

	sso_vector(size_type count, const T& value, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		assign(count, value);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	sso_vector(InputIt first, InputIt last, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		assign(first, last);
	}

	sso_vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
		: sso_vector(init.begin(), init.end(), alloc) {}

	// Copy: share heap iff shareable==1; else deep clone.
	sso_vector(const sso_vector& other)
		: alloc_(std::allocator_traits<Allocator>::select_on_container_copy_construction(other.alloc_))
		, state_(std::in_place_index<0>) {
		set_size_unsafe(other.size());
		if (other.is_inline()) {
			state_.template emplace<0>();
			const size_type n = other.size();
			T* dst = inl().data();
			const T* src = other.inl().data();
			size_type constructed = 0;
			try {
				for (; constructed < n; ++constructed) {
					std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, src[constructed]);
				}
			} catch (...) {
				destroy_range(dst, dst + constructed);
				set_size_unsafe(0);
				throw;
			}
			return;
		}

		state_.template emplace<1>();
		auto* b = other.hep().block;
		assert(b);

		if (sso_vector_detail::try_inc_ref_if_shareable(b)) {
			hep().block = b;
			return;
		}

		const size_type n = other.size();
		const size_type cap = b->capacity;

		auto* nb = sso_vector_detail::allocate_block<T>(cap, alloc_);
		T* dst = sso_vector_detail::block_data(nb);
		const T* src = sso_vector_detail::block_data(b);

		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, src[constructed]);
			}
		} catch (...) {
			destroy_range(dst, dst + constructed);
			sso_vector_detail::deallocate_block(nb, alloc_);
			throw;
		}
		hep().block = nb;
	}

	sso_vector(sso_vector&& other) noexcept
		: alloc_(std::move(other.alloc_)), state_(std::in_place_index<0>) {
		set_size_unsafe(0);
		swap(other);
	}

	~sso_vector() {
		clear();
		release_heap();
	}

	// ------------------------ assignment ------------------------

	sso_vector& operator=(const sso_vector& other) {
		if (this == &other) return *this;

		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			if (alloc_ != other.alloc_) {
				clear();
				release_heap();
				alloc_ = other.alloc_;
			}
		}

		clear();
		release_heap();

		set_size_unsafe(other.size());
		if (other.is_inline()) {
			state_.template emplace<0>();
			const size_type n = other.size();
			T* dst = inl().data();
			const T* src = other.inl().data();
			size_type constructed = 0;
			try {
				for (; constructed < n; ++constructed) {
					std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, src[constructed]);
				}
			} catch (...) {
				destroy_range(dst, dst + constructed);
				set_size_unsafe(0);
				throw;
			}
			return *this;
		}

		state_.template emplace<1>();
		auto* b = other.hep().block;
		assert(b);

		if (sso_vector_detail::try_inc_ref_if_shareable(b)) {
			hep().block = b;
			return *this;
		}

		const size_type n = other.size();
		const size_type cap = b->capacity;

		auto* nb = sso_vector_detail::allocate_block<T>(cap, alloc_);
		T* dst = sso_vector_detail::block_data(nb);
		const T* src = sso_vector_detail::block_data(b);

		size_type constructed = 0;
		try {
			for (; constructed < n; ++constructed) {
				std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, src[constructed]);
			}
		} catch (...) {
			destroy_range(dst, dst + constructed);
			sso_vector_detail::deallocate_block(nb, alloc_);
			throw;
		}
		hep().block = nb;
		return *this;
	}

	sso_vector& operator=(sso_vector&& other) noexcept(std::allocator_traits<Allocator>::is_always_equal::value) {
		if (this == &other) return *this;

		clear();
		release_heap();
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
			alloc_ = std::move(other.alloc_);
		}
		swap(other);
		return *this;
	}

	// ------------------------ element access ------------------------

	const_reference at(size_type pos) const {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return data_const()[pos];
	}
	reference at(size_type pos) {
		if (pos >= size()) throw std::out_of_range("sso_vector::at out of range");
		return (*this)[pos];
	}

	// Non-const operator[] returns proxy (no detach on read; detach on write).
	reference operator[](size_type pos) noexcept { return reference(this, pos); }
	const_reference operator[](size_type pos) const noexcept { return data_const()[pos]; }

	const_reference front() const noexcept { return data_const()[0]; }
	reference front() noexcept { return (*this)[0]; }

	const_reference back() const noexcept { return data_const()[size() - 1]; }
	reference back() noexcept { return (*this)[size() - 1]; }

	const_pointer data() const noexcept { return data_const(); }

	// Non-const data(): hands out raw mutable pointer -> clear SHAREABLE, then ensure unique.
	pointer data() {
		if (is_heap() && hep().block) {
			sso_vector_detail::clear_shareable(hep().block);
			ensure_unique_heap();
		}
		return data_mut_no_cow();
	}

	// ------------------------ iterators ------------------------

	iterator begin() noexcept { return iterator(this, 0); }
	iterator end() noexcept { return iterator(this, size()); }

	const_iterator begin() const noexcept { return data_const(); }
	const_iterator end() const noexcept { return data_const() + size(); }
	const_iterator cbegin() const noexcept { return begin(); }
	const_iterator cend() const noexcept { return end(); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	// ------------------------ capacity ------------------------

	bool empty() const noexcept { return size() == 0; }
	size_type size() const noexcept { return size_unsafe(); }

	size_type capacity() const noexcept { return is_inline() ? N : hep().capacity(); }

	void reserve(size_type new_cap) {
		if (new_cap <= capacity()) return;
		ensure_capacity(new_cap);
	}

	void shrink_to_fit() {
		const size_type n = size();
		if (n == 0) {
			clear();
			release_heap();
			state_.template emplace<0>();
			set_size_unsafe(0);
			return;
		}
		if (is_inline()) return;

		if (n <= N) {
			ensure_unique_heap();
			inline_storage ni{};
			T* dst = ni.data();
			T* src = hep().data();

			size_type constructed = 0;
			try {
				for (; constructed < n; ++constructed) {
					std::allocator_traits<Allocator>::construct(alloc_, dst + constructed, std::move_if_noexcept(src[constructed]));
				}
			} catch (...) {
				destroy_range(dst, dst + constructed);
				throw;
			}

			clear();
			release_heap();
			state_.template emplace<0>(ni);
			set_size_unsafe(n);
			return;
		}

		reallocate_heap(n);
	}

	// ------------------------ modifiers ------------------------

	void clear() noexcept {
		const size_type n = size();
		if (n == 0) { set_size_unsafe(0); return; }

		if (is_inline()) {
			destroy_range(inl().data(), inl().data() + n);
			set_size_unsafe(0);
			return;
		}

		ensure_unique_heap();
		destroy_range(hep().data(), hep().data() + n);
		set_size_unsafe(0);
	}

	void push_back(const T& value) { emplace_back(value); }
	void push_back(T&& value) { emplace_back(std::move(value)); }

	template<class... Args>
	reference emplace_back(Args&&... args) {
		const size_type n = size();
		ensure_capacity(n + 1);
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();
		std::allocator_traits<Allocator>::construct(alloc_, d + n, std::forward<Args>(args)...);
		set_size_unsafe(n + 1);
		return (*this)[n];
	}

	void pop_back() {
		const size_type n = size();
		if (n == 0) return;
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();
		std::allocator_traits<Allocator>::destroy(alloc_, d + (n - 1));
		set_size_unsafe(n - 1);
	}

	void resize(size_type count) {
		const size_type n = size();
		if (count == n) return;

		if (count < n) {
			if (is_heap()) ensure_unique_heap();
			T* d = data_mut_no_cow();
			for (size_type i = count; i < n; ++i) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + i);
			}
			set_size_unsafe(count);
			return;
		}

		ensure_capacity(count);
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();

		size_type constructed = 0;
		try {
			for (size_type i = n; i < count; ++i, ++constructed) {
				std::allocator_traits<Allocator>::construct(alloc_, d + i);
			}
		} catch (...) {
			for (size_type i = 0; i < constructed; ++i) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + (n + i));
			}
			throw;
		}
		set_size_unsafe(count);
	}

	void resize(size_type count, const T& value) {
		const size_type n = size();
		if (count == n) return;

		if (count < n) {
			if (is_heap()) ensure_unique_heap();
			T* d = data_mut_no_cow();
			for (size_type i = count; i < n; ++i) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + i);
			}
			set_size_unsafe(count);
			return;
		}

		ensure_capacity(count);
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();

		size_type constructed = 0;
		try {
			for (size_type i = n; i < count; ++i, ++constructed) {
				std::allocator_traits<Allocator>::construct(alloc_, d + i, value);
			}
		} catch (...) {
			for (size_type i = 0; i < constructed; ++i) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + (n + i));
			}
			throw;
		}
		set_size_unsafe(count);
	}

	void assign(size_type count, const T& value) {
		clear();
		ensure_capacity(count);
		if (count == 0) return;
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();

		size_type constructed = 0;
		try {
			for (; constructed < count; ++constructed) {
				std::allocator_traits<Allocator>::construct(alloc_, d + constructed, value);
			}
		} catch (...) {
			destroy_range(d, d + constructed);
			throw;
		}
		set_size_unsafe(count);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	void assign(InputIt first, InputIt last) {
		clear();
		for (; first != last; ++first) push_back(*first);
	}

	void assign(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

	/// @brief Insert by emplacement at position @p pos.
	/// @note For middle insertion this implementation provides the basic exception guarantee.
	/// If element move/assignment throws during shifting, the container remains valid but the
	/// exact prior element ordering is not guaranteed to be restored.
	template<class... Args>
	iterator emplace(iterator pos, Args&&... args) {
		const size_type idx = pos.index();
		const size_type n = size();
		assert(idx <= n);
		ensure_capacity(n + 1);
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();
		if (idx == n) {
			std::allocator_traits<Allocator>::construct(alloc_, d + n, std::forward<Args>(args)...);
		} else {
			std::allocator_traits<Allocator>::construct(alloc_, d + n, std::move_if_noexcept(d[n - 1]));
			try {
				for (size_type i = n - 1; i > idx; --i) {
					d[i] = std::move_if_noexcept(d[i - 1]);
				}
				T tmp(std::forward<Args>(args)...);
				d[idx] = std::move(tmp);
			} catch (...) {
				std::allocator_traits<Allocator>::destroy(alloc_, d + n);
				throw;
			}
		}
		set_size_unsafe(n + 1);
		return iterator(this, idx);
	}

	iterator insert(iterator pos, const T& value) { return emplace(pos, value); }
	iterator insert(iterator pos, T&& value) { return emplace(pos, std::move(value)); }
	iterator insert(iterator pos, size_type count, const T& value) {
		const size_type idx = pos.index();
		for (size_type i = 0; i < count; ++i) {
			(void)insert(iterator(this, idx + i), value);
		}
		return iterator(this, idx);
	}

	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(iterator pos, InputIt first, InputIt last) {
		const size_type idx = pos.index();
		size_type inserted = 0;
		for (; first != last; ++first, ++inserted) {
			(void)insert(iterator(this, idx + inserted), *first);
		}
		return iterator(this, idx);
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
		const size_type idx = pos.index();
		const size_type n = size();
		if (idx >= n) return end();
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();
		for (size_type i = idx; i + 1 < n; ++i) {
			d[i] = std::move_if_noexcept(d[i + 1]);
		}
		std::allocator_traits<Allocator>::destroy(alloc_, d + (n - 1));
		set_size_unsafe(n - 1);
		return iterator(this, idx);
	}
	iterator erase(iterator first, iterator last) {
		const size_type idx_first = first.index();
		const size_type idx_last = last.index();
		const size_type n = size();
		if (idx_first >= n || idx_first >= idx_last) return iterator(this, idx_first);
		const size_type count = idx_last > n ? (n - idx_first) : (idx_last - idx_first);
		if (is_heap()) ensure_unique_heap();
		T* d = data_mut_no_cow();
		for (size_type i = idx_first; i + count < n; ++i) {
			d[i] = std::move_if_noexcept(d[i + count]);
		}
		for (size_type i = n - count; i < n; ++i) {
			std::allocator_traits<Allocator>::destroy(alloc_, d + i);
		}
		set_size_unsafe(n - count);
		return iterator(this, idx_first);
	}
	iterator erase(const_iterator pos) {
		return erase(iterator(this, static_cast<size_type>(pos - cbegin())));
	}
	iterator erase(const_iterator first, const_iterator last) {
		return erase(iterator(this, static_cast<size_type>(first - cbegin())),
		             iterator(this, static_cast<size_type>(last - cbegin())));
	}

	void swap(sso_vector& other) noexcept(std::allocator_traits<Allocator>::is_always_equal::value) {
		if (this == &other) return;
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value) {
			std::swap(alloc_, other.alloc_);
		}
		// TODO(custom_indexed_variant): move this sideband-preservation logic into a shared
		// variant swap primitive once custom_indexed_variant guarantees sideband swap semantics.
		const std::size_t this_size_sideband = state_.sideband().val();
		const std::size_t other_size_sideband = other.state_.sideband().val();
		std::swap(state_, other.state_);
		state_.sideband().set_val(other_size_sideband);
		other.state_.sideband().set_val(this_size_sideband);
	}

	allocator_type get_allocator() const noexcept { return alloc_; }

private:
	Allocator alloc_{};
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
