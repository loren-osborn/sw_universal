#pragma once
// sso_vector.hpp: small-string-optimized vector utility mirroring std::vector (C++20)
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sw { namespace universal {

namespace internal {

/// @brief std::vector-compatible container with a small-buffer optimization.
/// @tparam T Element type.
/// @tparam Allocator Allocator type.
/// @note The inline storage size is chosen based on pointer-sized budget.
template<typename T, typename Allocator = std::allocator<T>>
class sso_vector {
public:
	using value_type = T;
	using allocator_type = Allocator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = value_type&;
	using const_reference = const value_type&;
	using pointer = typename std::allocator_traits<Allocator>::pointer;
	using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
	using iterator = pointer;
	using const_iterator = const_pointer;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	/// @brief Default constructor.
	sso_vector() noexcept(noexcept(Allocator()))
		: sso_vector(Allocator()) {}

	/// @brief Constructs with an allocator.
	explicit sso_vector(const Allocator& alloc) noexcept
		: alloc_(alloc), size_(0), capacity_(sso_capacity), data_(sso_data()) {}

	/// @brief Constructs with count default-inserted elements.
	sso_vector(size_type count, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		if (count == 0) {
			return;
		}
		ensure_capacity(count);
		uninitialized_default_n(data_, count);
		size_ = count;
	}

	/// @brief Constructs with count copies of value.
	sso_vector(size_type count, const T& value, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		if (count == 0) {
			return;
		}
		ensure_capacity(count);
		uninitialized_fill_n(data_, count, value);
		size_ = count;
	}

	/// @brief Constructs from an iterator range.
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	sso_vector(InputIt first, InputIt last, const Allocator& alloc = Allocator())
		: sso_vector(alloc) {
		assign_range(first, last);
	}

	/// @brief Constructs from an initializer list.
	sso_vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
		: sso_vector(init.begin(), init.end(), alloc) {}

	/// @brief Copy constructor.
	sso_vector(const sso_vector& other)
		: sso_vector(std::allocator_traits<Allocator>::select_on_container_copy_construction(other.alloc_)) {
		if (other.size_ == 0) {
			return;
		}
		ensure_capacity(other.size_);
		uninitialized_copy_n(other.data_, other.size_, data_);
		size_ = other.size_;
	}

	/// @brief Move constructor.
	sso_vector(sso_vector&& other) noexcept
		: alloc_(std::move(other.alloc_)), size_(0), capacity_(sso_capacity), data_(sso_data()) {
		move_from_other(std::move(other));
	}

	/// @brief Move constructor with allocator.
	sso_vector(sso_vector&& other, const Allocator& alloc)
		: alloc_(alloc), size_(0), capacity_(sso_capacity), data_(sso_data()) {
		if (alloc_ == other.alloc_) {
			move_from_other(std::move(other));
		} else {
			assign_range(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
			other.clear();
		}
	}

	/// @brief Destructor.
	~sso_vector() {
		clear();
		release_heap();
	}

	/// @brief Copy assignment.
	sso_vector& operator=(const sso_vector& other) {
		if (this == &other) {
			return *this;
		}
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
			if (alloc_ != other.alloc_) {
				clear();
				release_heap();
				alloc_ = other.alloc_;
			}
		}
		assign_range(other.begin(), other.end());
		return *this;
	}

	/// @brief Move assignment.
	sso_vector& operator=(sso_vector&& other) noexcept(std::allocator_traits<Allocator>::is_always_equal::value) {
		if (this == &other) {
			return *this;
		}
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
			clear();
			release_heap();
			alloc_ = std::move(other.alloc_);
			move_from_other(std::move(other));
			return *this;
		}
		if (alloc_ == other.alloc_) {
			clear();
			release_heap();
			move_from_other(std::move(other));
			return *this;
		}
		assign_range(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
		other.clear();
		return *this;
	}

	/// @brief Assign from initializer list.
	sso_vector& operator=(std::initializer_list<T> init) {
		assign_range(init.begin(), init.end());
		return *this;
	}

	/// @brief Returns the allocator.
	allocator_type get_allocator() const noexcept { return alloc_; }

	/// @brief Element access.
	reference at(size_type pos) {
		if (pos >= size_) {
			throw std::out_of_range("sso_vector::at out of range");
		}
		return data_[pos];
	}

	/// @brief Element access (const).
	const_reference at(size_type pos) const {
		if (pos >= size_) {
			throw std::out_of_range("sso_vector::at out of range");
		}
		return data_[pos];
	}

	/// @brief Element access (unchecked).
	reference operator[](size_type pos) { return data_[pos]; }

	/// @brief Element access (unchecked, const).
	const_reference operator[](size_type pos) const { return data_[pos]; }

	/// @brief Access first element.
	reference front() { return data_[0]; }

	/// @brief Access first element (const).
	const_reference front() const { return data_[0]; }

	/// @brief Access last element.
	reference back() { return data_[size_ - 1]; }

	/// @brief Access last element (const).
	const_reference back() const { return data_[size_ - 1]; }

	/// @brief Access underlying data.
	pointer data() noexcept { return data_; }

	/// @brief Access underlying data (const).
	const_pointer data() const noexcept { return data_; }

	/// @brief Begin iterator.
	iterator begin() noexcept { return data_; }

	/// @brief Begin iterator (const).
	const_iterator begin() const noexcept { return data_; }

	/// @brief Begin iterator (const).
	const_iterator cbegin() const noexcept { return data_; }

	/// @brief End iterator.
	iterator end() noexcept { return data_ + size_; }

	/// @brief End iterator (const).
	const_iterator end() const noexcept { return data_ + size_; }

	/// @brief End iterator (const).
	const_iterator cend() const noexcept { return data_ + size_; }

	/// @brief Reverse begin iterator.
	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

	/// @brief Reverse begin iterator (const).
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

	/// @brief Reverse begin iterator (const).
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

	/// @brief Reverse end iterator.
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

	/// @brief Reverse end iterator (const).
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	/// @brief Reverse end iterator (const).
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	/// @brief Checks if container is empty.
	bool empty() const noexcept { return size_ == 0; }

	/// @brief Returns the number of elements.
	size_type size() const noexcept { return size_; }

	/// @brief Returns the current capacity.
	size_type capacity() const noexcept { return capacity_; }

	/// @brief Returns the maximum size.
	size_type max_size() const noexcept {
		return std::allocator_traits<Allocator>::max_size(alloc_);
	}

	/// @brief Reserve storage.
	void reserve(size_type new_cap) {
		if (new_cap <= capacity_) {
			return;
		}
		reallocate(new_cap);
	}

	/// @brief Shrinks storage to fit size.
	void shrink_to_fit() {
		if (size_ == capacity_) {
			return;
		}
		if (size_ <= sso_capacity) {
			move_to_sso();
			return;
		}
		reallocate(size_);
	}

	/// @brief Clears the contents.
	void clear() noexcept {
		destroy_range(data_, data_ + size_);
		size_ = 0;
	}

	/// @brief Inserts a value at position.
	iterator insert(const_iterator pos, const T& value) {
		return emplace(pos, value);
	}

	/// @brief Inserts a value at position (move).
	iterator insert(const_iterator pos, T&& value) {
		return emplace(pos, std::move(value));
	}

	/// @brief Inserts count copies of value.
	iterator insert(const_iterator pos, size_type count, const T& value) {
		const size_type index = static_cast<size_type>(pos - cbegin());
		if (count == 0) {
			return begin() + index;
		}
		ensure_capacity(size_ + count);
		iterator insert_pos = begin() + index;
		move_tail_backward(insert_pos, count);
		std::uninitialized_fill_n(insert_pos, count, value);
		size_ += count;
		return insert_pos;
	}

	/// @brief Inserts a range.
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	iterator insert(const_iterator pos, InputIt first, InputIt last) {
		const size_type index = static_cast<size_type>(pos - cbegin());
		insert_range(index, first, last);
		return begin() + index;
	}

	/// @brief Inserts an initializer list.
	iterator insert(const_iterator pos, std::initializer_list<T> init) {
		return insert(pos, init.begin(), init.end());
	}

	/// @brief Emplaces an element at position.
	template<typename... Args>
	iterator emplace(const_iterator pos, Args&&... args) {
		const size_type index = static_cast<size_type>(pos - cbegin());
		ensure_capacity(size_ + 1);
		iterator insert_pos = begin() + index;
		if (insert_pos == end()) {
			std::allocator_traits<Allocator>::construct(alloc_, data_ + size_, std::forward<Args>(args)...);
			size_ += 1;
			return end() - 1;
		}
		move_tail_backward(insert_pos, 1);
		std::allocator_traits<Allocator>::construct(alloc_, insert_pos, std::forward<Args>(args)...);
		size_ += 1;
		return insert_pos;
	}

	/// @brief Emplaces an element at the end.
	template<typename... Args>
	reference emplace_back(Args&&... args) {
		ensure_capacity(size_ + 1);
		std::allocator_traits<Allocator>::construct(alloc_, data_ + size_, std::forward<Args>(args)...);
		++size_;
		return back();
	}

	/// @brief Adds an element at the end.
	void push_back(const T& value) {
		emplace_back(value);
	}

	/// @brief Adds an element at the end (move).
	void push_back(T&& value) {
		emplace_back(std::move(value));
	}

	/// @brief Removes the last element.
	void pop_back() {
		if (size_ == 0) {
			return;
		}
		std::allocator_traits<Allocator>::destroy(alloc_, data_ + size_ - 1);
		--size_;
	}

	/// @brief Erases the element at position.
	iterator erase(const_iterator pos) {
		return erase(pos, pos + 1);
	}

	/// @brief Erases a range.
	iterator erase(const_iterator first, const_iterator last) {
		const size_type index = static_cast<size_type>(first - cbegin());
		const size_type count = static_cast<size_type>(last - first);
		if (count == 0) {
			return begin() + index;
		}
		iterator dst = begin() + index;
		iterator src = dst + count;
		for (; src != end(); ++dst, ++src) {
			*dst = std::move(*src);
		}
		destroy_range(end() - count, end());
		size_ -= count;
		return begin() + index;
	}

	/// @brief Resizes to count default-inserted elements.
	void resize(size_type count) {
		resize(count, T());
	}

	/// @brief Resizes to count elements, value-initializing new ones with value.
	void resize(size_type count, const T& value) {
		if (count < size_) {
			destroy_range(data_ + count, data_ + size_);
			size_ = count;
			return;
		}
		if (count == size_) {
			return;
		}
		ensure_capacity(count);
		uninitialized_fill_n(data_ + size_, count - size_, value);
		size_ = count;
	}

	/// @brief Swaps with another vector.
	void swap(sso_vector& other) noexcept(std::allocator_traits<Allocator>::is_always_equal::value) {
		if (this == &other) {
			return;
		}
		if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value) {
			std::swap(alloc_, other.alloc_);
		}
		if (using_sso() && other.using_sso()) {
			swap_sso(other);
			return;
		}
		if (using_sso() || other.using_sso()) {
			sso_vector tmp(std::move(*this));
			*this = std::move(other);
			other = std::move(tmp);
			return;
		}
		std::swap(data_, other.data_);
		std::swap(size_, other.size_);
		std::swap(capacity_, other.capacity_);
	}

	/// @brief Assigns count copies of value.
	void assign(size_type count, const T& value) {
		clear();
		ensure_capacity(count);
		uninitialized_fill_n(data_, count, value);
		size_ = count;
	}

	/// @brief Assigns from range.
	template<typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
	void assign(InputIt first, InputIt last) {
		assign_range(first, last);
	}

	/// @brief Assigns from initializer list.
	void assign(std::initializer_list<T> init) {
		assign_range(init.begin(), init.end());
	}

	/// @brief Returns whether storage is using the inline buffer.
	bool using_sso() const noexcept {
		return data_ == sso_data();
	}

private:
	static constexpr size_type sso_bytes = 3 * sizeof(void*);
	static constexpr size_type sso_capacity = (sizeof(T) <= sso_bytes) ? (sso_bytes / sizeof(T)) : 0;

	pointer sso_data() noexcept {
		return reinterpret_cast<pointer>(sso_buffer_);
	}

	const_pointer sso_data() const noexcept {
		return reinterpret_cast<const_pointer>(sso_buffer_);
	}

	void fixup_sso_pointer() noexcept {
		if (data_ == nullptr) {
			return;
		}
		if (capacity_ <= sso_capacity) {
			data_ = sso_data();
		}
	}

	void ensure_capacity(size_type desired) {
		if (desired <= capacity_) {
			return;
		}
		reallocate(growth_capacity(desired));
	}

	size_type growth_capacity(size_type desired) const {
		const size_type max = max_size();
		if (desired > max) {
			throw std::length_error("sso_vector capacity exceeded");
		}
		const size_type doubled = capacity_ > 0 ? capacity_ * 2 : 1;
		return std::max(desired, doubled);
	}

	void move_to_sso() {
		if (using_sso()) {
			capacity_ = sso_capacity;
			return;
		}
		pointer old = data_;
		pointer target = sso_data();
		for (size_type i = 0; i < size_; ++i) {
			std::allocator_traits<Allocator>::construct(alloc_, target + i, std::move(old[i]));
		}
		destroy_range(old, old + size_);
		deallocate_heap(old, capacity_);
		data_ = target;
		capacity_ = sso_capacity;
	}

	void release_heap() noexcept {
		if (using_sso() || data_ == nullptr) {
			return;
		}
		deallocate_heap(data_, capacity_);
		data_ = sso_data();
		capacity_ = sso_capacity;
	}

	void reallocate(size_type new_cap) {
		pointer new_data = allocate_heap(new_cap);
		pointer old_data = data_;
		size_type old_size = size_;
		for (size_type i = 0; i < old_size; ++i) {
			std::allocator_traits<Allocator>::construct(alloc_, new_data + i, std::move_if_noexcept(old_data[i]));
		}
		destroy_range(old_data, old_data + old_size);
		if (!using_sso()) {
			deallocate_heap(old_data, capacity_);
		}
		data_ = new_data;
		capacity_ = new_cap;
	}

	void move_from_other(sso_vector&& other) {
		if (other.using_sso()) {
			ensure_capacity(other.size_);
			for (size_type i = 0; i < other.size_; ++i) {
				std::allocator_traits<Allocator>::construct(alloc_, data_ + i, std::move(other.data_[i]));
			}
			size_ = other.size_;
			other.clear();
			return;
		}
		data_ = other.data_;
		size_ = other.size_;
		capacity_ = other.capacity_;
		other.data_ = other.sso_data();
		other.capacity_ = sso_capacity;
		other.size_ = 0;
	}

	void swap_sso(sso_vector& other) noexcept {
		const size_type max_size = std::max(size_, other.size_);
		for (size_type i = 0; i < max_size; ++i) {
			if (i < size_ && i < other.size_) {
				using std::swap;
				swap(data_[i], other.data_[i]);
			} else if (i < size_) {
				std::allocator_traits<Allocator>::construct(other.alloc_, other.data_ + i, std::move(data_[i]));
				std::allocator_traits<Allocator>::destroy(alloc_, data_ + i);
			} else if (i < other.size_) {
				std::allocator_traits<Allocator>::construct(alloc_, data_ + i, std::move(other.data_[i]));
				std::allocator_traits<Allocator>::destroy(other.alloc_, other.data_ + i);
			}
		}
		std::swap(size_, other.size_);
	}

	void move_tail_backward(iterator insert_pos, size_type count) {
		if (count == 0) {
			return;
		}
		iterator old_end = end();
		for (iterator src = old_end; src != insert_pos; --src) {
			iterator src_elem = src - 1;
			iterator dst_elem = src_elem + count;
			if (dst_elem >= old_end) {
				std::allocator_traits<Allocator>::construct(alloc_, dst_elem, std::move(*src_elem));
			} else {
				*dst_elem = std::move(*src_elem);
			}
		}
		const size_type tail = static_cast<size_type>(old_end - insert_pos);
		const size_type destroy_count = std::min(count, tail);
		destroy_range(insert_pos, insert_pos + destroy_count);
	}

	template<typename InputIt>
	void insert_range(size_type index, InputIt first, InputIt last) {
		using category = typename std::iterator_traits<InputIt>::iterator_category;
		if constexpr (std::is_base_of_v<std::forward_iterator_tag, category>) {
			const size_type count = static_cast<size_type>(std::distance(first, last));
			if (count == 0) {
				return;
			}
			ensure_capacity(size_ + count);
			iterator insert_pos = begin() + index;
			move_tail_backward(insert_pos, count);
			uninitialized_copy_n(first, count, insert_pos);
			size_ += count;
		} else {
			for (; first != last; ++first) {
				emplace(begin() + index++, *first);
			}
		}
	}

	template<typename InputIt>
	void assign_range(InputIt first, InputIt last) {
		clear();
		using category = typename std::iterator_traits<InputIt>::iterator_category;
		if constexpr (std::is_base_of_v<std::forward_iterator_tag, category>) {
			const size_type count = static_cast<size_type>(std::distance(first, last));
			ensure_capacity(count);
			uninitialized_copy_n(first, count, data_);
			size_ = count;
		} else {
			for (; first != last; ++first) {
				emplace_back(*first);
			}
		}
	}

	void uninitialized_default_n(pointer dest, size_type count) {
		for (size_type i = 0; i < count; ++i) {
			std::allocator_traits<Allocator>::construct(alloc_, dest + i);
		}
	}

	void uninitialized_fill_n(pointer dest, size_type count, const T& value) {
		for (size_type i = 0; i < count; ++i) {
			std::allocator_traits<Allocator>::construct(alloc_, dest + i, value);
		}
	}

	template<typename InputIt>
	void uninitialized_copy_n(InputIt src, size_type count, pointer dest) {
		for (size_type i = 0; i < count; ++i, ++src) {
			std::allocator_traits<Allocator>::construct(alloc_, dest + i, *src);
		}
	}

	void destroy_range(pointer first, pointer last) noexcept {
		for (; first != last; ++first) {
			std::allocator_traits<Allocator>::destroy(alloc_, first);
		}
	}

	pointer allocate_heap(size_type count) {
		return std::allocator_traits<Allocator>::allocate(alloc_, count);
	}

	void deallocate_heap(pointer ptr, size_type count) noexcept {
		std::allocator_traits<Allocator>::deallocate(alloc_, ptr, count);
	}

	Allocator alloc_{};
	size_type size_{0};
	size_type capacity_{sso_capacity};
	pointer data_{sso_data()};
	alignas(T) unsigned char sso_buffer_[sso_capacity > 0 ? sso_capacity * sizeof(T) : 1]{};
};

/// @brief Equality comparison.
template<typename T, typename Allocator>
inline bool operator==(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// @brief Lexicographical comparison.
template<typename T, typename Allocator>
inline bool operator<(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

/// @brief Inequality comparison.
template<typename T, typename Allocator>
inline bool operator!=(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return !(lhs == rhs);
}

/// @brief Greater-than comparison.
template<typename T, typename Allocator>
inline bool operator>(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return rhs < lhs;
}

/// @brief Less-or-equal comparison.
template<typename T, typename Allocator>
inline bool operator<=(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return !(rhs < lhs);
}

/// @brief Greater-or-equal comparison.
template<typename T, typename Allocator>
inline bool operator>=(const sso_vector<T, Allocator>& lhs, const sso_vector<T, Allocator>& rhs) {
	return !(lhs < rhs);
}

} // namespace internal

}} // namespace sw::universal
