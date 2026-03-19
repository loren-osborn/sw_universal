#pragma once
/// @file pack_element.hpp
/// @brief Internal utility for selecting the Ith type from a template parameter pack.

#include <cstddef>
#include <tuple>

namespace sw::universal::internal_utility {

template <std::size_t I, class... Ts>
using pack_element_t = std::tuple_element_t<I, std::tuple<Ts...>>;

} // namespace sw::universal::internal_utility
