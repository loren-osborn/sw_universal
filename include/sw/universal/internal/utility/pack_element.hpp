#pragma once
/// @file pack_element.hpp
/// @brief Small internal metaprogramming utility for selecting the Ith type from a pack.

#include <cstddef>
#include <tuple>

namespace sw::universal::internal_utility {

/// @brief Selects the Ith type from the parameter pack `Ts...`.
/// @tparam I Zero-based element index within `Ts...`.
/// @tparam Ts Candidate types.
/// @details This shared alias exists so nearby internal components can use one
///          consistent "pack element by index" utility instead of each spelling
///          their own `std::tuple_element_t<std::tuple<...>>` wrapper.
template <std::size_t I, class... Ts>
using pack_element_t = std::tuple_element_t<I, std::tuple<Ts...>>;

} // namespace sw::universal::internal_utility
