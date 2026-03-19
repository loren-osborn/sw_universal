// pack_element.cpp: unit tests for pack_element_t
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include <universal/internal/utility/pack_element.hpp>
#include <universal/internal/bitvector/bitfield_pack.hpp>

using sw::universal::bitfield_field_spec;
using sw::universal::bitfield_remainder;
using sw::universal::internal_utility::pack_element_t;

static_assert(std::is_same_v<pack_element_t<0, int, double, std::string, std::uint32_t>, int>);
static_assert(std::is_same_v<pack_element_t<1, int, double, std::string, std::uint32_t>, double>);
static_assert(std::is_same_v<pack_element_t<3, int, double, std::string, std::uint32_t>, std::uint32_t>);

using bitfield_specs = pack_element_t<2, bitfield_field_spec<1>, bitfield_field_spec<8>, bitfield_remainder>;
static_assert(std::is_same_v<bitfield_specs, bitfield_remainder>);

int main() {
	std::cout << "OK\n";
	return 0;
}
