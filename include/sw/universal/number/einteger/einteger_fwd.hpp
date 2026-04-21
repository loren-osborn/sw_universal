#pragma once
// einteger_fwd.hpp: type forwards of adaptive arbitrary precision integer numbers
//
// Copyright (C) 2017-2022 Stillwater Supercomputing, Inc.
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.

#include <string>
#include <vector>

namespace sw { namespace universal {

// forward references
template<typename BlockType, typename BlockContainer = std::vector<BlockType>> class einteger;
template<typename BlockType, typename BlockContainer>
bool parse(const std::string& number, einteger<BlockType, BlockContainer>& v);

}} // namespace sw::universal
