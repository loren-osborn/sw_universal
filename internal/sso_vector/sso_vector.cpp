// sso_vector.cpp: unit tests for sso_vector
//
// Copyright (C) 2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT
//
// This file is part of the universal numbers project, which is released under an MIT Open Source license.
#include <cstdlib>
#include <iostream>
#include <string>

#include <universal/internal/sso_vector/sso_vector.hpp>
#include <universal/verification/test_status.hpp>

namespace {

void check(bool condition, int& failures, const char* label) {
	if (!condition) {
		std::cerr << "FAIL: " << label << "\n";
		++failures;
	}
}

template<typename Exception, typename Fn>
bool expect_throw(int& failures, const char* label, Fn&& fn) {
	try {
		fn();
		std::cerr << "FAIL: " << label << " did not throw\n";
		++failures;
		return false;
	} catch (const Exception&) {
		return true;
	} catch (...) {
		std::cerr << "FAIL: " << label << " threw unexpected exception\n";
		++failures;
		return false;
	}
}

} // namespace

int main() {
	using namespace sw::universal::internal;
	int nrOfFailedTestCases = 0;

	sso_vector<int> v;
	check(v.empty(), nrOfFailedTestCases, "default empty");
	check(v.size() == 0, nrOfFailedTestCases, "default size 0");
	check(v.capacity() >= v.size(), nrOfFailedTestCases, "capacity >= size");

	v.push_back(1);
	v.push_back(2);
	v.push_back(3);
	check(v.size() == 3, nrOfFailedTestCases, "push_back increases size");
	check(v.front() == 1, nrOfFailedTestCases, "front value");
	check(v.back() == 3, nrOfFailedTestCases, "back value");

	v.insert(v.begin() + 1, 5);
	check(v.size() == 4, nrOfFailedTestCases, "insert single element");
	check(v[1] == 5, nrOfFailedTestCases, "insert position value");

	v.erase(v.begin() + 2);
	check(v.size() == 3, nrOfFailedTestCases, "erase single element");
	check(v[1] == 5, nrOfFailedTestCases, "erase keeps order");

	v.resize(5, 7);
	check(v.size() == 5, nrOfFailedTestCases, "resize grows");
	check(v[4] == 7, nrOfFailedTestCases, "resize fill value");
	v.resize(2);
	check(v.size() == 2, nrOfFailedTestCases, "resize shrinks");

	v.assign(4, 9);
	check(v.size() == 4, nrOfFailedTestCases, "assign count");
	check(v[0] == 9 && v[3] == 9, nrOfFailedTestCases, "assign count values");

	v.clear();
	check(v.empty(), nrOfFailedTestCases, "clear empties");

	sso_vector<std::string> vs({"a", "b", "c"});
	check(vs.size() == 3, nrOfFailedTestCases, "init list size");
	check(vs[1] == "b", nrOfFailedTestCases, "init list values");

	sso_vector<std::string> vs_copy(vs);
	check(vs_copy == vs, nrOfFailedTestCases, "copy equality");

	sso_vector<std::string> vs_move(std::move(vs_copy));
	check(vs_move.size() == 3, nrOfFailedTestCases, "move size");
	check(vs_move[2] == "c", nrOfFailedTestCases, "move data");

	sso_vector<int> range_src;
	for (int i = 0; i < 10; ++i) range_src.push_back(i);
	sso_vector<int> range_dst(range_src.begin(), range_src.end());
	check(range_dst.size() == 10, nrOfFailedTestCases, "range constructor size");
	check(range_dst[9] == 9, nrOfFailedTestCases, "range constructor values");

	const auto old_capacity = range_dst.capacity();
	range_dst.reserve(old_capacity + 10);
	check(range_dst.capacity() >= old_capacity + 10, nrOfFailedTestCases, "reserve grows capacity");
	range_dst.shrink_to_fit();
	check(range_dst.capacity() >= range_dst.size(), nrOfFailedTestCases, "shrink_to_fit keeps capacity >= size");

	expect_throw<std::out_of_range>(nrOfFailedTestCases, "at throws", [&]() {
		(void)range_dst.at(1000);
	});

	sw::universal::ReportTestResult(nrOfFailedTestCases, "sso_vector", "unit test");
	return (nrOfFailedTestCases > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
