#!/usr/bin/env sh
set -eu
TEST_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TEST_BIN=$(mktemp "${TMPDIR:-/tmp}/gausr-footprint.XXXXXX")
trap 'rm -f "$TEST_BIN"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++14 -O2 -Wall -Wextra -pedantic "$TEST_DIR/test_footprint.cpp" -o "$TEST_BIN"
"$TEST_BIN"
