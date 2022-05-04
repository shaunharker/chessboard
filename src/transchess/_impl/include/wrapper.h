/// wrapper.h
/// Shaun Harker
/// 2022-05-04
/// MIT LICENSE

#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

/// Python Bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

inline void
Binding(py::module &m) {
  // See pybind11 doc
}
