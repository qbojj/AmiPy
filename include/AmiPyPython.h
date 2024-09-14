#pragma once

#include <Python.h>

static_assert(PY_MAJOR_VERSION >= 3, "AmiPy only supports Python 3");

PyObject *PyInit_AmiPy(void);
