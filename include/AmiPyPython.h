#pragma once

#include "StdAfx.h"

#if PY_MAJOR_VERSION >= 3
PyObject *PyInit_AmiPy(void);
#else
#error AmiPy doesn't support Python 2
#endif