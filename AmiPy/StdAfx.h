#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX // do not defile min & max

#define NTDDI_VERSION	NTDDI_WINXP
#define _WIN32_WINNT	_WIN32_WINNT_WINXP

#define VA_ARGS_ARG_CNT 20
#define VA_ARGS_EMPTY_VAL (-1e11f)

#define Py_LIMITED_API 0x03070000 // first version that has ABI

#define PY_SSIZE_T_CLEAN
#define NO_IMPORT_ARRAY
#include <python/Python.h>
#include <numpy/ndarrayobject.h> // from first numpy that supported Python 3 [1.5.0]

#if !defined(PY_VERSION_HEX) || ( defined(Py_LIMITED_API) && PY_VERSION_HEX < Py_LIMITED_API )
#error Python headers version is below min required
#endif

// MFC
#include <afxwin.h>
#include <afxmt.h>

#include "framework.h"
#include "Plugin_Extended.h"

#include "Logger.h"

#define QUIET_ERROR(a, ...) { __ErrStr.Format(a, ##__VA_ARGS__); OutputDebugStringA(__ErrStr+"\n"); }

#define WARNING(a, ...) { QUIET_ERROR(a, ##__VA_ARGS__); gLogger("Warning: " + __ErrStr, Logger::MSG_WARNING); }
#define PRINT_ERROR(a, ...) { QUIET_ERROR(a, ##__VA_ARGS__); gLogger("Error: " + __ErrStr, Logger::MSG_ERROR); AmiError(__ErrStr); }

/////////////////////////////////////////////////////////////////////////////////////////

extern CStringA __ErrStr;