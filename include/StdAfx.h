#pragma once

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <afxmt.h>
#include <afxwin.h>

#include <Plugin_Extended.h>
#include <framework.h>

#include "Logger.h"

extern CStringA __ErrStr;

#define QUIET_ERROR(a, ...)                                                    \
  {                                                                            \
    __ErrStr.Format(a, ##__VA_ARGS__);                                         \
    OutputDebugStringA(__ErrStr + "\n");                                       \
  }

#define WARNING(a, ...)                                                        \
  {                                                                            \
    QUIET_ERROR(a, ##__VA_ARGS__);                                             \
    gLogger("Warning: " + __ErrStr, Logger::MSG_WARNING);                      \
  }
#define PRINT_ERROR(a, ...)                                                    \
  {                                                                            \
    QUIET_ERROR(a, ##__VA_ARGS__);                                             \
    gLogger("Error: " + __ErrStr, Logger::MSG_ERROR);                          \
    AmiError(__ErrStr);                                                        \
  }
