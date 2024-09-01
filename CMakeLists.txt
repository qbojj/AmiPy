cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(AmiPy CXX)

set(CMAKE_MFC_FLAG 2)
add_definitions(-D_AFXDLL)

find_package(Python COMPONENTS Development.SABIModule NumPy REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

add_library(AmiPy SHARED ${SOURCES} ${HEADERS})
target_link_libraries(AmiPy PRIVATE Python::SABIModule Python::NumPy)
target_include_directories(AmiPy PUBLIC include PRIVATE extern/include)
target_compile_definitions(AmiPy PUBLIC
  _CRT_SECURE_NO_WARNINGS
  _USRDLL
  NOMINMAX
  NTDDI_VERSION=NTDDI_WINXP
  _WIN32_WINNT=_WIN32_WINNT_WINXP
  VA_ARGS_ARG_CNT=20
  "VA_ARGS_EMPTY_VAL=(-1e11f)"
  Py_LIMITED_API=0x03090000
  PY_SSIZE_T_CLEAN
  NO_IMPORT_ARRAY
  NPY_TARGET_VERSION=NPY_2_0_API_VERSION
)
target_precompile_headers(AmiPy PRIVATE 
  <Python.h>
  <numpy/ndarrayobject.h>
  <afxmt.h>
  <afxwin.h>
  <Plugin_Extended.h>
  <framework.h>
)
