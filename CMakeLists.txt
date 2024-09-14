cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(AmiPy VERSION 1.0.3 LANGUAGES CXX)

set(CMAKE_MFC_FLAG 2)
add_definitions(-D_AFXDLL)

find_package(Python COMPONENTS Development.SABIModule NumPy REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

add_library(AmiPy MODULE ${SOURCES} ${HEADERS})
target_compile_features(AmiPy PUBLIC cxx_std_23)
target_link_libraries(AmiPy PRIVATE Python::SABIModule Python::NumPy)
target_include_directories(AmiPy PUBLIC include PRIVATE extern/include)
target_compile_definitions(AmiPy PUBLIC
  AMIPY_PLUGIN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  AMIPY_PLUGIN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  AMIPY_PLUGIN_VERSION_RELEASE=${PROJECT_VERSION_PATCH}
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
  NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
  DEFAULT_LOG_PATH=\"AmiPy.log\"
  INI_PATH=\"AmiPy.ini\"
)
target_precompile_headers(AmiPy PRIVATE 
  <Python.h>
  <numpy/ndarrayobject.h>
)
set_target_properties(AmiPy PROPERTIES CXX_VISIBILITY_PRESET hidden)

install(TARGETS AmiPy
RUNTIME
    COMPONENT Runtime
    DESTINATION ./
LIBRARY
    COMPONENT Runtime
    DESTINATION ./
)
install(FILES LICENCE.txt DESTINATION ./)
