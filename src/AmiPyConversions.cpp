#include "AmiPyConversions.h"

#include "AmiPyFunctions.h"
#include "AmiPyIsolation.h"
#include "AmiPyPython.h"
#include "Logger.h"

#include <algorithm>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <map>
#include <windows.h>
#endif

#include <semaphore>
#include <string>
#include <thread>
#include <cassert>
#include <utility>

std::atomic_bool g_bIsClosing = false;

PyInterpreterState *g_pMainIS;
static PyThreadState *pMainTS;

static std::atomic<bool> bInitError = false;
static std::string oInitErrorMsg = "";

static std::once_flag bInitializingFlag;
static std::atomic<bool> bProperlyInitialized = false;

static std::binary_semaphore oClose{0};
static std::binary_semaphore oInitialized{0};

static std::timed_mutex oMainPythonThreadCS;

static std::atomic<unsigned int> bFinalizing = 0;

static bool StartAmiPy();
static bool EndAmiPy();

#ifdef _WIN32
#define VAL_AND_NAME(v) {v, #v}

static const std::map<DWORD, const char *> exception_names{
    VAL_AND_NAME(EXCEPTION_ACCESS_VIOLATION),
    VAL_AND_NAME(EXCEPTION_ARRAY_BOUNDS_EXCEEDED),
    VAL_AND_NAME(EXCEPTION_BREAKPOINT),
    VAL_AND_NAME(EXCEPTION_DATATYPE_MISALIGNMENT),
    VAL_AND_NAME(EXCEPTION_FLT_DENORMAL_OPERAND),
    VAL_AND_NAME(EXCEPTION_FLT_DIVIDE_BY_ZERO),
    VAL_AND_NAME(EXCEPTION_FLT_INEXACT_RESULT),
    VAL_AND_NAME(EXCEPTION_FLT_INVALID_OPERATION),
    VAL_AND_NAME(EXCEPTION_FLT_OVERFLOW),
    VAL_AND_NAME(EXCEPTION_FLT_STACK_CHECK),
    VAL_AND_NAME(EXCEPTION_FLT_UNDERFLOW),
    VAL_AND_NAME(EXCEPTION_GUARD_PAGE),
    VAL_AND_NAME(EXCEPTION_ILLEGAL_INSTRUCTION),
    VAL_AND_NAME(EXCEPTION_IN_PAGE_ERROR),
    VAL_AND_NAME(EXCEPTION_INT_DIVIDE_BY_ZERO),
    VAL_AND_NAME(EXCEPTION_INT_OVERFLOW),
    VAL_AND_NAME(EXCEPTION_INVALID_DISPOSITION),
    VAL_AND_NAME(EXCEPTION_INVALID_HANDLE),
    VAL_AND_NAME(EXCEPTION_NONCONTINUABLE_EXCEPTION),
    VAL_AND_NAME(EXCEPTION_PRIV_INSTRUCTION),
    VAL_AND_NAME(EXCEPTION_SINGLE_STEP),
    VAL_AND_NAME(EXCEPTION_STACK_OVERFLOW),
    VAL_AND_NAME(STATUS_UNWIND_CONSOLIDATE),
};

static void try_init_except(char msgBuff[1024], unsigned long ec) {
  auto it = exception_names.find(ec);

  sprintf_s(msgBuff, sizeof(msgBuff),
            "Exception during python initialization (%s)",
            it != exception_names.end() ? it->second : "unknown");

  OutputDebugStringA(msgBuff);
  gLogger(msgBuff, Logger::MSG_ERROR);

  bProperlyInitialized = false;
  bInitError = true;
  oInitErrorMsg = msgBuff;
}

static void try_init(char msgBuff[1024]) {
  __try {
    bProperlyInitialized = StartAmiPy();
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    try_init_except(msgBuff, GetExceptionCode());
  }
}
#else
static void try_init(char msgBuff[1024]) {
  bProperlyInitialized = StartAmiPy();
}
#endif

static void MainPythonThreadExecutor(std::unique_lock<std::timed_mutex>) {
  gLogger("MainPythonThread started", Logger::MSG_DEBUG);

  char msgBuff[1024] = "";
  try_init(msgBuff);

  gLogger("Setting initialized event", Logger::MSG_DEBUG);

  oInitialized.release();

  if (!bProperlyInitialized) {
    gLogger(
        "AmiPy was not initialized successfully => stopping MainPythonThread",
        Logger::MSG_WARNING);
    return;
  }

  gLogger("waiting for close event", Logger::MSG_DEBUG);

  oClose.acquire();

  gLogger("close event recived", Logger::MSG_DEBUG);
  gLogger("MainPythonThread stopped", Logger::MSG_DEBUG);
}

static inline bool InitAmiPyModule() {
  static bool bInitialized = false;

  return bInitialized
             ? true
             : bInitialized = !PyImport_AppendInittab("AmiPy", PyInit_AmiPy);
}

void **PyArray_API = NULL;
int PyArray_RUNTIME_VERSION;

/*
 * On exceedingly few platforms these sizes may not match, in which case
 * We do not support older NumPy versions at all.
 */
static_assert(sizeof(Py_ssize_t) == sizeof(Py_intptr_t) ||
                  PyArray_RUNTIME_VERSION < NPY_2_0_API_VERSION,
              "module compiled against NumPy 2.0 but running on NumPy 1.x. "
              "Unfortunately, this is not supported on niche platforms where "
              "`sizeof(size_t) != sizeof(inptr_t)`.");

static inline bool InitArrayAPI() {
  if (PyArray_API != NULL)
    return true;

  int st;
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  }

  if (numpy == NULL) {
    return false;
  }

  PyObject *c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
    return false;
  }

  if (!PyCapsule_CheckExact(c_api)) {
    PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
    Py_DECREF(c_api);
    return false;
  }
  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyArray_API == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
    return false;
  }

  /*
   * Perform runtime check of C API version.  As of now NumPy 2.0 is ABI
   * backwards compatible (in the exposed feature subset!) for all practical
   * purposes.
   */
  if (NPY_VERSION < PyArray_GetNDArrayCVersion()) {
    PyErr_Format(PyExc_RuntimeError,
                 "module compiled against "
                 "ABI version 0x%x but this version of numpy is 0x%x",
                 (int)NPY_VERSION, (int)PyArray_GetNDArrayCVersion());
    return false;
  }
  PyArray_RUNTIME_VERSION = (int)PyArray_GetNDArrayCFeatureVersion();
  if (NPY_FEATURE_VERSION > PyArray_RUNTIME_VERSION) {
    PyErr_Format(
        PyExc_RuntimeError,
        "module was compiled against NumPy C-API version 0x%x "
        "(NumPy " NPY_FEATURE_VERSION_STRING ") "
        "but the running NumPy has C-API version 0x%x. "
        "Check the section C-API incompatibility at the "
        "Troubleshooting ImportError section at "
        "https://numpy.org/devdocs/user/troubleshooting-importerror.html"
        "#c-api-incompatibility "
        "for indications on how to solve this problem.",
        (int)NPY_FEATURE_VERSION, PyArray_RUNTIME_VERSION);
    return false;
  }

  /*
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
    PyErr_SetString(PyExc_RuntimeError,
                    "FATAL: module compiled as unknown endian");
    return false;
  }
#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
    PyErr_SetString(PyExc_RuntimeError,
                    "FATAL: module compiled as big endian, but "
                    "detected different endianness at runtime");
    return false;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  if (st != NPY_CPU_LITTLE) {
    PyErr_SetString(PyExc_RuntimeError,
                    "FATAL: module compiled as little endian, but "
                    "detected different endianness at runtime");
    return false;
  }
#endif

  std::string msg =
      std::format("Loaded NumPy version 0x{:x}", PyArray_RUNTIME_VERSION);
#ifdef _WIN32
  OutputDebugStringA(msg);
#endif
  gLogger(msg, Logger::MSG_MAIN_EVENTS);

  return true;
}

#define STDERR_FILE "./stderr.txt"
#define STDOUT_FILE "./stdout.txt"
static bool bRedirectedStdIO = false;

static bool RedirectStdIO() {
  if (bRedirectedStdIO)
    return true;

  if (!std::freopen(STDERR_FILE, "w", stderr))
    return false;
  if (!std::freopen(STDOUT_FILE, "w", stdout)) {
    fclose(stderr);
    return false;
  }

  return bRedirectedStdIO = true;
}

static bool CloseRedirectedStdIO() {
  if (!bRedirectedStdIO)
    return true;
  bRedirectedStdIO = false;

  std::fclose(stderr);
  std::fclose(stdout);

  return true;
}

#ifdef _WIN32
#define INIT_ERROR_AND_EXIT(msg)                                               \
  {                                                                            \
    bInitError = true;                                                         \
    oInitErrorMsg = msg;                                                       \
    OutputDebugStringA(oInitErrorMsg.data());                                  \
    gLogger("Error: " + oInitErrorMsg, Logger::MSG_ERROR);                     \
    return false;                                                              \
  }
#else
#define INIT_ERROR_AND_EXIT(msg)                                               \
  {                                                                            \
    bInitError = true;                                                         \
    oInitErrorMsg = msg;                                                       \
    gLogger("Error: " + oInitErrorMsg, Logger::MSG_ERROR);                     \
    return false;                                                              \
  }
#endif

inline static void FormatPyErr(std::string &msg) {
  msg += '\n' + AmiPyPythonErrorToString();
}

static unsigned int GetPythonVersionHex() {
  std::string versionValStr = Py_GetVersion();

  int end = versionValStr.find(' ');
  versionValStr = versionValStr.substr(0, end);
  for (char &c : versionValStr)
    c = std::tolower(c);

  int numOfDots = std::count(versionValStr.begin(), versionValStr.end(), '.');

  int maj, min, mic = 0, ser = 0;
  char rel_lev = 'f';

  if (numOfDots == 1) {
    if (std::sscanf(versionValStr.data(), "%d.%d%c%d", &maj, &min, &rel_lev,
                    &ser) < 2)
      return 0;
  } else if (numOfDots == 2) {
    if (std::sscanf(versionValStr.data(), "%d.%d.%d%c%d", &maj, &min, &mic,
                    &rel_lev, &ser) < 3)
      return 0;
  } else
    return 0;

  if (rel_lev != 'a' && rel_lev != 'b' && rel_lev != 'c')
    rel_lev = 'f';

  unsigned int version = (maj << 24) | (min << 16) | (mic << 8) |
                         ((rel_lev - 'a' + 0xA) << 4) | (ser << 0);

  return version;
}

static bool StartPython() {
  gLogger("Starting python", Logger::MSG_DEBUG);

  gLogger("Checking if valid python version", Logger::MSG_DEBUG);
  if (GetPythonVersionHex() < Py_LIMITED_API)
    INIT_ERROR_AND_EXIT("Python version is below required");

  gLogger("Redirecting standard streams", Logger::MSG_DEBUG);
  if (!RedirectStdIO())
    INIT_ERROR_AND_EXIT("Cannot redirect std IO");

  gLogger("Initializing AmiPy module", Logger::MSG_DEBUG);
  if (!InitAmiPyModule())
    INIT_ERROR_AND_EXIT("Cannot import AmiPy Module");

  gLogger("Initializing python...", Logger::MSG_DEBUG);
  Py_InitializeEx(0);
  gLogger("Initializing python exited", Logger::MSG_DEBUG);

  if (!Py_IsInitialized()) {
    Py_Finalize();
    INIT_ERROR_AND_EXIT("Cannot initialize python");
  }

  pMainTS = PyEval_SaveThread();
  gLogger("Python started", Logger::MSG_MAIN_EVENTS);

  return true;
}

static bool StartAmiPy() {
  if (!StartPython())
    return false;

  gLogger("Initializing numpy", Logger::MSG_DEBUG);

  PyEval_RestoreThread(pMainTS);

  if (!InitArrayAPI()) {
    std::string msg =
        "cannot import array API (no numpy?) [numpy.core.multiarray "
        "failed to import]";

    FormatPyErr(msg);
    INIT_ERROR_AND_EXIT(msg);
  }

  PyEval_SaveThread();

  gLogger("Numpy initialized", Logger::MSG_MAIN_EVENTS);

  return true;
}

#undef INIT_ERROR_AND_EXIT

static bool EndAmiPy() {
  bool ret = true;

  if (bProperlyInitialized) {
    gLogger("Ending python", Logger::MSG_MAIN_EVENTS);

    PyEval_RestoreThread(pMainTS);

    AmiPyIsolation_ClearAll(); // close all resources
    Py_Finalize();
    ret = true;
  }

  if (!ret) {
    gLogger("Error: Cannot finalize Python", Logger::MSG_ERROR);
#ifdef _WIN32
    OutputDebugStringA("Error: Cannot finalize Python\n");
#endif
  } else
    gLogger("Python Ended", Logger::MSG_MAIN_EVENTS);

  CloseRedirectedStdIO();

  return ret;
}

bool EnsureInitialized() {
  if (bFinalizing)
    return false;
  if (bProperlyInitialized)
    return true;
  
  gLogger.Initialize();

  std::call_once(bInitializingFlag, []{
    gLogger("starting MainPythonThread thread", Logger::MSG_DEBUG);
    std::thread{MainPythonThreadExecutor, std::unique_lock{oMainPythonThreadCS}}.detach();
  });

  if (bFinalizing)
    return false;

  if (oInitialized.try_acquire_for(std::chrono::seconds(20))) {
    std::string msg = "Error: waiting for initialized event: timeout";

#ifdef _WIN32
    OutputDebugStringA(msg.data());
#endif
    gLogger("Error: " + msg, Logger::MSG_ERROR);

    return false;
  }

  oInitialized.release();

  if (bInitError) {
    PRINT_ERROR(oInitErrorMsg.data());
    return false;
  }

  return bProperlyInitialized;
}

bool SaveClose() {
  bFinalizing = true;
  
  bool never_initialized = false;
  std::call_once(bInitializingFlag, [&]{
    never_initialized = true;
  });

  if (never_initialized || !bProperlyInitialized)
    return true;

  oClose.release();
  gLogger("waiting for close of MainPythonThread", Logger::MSG_DEBUG);

  if (!oMainPythonThreadCS.try_lock_for(std::chrono::seconds(30))) {
    std::string msg = "Error: python thread did not close: timeout";

#ifdef _WIN32
    OutputDebugStringA(msg.data());
#endif
    gLogger(msg, Logger::MSG_ERROR);

    return false;
  }

  oMainPythonThreadCS.unlock();

  return true;
}

void ForceClose() {
  bFinalizing = true;
  // we cannot do anything else
}

PyObject *AmiVarToObj(AmiVar variable) {
  gLogger("Converting AFL var to Py var", Logger::MSG_DEBUG);

  npy_intp dims[2];
  PyArrayObject *pArrayObj;

  switch (variable.type) {
  case VAR_NONE:
    return Py_IncRef(Py_None), Py_None;

  case VAR_FLOAT:
    if (IS_EMPTY(variable.val))
      return Py_IncRef(Py_None), Py_None;
    else
      return PyFloat_FromDouble(variable.val);

  case VAR_STRING:
    return PyUnicode_FromString(variable.string);

  case VAR_ARRAY:
    dims[0] = gSite.GetArraySize();

    pArrayObj = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);

    if (!pArrayObj)
      return NULL;

    memcpy(PyArray_DATA(pArrayObj), variable.array, dims[0] * sizeof(float));

    return (PyObject *)pArrayObj;

  case VAR_MATRIX:
    dims[0] = variable.matrix->rows;
    dims[1] = variable.matrix->cols;

    pArrayObj = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);

    if (!pArrayObj)
      return NULL;

    memcpy(PyArray_DATA(pArrayObj), variable.matrix->data,
           PyArray_NBYTES(pArrayObj));

    return (PyObject *)pArrayObj;

  default:
    PRINT_ERROR("AmiPy doesnot support AFL <%d> type", variable.type);
    return Py_IncRef(Py_None), Py_None;
  }
}

#define RETURN_WITH_ERROR(...)                                                 \
  do {                                                                         \
    PyErr_Clear();                                                             \
    PRINT_ERROR(__VA_ARGS__);                                                  \
    return AmiVar{VAR_FLOAT, EMPTY_VAL};                                       \
  } while (0)

#define RETURN_WITH_PY_ERROR(msg)                                              \
  do {                                                                         \
    AmiPyPrintError(msg);                                                      \
    return AmiVar{VAR_FLOAT, EMPTY_VAL};                                       \
  } while (0)

AmiVar ObjToAmiVar(PyObject *variable) {
  if (variable == NULL || variable == Py_None)
    return AmiVar{VAR_FLOAT, EMPTY_VAL};

  gLogger("Converting Py var to AFL var", Logger::MSG_DEBUG);

  if (PyArray_IsZeroDim(variable)) {
    Py_IncRef(variable);
    PyObject *pObj = PyArray_Return((PyArrayObject *)variable);

    if (!pObj)
      RETURN_WITH_PY_ERROR("Cannot convert numpy 0-dim ndarray to scalar");
    AmiVar Var = ObjToAmiVar(pObj);

    Py_DecRef(pObj);
    return Var;
  }

  if (PyArray_Check(variable)) {
    PyArrayObject *pObj = (PyArrayObject *)variable;

    int ndim = PyArray_NDIM(pObj);
    bool bArray = ndim == 1 && PyArray_DIM(pObj, 0) == gSite.GetArraySize();
    bool bMatrix = ndim == 2;

    if (ndim != 1 && ndim != 2)
      RETURN_WITH_ERROR(
          "Result array must have 1 dim. for AFL array or 2 dimensions");
    if (ndim == 1 && !bArray)
      RETURN_WITH_ERROR(
          "Result array with 1 dim. must be of size of AFL array");

    PyArrayObject *pConverted = (PyArrayObject *)PyArray_Cast(pObj, NPY_FLOAT);
    if (!pConverted)
      RETURN_WITH_PY_ERROR("Cannot cast array type to float");

    AmiVar Var;

    switch (ndim) {
    case 1:
      // AFL Array
      assert(bArray);
      Var = SetArr((float *)PyArray_DATA(pConverted));
      break;
    case 2:
      // AFL Matrix
      assert(bMatrix);
      Var = SetMat((int)PyArray_DIM(pObj, 0), (int)PyArray_DIM(pObj, 1),
                   (float *)PyArray_DATA(pConverted));
      break;

    default:
      std::unreachable(); // checked for it before
    }

    Py_DecRef((PyObject *)pConverted);
    return Var;
  }

  if (PyUnicode_Check(variable)) {
    ASCIICapsule Text = ASCIICapsule::FromUnicode(variable, false);

    if (!Text.Data())
      RETURN_WITH_PY_ERROR("Cannot convert returned string to ascii string");
    return SetTxt(Text.Data());
  }

  if (PyNumber_Check(variable)) {
    float res = (float)PyFloat_AsDouble(variable);
    if (PyErr_Occurred())
      RETURN_WITH_PY_ERROR("Cannot convert returned number to float");

    return SetVal((float)res);
    ;
  }

  RETURN_WITH_ERROR("Unsupported return type");
}

////////////////////////////////////////////////////////////////////////////////////////////
// ASCIICapsule function definitions

ASCIICapsule::ASCIICapsule() { m_pASCIIObj = NULL; }

ASCIICapsule::ASCIICapsule(PyObject *pASCIIObj, bool Steal) {
  if (!Steal && pASCIIObj)
    Py_IncRef(pASCIIObj);
  m_pASCIIObj = pASCIIObj;
}

ASCIICapsule::ASCIICapsule(const ASCIICapsule &Obj) {
  if (Obj.m_pASCIIObj)
    Py_IncRef(Obj.m_pASCIIObj);
  m_pASCIIObj = Obj.m_pASCIIObj;
}

ASCIICapsule::~ASCIICapsule() {
  if (m_pASCIIObj) {
    Py_DecRef(m_pASCIIObj);
    m_pASCIIObj = NULL;
  }
}

const char *ASCIICapsule::Data() const {
  const char *dat = m_pASCIIObj ? PyBytes_AsString(m_pASCIIObj) : NULL;
  return dat; // ? dat : "<internal read string error>";
}

ASCIICapsule::operator const char *() const { return Data(); }

ASCIICapsule ASCIICapsule::FromUnicode(PyObject *pUnicode, bool Steal) {
  if (!pUnicode)
    return ASCIICapsule();

  PyObject *pASCIIObj = PyUnicode_AsEncodedString(
      pUnicode, "ascii", "replace"); // PyUnicode_AsASCIIString( pUnicode );
  if (Steal)
    Py_DecRef(pUnicode);

  return ASCIICapsule(pASCIIObj, true);
}

ASCIICapsule ASCIICapsule::FromObject(PyObject *pObj, bool Steal) {
  if (!pObj)
    return ASCIICapsule();

  PyObject *pUnicodeObj = PyObject_Str(pObj);
  if (Steal)
    Py_DecRef(pObj);

  return FromUnicode(pUnicodeObj, true);
}

/////////////////////////////////////////////////////////////////////////////////////////
