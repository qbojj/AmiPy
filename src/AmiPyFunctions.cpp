#include "AmiPyFunctions.h"

#include "AmiPyConversions.h"
#include "AmiPyIsolation.h"
#include "AmiPyPython.h"
#include "AmiPyRun.h"
#include "AmiVar.h"
#include "Logger.h"

#include <format>

static std::string AmiPyPythonExceptionOnlyToString(PyObject *type,
                                                 PyObject *value,
                                                 PyObject *tracebackModule) {
  assert(type);
  assert(value);
  assert(tracebackModule);

  if (!value || !type || !tracebackModule)
    return "";

  PyObject *format_exception_only =
      PyObject_GetAttrString(tracebackModule, "format_exception_only");

  if (!format_exception_only || !PyCallable_Check(format_exception_only)) {
    if (format_exception_only)
      Py_DecRef(format_exception_only);
    return "* traceback module does not have format_exception_only function: "
           "cannot acquire exception info\n";
  }

  PyObject *ExcStrs =
      PyObject_CallFunctionObjArgs(format_exception_only, type, value, NULL);
  Py_DecRef(format_exception_only);

  if (!ExcStrs || !PyList_Check(ExcStrs)) {
    if (ExcStrs)
      Py_DecRef(ExcStrs);
    return "* cannot acquire exception info\n";
  }

  std::string ErrMsg;

  Py_ssize_t siz = PyList_Size(ExcStrs);

  for (int i = 0; i < siz; i++) {
    PyObject *cur = PyList_GetItem(ExcStrs, i);
    if (!cur)
      ErrMsg += "* cannot get exception info string\n";
    else
      ErrMsg += std::string{} + "* " + ASCIICapsule::FromObject(cur, false).Data();
  }

  Py_DecRef(ExcStrs);
  return ErrMsg;
}

static std::string AmiPyPythonCallstackToString(PyObject *traceback,
                                             PyObject *tracebackModule) {
  assert(traceback);
  assert(tracebackModule);

  if (!traceback || !tracebackModule)
    return "";

  std::string TbMsg = "* callstack:\n";

  PyObject *extract_tb = PyObject_GetAttrString(tracebackModule, "extract_tb");

  if (!extract_tb || !PyCallable_Check(extract_tb)) {
    if (extract_tb)
      Py_DecRef(extract_tb);

    return TbMsg + "* * traceback module does not have extract_tb function: "
                   "cannot acquire traceback info\n";
  }

  PyObject *tracebackList =
      PyObject_CallFunctionObjArgs(extract_tb, traceback, NULL);
  PyObject *FrameSummaryType =
      PyObject_GetAttrString(tracebackModule, "FrameSummary");
  Py_DecRef(extract_tb);

  if (!tracebackList || !PyList_Check(tracebackList) || !FrameSummaryType ||
      !PyType_Check(FrameSummaryType)) {
    if (tracebackList)
      Py_DecRef(tracebackList);
    if (FrameSummaryType)
      Py_DecRef(FrameSummaryType);

    return TbMsg + "* * could not acquire traceback list\n";
  }

  Py_ssize_t siz = PyList_Size(tracebackList);

  for (Py_ssize_t i = siz; i--;) {
    PyObject *cur = PyList_GetItem(tracebackList, i); // borrowed
    // type == traceback.FrameSummary

    if (cur &&
        PyType_IsSubtype(Py_TYPE(cur), (PyTypeObject *)FrameSummaryType)) {
      PyObject *filename = PyObject_GetAttrString(cur, "filename");
      PyObject *lineno = PyObject_GetAttrString(cur, "lineno");
      PyObject *name = PyObject_GetAttrString(cur, "name");
      PyObject *line = PyObject_GetAttrString(cur, "line");

      if (filename && PyUnicode_Check(filename) && lineno &&
          PyLong_Check(lineno) && name && PyUnicode_Check(name) && line &&
          PyUnicode_Check(line)) {
        TbMsg += std::format("* * File: '{}', line {:<3} (in {})\n",
                           ASCIICapsule::FromUnicode(filename, false).Data(),
                           PyLong_AsLong(lineno),
                           ASCIICapsule::FromUnicode(name, false).Data());

        if (line != Py_None)
          TbMsg += std::string{} + "* * * " +
                   ASCIICapsule::FromUnicode(line, false).Data() + "\n";
      } else
        TbMsg += "* * Invalid FrameSummary: filename/lineno/name/line is NULL "
                 "or is not expected type\n";

      if (filename)
        Py_DecRef(filename);
      if (lineno)
        Py_DecRef(lineno);
      if (name)
        Py_DecRef(name);
      if (line)
        Py_DecRef(line);
    } else
      TbMsg += "* * traceback item is not FrameSummary\n";
  }

  Py_DecRef(tracebackList);
  Py_DecRef(FrameSummaryType);

  return TbMsg;
}

std::string AmiPyPythonErrorToString() {
  if (!PyErr_Occurred())
    return "* Python error did not occur\n";

  gLogger("Converting python error to string", Logger::MSG_DEBUG);

  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  PyObject *TracebackStr = NULL, *tracebackModule = NULL;

  TracebackStr = PyUnicode_FromString("traceback");
  if (TracebackStr) {
    tracebackModule = PyImport_Import(TracebackStr);
    Py_DecRef(TracebackStr);
  }

  std::string ErrMsg;

  if (!tracebackModule)
    ErrMsg +=
        "* traceback module not avaiable: cannot acquire exception info\n";
  else {
    ErrMsg += AmiPyPythonExceptionOnlyToString(type, value, tracebackModule);

    if (traceback)
      ErrMsg += AmiPyPythonCallstackToString(traceback, tracebackModule);
    Py_DecRef(tracebackModule);
  }

  if (type)
    Py_DecRef(type);
  if (value)
    Py_DecRef(value);
  if (traceback)
    Py_DecRef(traceback);

  PyErr_Clear();

  return ErrMsg;
}

void AmiPyPrintError(const char *BaseMsg) {
  if (!PyErr_Occurred())
    return;

  gLogger("Printing Py Error", Logger::MSG_DEBUG);

  std::string ErrMsg = std::string{} + BaseMsg + '\n' + AmiPyPythonErrorToString();

  gLogger("ERROR: " + ErrMsg, Logger::MSG_ERROR);

#ifdef _WIN32
  OutputDebugStringA(ErrMsg);
#endif

  AmiPrintStr(ErrMsg.data());
  AmiError(ErrMsg.data());
}

// VA_ARGS call
AmiVar AmiPyEvalFunction(int iNumArgs, AmiVar *pArgs) {
  if (!EnsureInitialized())
    return AmiVar{VAR_FLOAT, EMPTY_VAL};

  if (iNumArgs < 2 || pArgs[0].type != VAR_STRING ||
      pArgs[1].type != VAR_STRING) {
    PRINT_ERROR("First argument is context name and second is function name");
    return AmiVar{VAR_FLOAT, EMPTY_VAL};
  }

  gLogger(std::string() + "Start PyEvalFunction( " + "\"" + pArgs[0].string +
              "\", " + "\"" + pArgs[1].string + "\", ... )",
          Logger::MSG_FN_START);

  iNumArgs = GetRealSize(iNumArgs, pArgs);

  AmiVar res;

  {
    GilGuard oGuard(pArgs[0].string);

    PyObject *pGlobals = AmiPyIsolation_GetDict(pArgs[0].string);
    if (!pGlobals)
      return AmiVar{VAR_FLOAT, EMPTY_VAL};

    // Find function
    PyObject *pFunction =
        PyDict_GetItemString(pGlobals, pArgs[1].string); // Borrowed ref.

    if (pFunction == NULL || !PyCallable_Check(pFunction)) {
      PRINT_ERROR("'%s' does not exist in global scope OR is not callable (not "
                  "a function)",
                  pArgs[1].string);
      return AmiVar{VAR_FLOAT, EMPTY_VAL};
    }

    PyObject *pPyArgs = PyTuple_New(iNumArgs - 2); // New ref.

    for (int j = 0; j < iNumArgs - 2; j++) {
      PyObject *pObj = AmiVarToObj(pArgs[j + 2]);

      if (!pObj) {
        // error converting value (PythonError)
        if (PyErr_Occurred())
          AmiPyPrintError(
              "Error occured during AFL to Py argument conversion:");

        Py_DecRef(pPyArgs);
        return AmiVar{VAR_FLOAT, EMPTY_VAL};
      }

      PyTuple_SetItem(pPyArgs, j, pObj); // Steals ref.
    }

    gLogger("Start Call Object", Logger::MSG_DEBUG);

    PyObject *pRes = PyObject_Call(pFunction, pPyArgs, nullptr); // New ref.

    Py_DecRef(pPyArgs);

    gLogger("End Call Object", Logger::MSG_DEBUG);

    if (PyErr_Occurred())
      AmiPyPrintError("Error occured during function call:");

    res = ObjToAmiVar(pRes);
    if (pRes)
      Py_DecRef(pRes);
  }

  gLogger("End PyEvalFunction", Logger::MSG_FN_END);

  return res;
}

AmiVar AmiPyLoadFromFile(int iNumArgs, AmiVar *pArgs) {
  if (!EnsureInitialized())
    return AmiVar{VAR_FLOAT, EMPTY_VAL};

  if (iNumArgs != 2 || pArgs[0].type != VAR_STRING ||
      pArgs[1].type != VAR_STRING) {
    PRINT_ERROR("First argument is context name and second is file name");
    return AmiVar{VAR_FLOAT, EMPTY_VAL};
  }

  gLogger(std::string{} + "Start PyLoadFromFile( " + "\"" + pArgs[0].string +
              "\", " + "\"" + pArgs[1].string + "\" )",
          Logger::MSG_FN_START);

  FILE *fh = fopen(pArgs[1].string, "rb");

  if (!fh) {
    PRINT_ERROR("Cannot open '%s'", pArgs[1].string);
    return AmiVar{VAR_FLOAT, EMPTY_VAL};
  }

  {
    GilGuard oGuard(pArgs[0].string);

    PyObject *pGlobals = AmiPyIsolation_GetDict(pArgs[0].string);
    if (!pGlobals)
      return AmiVar{VAR_FLOAT, EMPTY_VAL};

    // PyImport_AddModule( "__main__" );
    gLogger("Start Run File", Logger::MSG_DEBUG);

    AmiPyRun_File(fh, pArgs[1].string, pGlobals, true);

    gLogger("End Run File", Logger::MSG_DEBUG);

    if (PyErr_Occurred())
      AmiPyPrintError("Error occured during Python execution:");
  }

  gLogger("End PyLoadFromFile", Logger::MSG_FN_END);

  return AmiVar{VAR_FLOAT, EMPTY_VAL};
}
