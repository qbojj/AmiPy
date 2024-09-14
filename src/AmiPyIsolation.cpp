#include "AmiPyIsolation.h"

#include "AmiPyConversions.h"
#include "Logger.h"

#include <Python.h>

#include <chrono>
#include <unordered_map>
#include <string>
#include <mutex>
#include <string_view>
#include <atomic>

typedef std::unordered_map<std::string, PyObject *> CPyObjectMap;

static CPyObjectMap oGlobalsMap;
static std::atomic<bool> bInitialzied = false;
static std::atomic<bool> bFinalizing = false;

static std::timed_mutex oIsolationCS;

PyObject *AmiPyIsolation_GetDict(const char *ctx) {
  std::string sKey = ctx;

  std::unique_lock<std::timed_mutex> lock(oIsolationCS, std::defer_lock);

  PyObject *pDict = NULL;

  if (bFinalizing)
    return NULL;

  if (!lock.try_lock_for(std::chrono::seconds(1))) {
    PRINT_ERROR("could not lock isolation environment");
    return NULL;
  }

  if (bFinalizing) {
    return NULL;
  }

  if (auto el = oGlobalsMap.find(sKey); el != oGlobalsMap.end()) {
    gLogger("found dict (key=\"" + sKey + "\")", Logger::MSG_DEBUG);
    return el->second;
  }
  
  gLogger("creating new dict (key=\"" + sKey + "\")", Logger::MSG_DEBUG);

  pDict = PyDict_New();

  if (!pDict) {
    PRINT_ERROR("could not create new dict for new context");
    return nullptr;
  }

  PyObject *builtins = PyEval_GetBuiltins();
  if (!builtins ||
      PyDict_SetItemString(pDict, "__builtins__", builtins) < 0) {
    PRINT_ERROR("could not set __builtins__ in new context");
    Py_DecRef(pDict);
    return nullptr;
  }

  oGlobalsMap[sKey] = pDict;

  return pDict;
}

void AmiPyIsolation_ClearAll() {
  std::unique_lock<std::timed_mutex> lock(oIsolationCS, std::defer_lock);

  if (!lock.try_lock_for(std::chrono::seconds(5))) // wait max 5s
  {
    PRINT_ERROR("could not lock isolation environment during destruction");
    return;
  }

  bFinalizing = true;

  if (!bInitialzied) {
    return;
  }
   
  gLogger("releasing all dicts", Logger::MSG_DEBUG);
  TimeMeasurer t;

  for (auto &[sKey, pDict] : oGlobalsMap) {
    Py_DecRef(pDict);
  }

  oGlobalsMap.clear();

  gLogger("all dicts released (" + t.getAsString() + ")",
          Logger::MSG_MAIN_EVENTS);
}

GilGuard::GilGuard(const char *ctx) {
  gLogger("Get GIL", Logger::MSG_DEBUG);
  m_oGILState = PyGILState_Ensure();
  gLogger("GIL Set", Logger::MSG_DEBUG);
}

GilGuard::~GilGuard() {
  gLogger("Release GIL", Logger::MSG_DEBUG);
  PyGILState_Release(m_oGILState);
  gLogger("GIL Released", Logger::MSG_DEBUG);
}
