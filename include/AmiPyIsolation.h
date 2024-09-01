#pragma once

// must hold GIL
void AmiPyIsolation_ClearAll();

PyObject *AmiPyIsolation_GetDict(const char *ctx);

class GilGuard {
public:
  GilGuard(const char *ctx);
  ~GilGuard();

protected:
  PyThreadState *m_pThreadState;
  PyGILState_STATE m_oGILState;
};