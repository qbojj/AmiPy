#pragma once
#include "AmiVar.h"
#include <atomic>

////////////////////////////////////////////////////////////////////////////////

// returns new reference
// returns NULL on Python error (if type not supported None is returned)
PyObject *AmiVarToObj(AmiVar variable);
AmiVar ObjToAmiVar(PyObject *variable);

////////////////////////////////////////////////////////////////////////////////

class ASCIICapsule {
public:
  ASCIICapsule();
  ASCIICapsule(PyObject *pASCIIObj, bool Steal);
  ASCIICapsule(const ASCIICapsule &);
  ~ASCIICapsule();

  const char *Data() const;
  operator const char *() const;

  static ASCIICapsule FromUnicode(PyObject *pUnicode, bool Steal);
  static ASCIICapsule FromObject(PyObject *pObj, bool Steal);

private:
  PyObject *m_pASCIIObj = NULL;
};

////////////////////////////////////////////////////////////////////////////////

// extern PyInterpreterState *g_pMainIS;
extern std::atomic_bool g_bIsClosing;

bool EnsureInitialized();
bool SaveClose();

void ForceClose();