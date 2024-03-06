#include "StdAfx.h"

#include "AmiPyIsolation.h"

#include "AmiPyConversions.h"

typedef CMap<CStringA, LPCSTR, PyObject *, PyObject *> CPyObjectMap;

static CPyObjectMap oGlobalsMap;
static bool bInitialzied = false;
static bool bFinalizing = false;

static CCriticalSection oIsolationCS;

PyObject *AmiPyIsolation_GetDict(const char *ctx) {
  CStringA sKey = ctx;

  CSingleLock lock(&oIsolationCS);

  PyObject *pDict = NULL;

  if (bFinalizing)
    return NULL;

  if (lock.Lock(1000)) {
    if (bFinalizing) {
      lock.Unlock();
      return NULL;
    }

    if (!bInitialzied) {
      gLogger("Initializing dict map", Logger::MSG_MAIN_EVENTS);

      oGlobalsMap.InitHashTable(1013);
      bInitialzied = true;
    }

    BOOL exists = oGlobalsMap.Lookup(sKey, pDict);

    if (!exists) {
      gLogger("creating new dict (key=\"" + sKey + "\")", Logger::MSG_DEBUG);

      pDict = PyDict_New();

      if (!pDict) {
        PRINT_ERROR("could not create new dict for new context");
        goto cleanUp;
      }

      PyObject *builtins = PyEval_GetBuiltins();
      if (!builtins ||
          PyDict_SetItemString(pDict, "__builtins__", builtins) < 0) {
        PRINT_ERROR("could not set __builtins__ in new context");
        Py_DecRef(pDict);
        goto cleanUp;
      }

      oGlobalsMap[sKey] = pDict;

      /*
      // create new interpreter
      TimeMeasurer t;

      PyThreadState *pMainTS = PyThreadState_New( g_pMainIS );
      PyEval_RestoreThread( pMainTS ); // take GIL

      // PyEval_RestoreThread( g_pMainTS );

      // create new interpreter (sets current thread state)
      PyThreadState *pTS = Py_NewInterpreter();

      oInterpreters[sKey] = pIS = pTS->interp;

      PyThreadState_Clear( pTS );
      PyThreadState_Swap( pMainTS );

      PyThreadState_Delete( pTS );

      PyThreadState_Clear( pMainTS );
      PyThreadState_DeleteCurrent(); // release GIL

      gLogger( "created interpreter (" + t.getAsString() + ")" );
      */
    } else
      gLogger("found dict (key=\"" + sKey + "\")", Logger::MSG_DEBUG);

  cleanUp:
    lock.Unlock();
  } else {
    PRINT_ERROR("could not lock isolation environment");
    return NULL;
  }

  return pDict;
}

void AmiPyIsolation_ClearAll() {
  CSingleLock lock(&oIsolationCS);

  if (lock.Lock(5000)) // wait max 5s
  {
    bFinalizing = true;

    if (bInitialzied) {
      gLogger("releasing all dicts", Logger::MSG_DEBUG);
      TimeMeasurer t;

      POSITION pos = oGlobalsMap.GetStartPosition();

      CStringA sKey;
      PyObject *pDict = NULL;

      while (pos) {
        oGlobalsMap.GetNextAssoc(pos, sKey, pDict);

        Py_DecRef(pDict);

        // TimeMeasurer t2;
        // gLogger( "closing interpreter: " + sKey );

        // gLogger( "interpreter closed (" + t2.getAsString() + ")" );
      }

      oGlobalsMap.RemoveAll();

      gLogger("all dicts released (" + t.getAsString() + ")",
              Logger::MSG_MAIN_EVENTS);
    }

    lock.Unlock();
  } else
    PRINT_ERROR("could not lock isolation environment during destruction");
}

GilGuard::GilGuard(const char *ctx) {
  (void)ctx;

  gLogger("Get GIL", Logger::MSG_DEBUG);

  // m_pThreadState = PyThreadState_New( g_pMainIS );
  // PyEval_RestoreThread( m_pThreadState );

  m_oGILState = PyGILState_Ensure();

  gLogger("GIL Set", Logger::MSG_DEBUG);
}

GilGuard::~GilGuard() {
  gLogger("Release GIL", Logger::MSG_DEBUG);
  // PyThreadState_Clear( m_pThreadState );
  // PyThreadState_DeleteCurrent();
  PyGILState_Release(m_oGILState);
  gLogger("GIL Released", Logger::MSG_DEBUG);
}

/*
void InterpreterCapsule::init( PyInterpreterState *pInterpreter )
{
        gLogger( "creating and setting thread state" );
        TimeMeasurer t;

        m_pThreadState = PyThreadState_New( pInterpreter );

        PyEval_RestoreThread( m_pThreadState );

        gLogger( "created and set thread state (" + t.getAsString() + ")" );
}

InterpreterCapsule::InterpreterCapsule( PyInterpreterState *pInterpreter )
{
        init( pInterpreter );
}

InterpreterCapsule::InterpreterCapsule( CStringA sKey )
{
        PyInterpreterState *pIS = AmiPyInterpreters_GetAt( sKey );
        init( pIS );
}

InterpreterCapsule::~InterpreterCapsule()
{
        gLogger( "releasing and deleting thread state" );
        TimeMeasurer t;

        PyThreadState_Clear( m_pThreadState );
        PyThreadState_DeleteCurrent(); // release GIL

        gLogger( "released and deleted thread state (" + t.getAsString() + ")"
);
}
*/