#include "StdAfx.h"

#include "AmiPyRun.h"

#include "AmiPyConversions.h"

#undef _Py_static_string_init
#define _Py_static_string_init(name)                                           \
  { 0, name, 0 }

#include <string>
using namespace std;

string GetFileContent(FILE *fh) {
  fseek(fh, 0L, SEEK_END);
  size_t siz = ftell(fh);
  fseek(fh, 0L, SEEK_SET);

  if (siz == 0)
    return "";

  string res(siz + 1, '\0');
  fread(&res[0], 1, siz, fh);

  return res;
}

int AmiPyRun_File(FILE *fh, const char *fileName, PyObject *dict,
                  bool closeit) {
  ASSERT(fh);
  ASSERT(fileName != NULL);
  ASSERT(dict != NULL);

  PyObject *res, *code, *f;
  string content;

  content = GetFileContent(fh);
  if (closeit)
    fclose(fh);

  code = Py_CompileString(content.c_str(), fileName, Py_file_input);
  if (!code)
    return -1;

  bool set_file_name = false, set_cached = false;
  int ret = -1;
  if (PyDict_GetItemString(dict, "__file__") == NULL) {
    f = PyUnicode_DecodeFSDefault(fileName);
    if (f == NULL)
      goto done;

    if (PyDict_SetItemString(dict, "__file__", f) < 0) {
      // PRINT_ERROR( "Couldnot set python __file__" );
      Py_DecRef(f);
      goto done;
    }

    Py_DecRef(f);
    set_file_name = true;
  }

  if (PyDict_GetItemString(dict, "__cached__") == NULL) {
    if (PyDict_SetItemString(dict, "__cached__", Py_None) < 0) {
      // PRINT_ERROR( "Couldnot set python __cached__" );
      goto done;
    }

    set_cached = true;
  }

  res = PyEval_EvalCode(code, dict, dict);
  if (res)
    Py_DecRef(res);

  ret = 0;
done:
  if (set_file_name)
    PyDict_DelItemString(dict, "__file__");
  if (set_cached)
    PyDict_DelItemString(dict, "__cached__");

  return ret;
}