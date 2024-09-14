#pragma once

#include <Python.h>

#include <stdio.h>

int AmiPyRun_File(FILE *fh, const char *fileName, PyObject *dict,
                  bool closeit = 0);
