#pragma once

#include <Plugin_Extended.h>
#include <string>

std::string AmiPyPythonErrorToString();
void AmiPyPrintError(const char *BaseMsg);

AmiVar AmiPyLoadFromFile(int iNumArgs, AmiVar *pArgs);
AmiVar AmiPyEvalFunction(int iNumArgs, AmiVar *pArgs);

// AmiVar AmiPyChangeLoggingOptions( int iNumArgs, AmiVar *pArgs );
