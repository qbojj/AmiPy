#include "AmiVar.h"
#include "AmiPyConversions.h"


// printf from AmiBroker
static void AmiPrintf(int s, AmiVar *v) {
  const int NumOfPrintfArgs = 21;
  AmiVar PrintfArgs[NumOfPrintfArgs];

  for (int i = 0; i < s; i++)
    PrintfArgs[i] = v[i];
  for (int i = s; i < NumOfPrintfArgs; i++)
    PrintfArgs[i] = AmiVar{VAR_FLOAT, VA_ARGS_EMPTY_VAL};

  if (!g_bIsClosing)
    gSite.CallFunction("printf", NumOfPrintfArgs, PrintfArgs);
}

void AmiPrintStr(const char *str) {
  AmiVar dat[2] = {SetTxt("%s"), SetTxt(str)};

  AmiPrintf(2, dat);

  FreeAmiVarArray(2, dat);
}

void AmiError(const char *str) {
  AmiVar txt = SetTxt(str);

  if (!g_bIsClosing)
    gSite.CallFunction("Error", 1, &txt);

  FreeAmiVar(txt);
}
