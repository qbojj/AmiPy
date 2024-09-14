#pragma once
////////////////////////////////////////////////////
// Plugin.h
// Standard header file for all AmiBroker plug-ins
//
// Version 2.1a
///////////////////////////////////////////////////////////////////////
// Copyright (c) 2001-2010 AmiBroker.com. All rights reserved.
//
// Users and possessors of this source code are hereby granted a nonexclusive,
// royalty-free copyright license to use this code in individual and commercial
// software.
//
// AMIBROKER.COM MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
// CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
// WARRANTY OF ANY KIND. AMIBROKER.COM DISCLAIMS ALL WARRANTIES WITH REGARD TO
// THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL AMIBROKER.COM BE LIABLE
// FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
// DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
// CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.
//
// Any use of this source code must include the above notice,
// in the user documentation and internal comments to the code.
///////////////////////////////////////////////////////////////////////
//
// Version 1.0 : 2002-11-18 TJ
// Version 2.0 : 2009-07-30 TJ
//       added support for
//			* 64-bit date time format
//			* float volume/open int
//			* 2 user fields (Aux) in Quotation structure
//			* 100 new user fields (fundamentals) in StockInfo
//			* proper alignment for 64-bit platforms (8 byte
//boundary)
// Version 2.1 : 2010-03-31 TJ
//          * struct PluginNotification has new field
//            StockInfo * pCurrentSINew
//            and old field pCurrentSI type has changed to StockInfoFormat4 *
//
// Version 2.1a:
//          * added #pragma(pack) to ensure AmiVar struct packing compatible
//          with AmiBroker
//

////////////////////////////////////////////////////

#include <cstdint>

#ifndef PLUGIN_H
#define PLUGIN_H 1

// Possible types of plugins
// currently 4 types are defined
// PLUGIN_TYPE_AFL - for AFL function plugins
// PLUGIN_TYPE_DATA - for data feed plugins (requires 3.81 or higher)
// PLUGIN_TYPE_AFL_AND_DATA - for combined AFL/Data plugins
// PLUGIN_TYPE_OPTIMIZER - for optimization engine plugins (requires v5.12 or
// higher)

#define PLUGIN_TYPE_AFL 1
#define PLUGIN_TYPE_DATA 2
#define PLUGIN_TYPE_AFL_AND_DATA 3
#define PLUGIN_TYPE_OPTIMIZER 4

// all exportable functions must have undecorated names
#ifdef _MSC_VER
#ifdef _USRDLL
#define PLUGINAPI extern "C" __declspec(dllexport)
#else
#define PLUGINAPI extern "C" __declspec(dllimport)
#endif
#else
#define PLUGINAPI extern "C"
#endif

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
//
// Structures and functions
// COMMON for all kinds of AmiBroker plugins
//
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

// 64-bit date/time integer
#define DATE_TIME_INT std::uint64_t

// define signed and unsigned one byte types
typedef std::uint8_t UBYTE;
typedef std::int8_t SBYTE;

// useful macros for empty values
#define EMPTY_VAL (-1e10f)
#define IS_EMPTY(x) (x == EMPTY_VAL)
#define NOT_EMPTY(x) (x != EMPTY_VAL)

#define PIDCODE(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))

// PluginInfo structure holds
// general information about plugin
struct PluginInfo {
  int nStructSize; // this is sizeof( struct PluginInfo )
  int nType; // plug-in type currently 1 - indicator is the only one supported
  int nVersion; // plug-in version coded to int as MAJOR*10000 + MINOR * 100 +
                // RELEASE
  int nIDCode;  // ID code used to uniquely identify the data feed (set it to
                // zero for AFL plugins)
  char szName[64];    // long name of plug-in displayed in the Plugin dialog
  char szVendor[64];  // name of the plug-in vendor
  int nCertificate;   // certificate code - set it to zero for private plug-ins
  int nMinAmiVersion; // minimum required AmiBroker version (should be >= 380000
                      // -> AmiBroker 3.8)
};

// the list of AmiVar types
enum { VAR_NONE, VAR_FLOAT, VAR_ARRAY, VAR_STRING, VAR_DISP, VAR_MATRIX = 7 };

// undocumented Matrix AFL type
typedef struct Matrix {
  int rows;
  int cols;

#pragma warning(suppress : 4200)
  float data[];
} Matrix;

// AmiVar is a variant-like structure/union
// that holds any AFL value
// type member holds variable type (see VAR_ enum above)
#pragma pack(push, 2)
typedef struct AmiVar {
  int type;
  union {
    float val;
    float *array;
    char *string;
    void *disp;
    Matrix *matrix;
  };
} AmiVar;
#pragma pack(pop)

///////////////////////////////////////////////////
// COMMON EXPORTED FUNCTONS
//
// Each AmiBroker plug-in DLL must export the following
// functions:
// 1. GetPluginInfo	- called when DLL is loaded
// 2. Init - called when AFL engine is being initialized
// 3. Release - called when AFL engine is being closed

PLUGINAPI int GetPluginInfo(struct PluginInfo *pInfo);
PLUGINAPI int Init(void);
PLUGINAPI int Release(void);

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
//
// Structures and functions
// for AFL Plugins
//
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

// SiteInterface structure
// defines call-back function pointers
// the structure is filled with correct
// pointers by the AmiBroker and passed to DLL via SetSiteInterface() function
// call
//
// SiteInterface is used as a way to call-back AmiBroker built-in functions
//

struct SiteInterface {
  int nStructSize;
  int (*GetArraySize)(void);
  float *(*GetStockArray)(int nType);
  AmiVar (*GetVariable)(const char *pszName);
  void (*SetVariable)(const char *pszName, AmiVar newValue);
  AmiVar (*CallFunction)(const char *szName, int nNumArgs, AmiVar *ArgsTable);
  AmiVar (*AllocArrayResult)(void);
  void *(*Alloc)(unsigned int nSize);
  void (*Free)(void *pMemory);
  DATE_TIME_INT *(*GetDateTimeArray)(void); // new in 5.30
};

// FunDesc structure
// holds the pointer to actual user-defined function
// that can be called by AmiBroker.
// It holds also the number of array, string, float and default arguments
// for the function and the default values
//
typedef struct FunDesc {
  AmiVar (*Function)(int NumArgs, AmiVar *ArgsTable);
  UBYTE ArrayQty;       // number of Array arguments required
  UBYTE StringQty;      // number of String arguments required
  SBYTE FloatQty;       // number of float args
  UBYTE DefaultQty;     // number of default float args
  float *DefaultValues; // the pointer to defaults table
} FunDesc;

// FunctionTag struct
// holds the Name of the function and the corresponding
// FunDesc structure.
// This structure is used to define function table
// that is retrieved by AmiBroker via GetFunctionTable() call
// when AFL engine is initialized.
// That way new function names are added to the AFL symbol table
// and they become accessible.

typedef struct FunctionTag {
  const char *Name;
  FunDesc Descript;
} FunctionTag;

// Indicator plugin exported functions:
// 1. GetFunctionTable - called when AFL engine is being initialized
// 1. SetSiteInteface - called when AFL engine is being initialized
//
// Each function may be called multiple times.
//
// The order of calling functions during intialization is
// as follows:
//
// SetSiteInterface -> GetFunctionTable	-> Init ->
// ... normal work ....
// Release
//
// This cycle may repeat multiple times
//
// All functions in the plug in DLL use _cdecl calling convention
// (the default for C compiler)

PLUGINAPI int GetFunctionTable(FunctionTag **ppFunctionTable);
PLUGINAPI int SetSiteInterface(struct SiteInterface *pInterface);

////////////////////////////////////////
// Global-scope data for indicator plugins
////////////////////////////////////////

// FunctionTable should be defined
// in the implementation file of your functions
extern FunctionTag gFunctionTable[];
extern int gFunctionTableSize;

// Site interface is defined in Plugin.cpp
extern struct SiteInterface gSite;

#endif
