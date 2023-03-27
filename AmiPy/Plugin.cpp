////////////////////////////////////////////////////
// Plugin.cpp
// Standard implementation file for all AmiBroker plug-ins
//
///////////////////////////////////////////////////////////////////////
// Copyright (c) 2001-2009 AmiBroker.com. All rights reserved. 
//
// Users and possessors of this source code are hereby granted a nonexclusive, 
// royalty-free copyright license to use this code in individual and commercial software.
//
// AMIBROKER.COM MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE CODE FOR ANY PURPOSE. 
// IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
// AMIBROKER.COM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOURCE CODE, 
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
// IN NO EVENT SHALL AMIBROKER.COM BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR 
// CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, 
// ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.
// 
// Any use of this source code must include the above notice, 
// in the user documentation and internal comments to the code.
///////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Plugin.h"

#include "AmiPyFunctions.h"
// #include "AmiVar.h"
#include "AmiPyConversions.h"

#define AMIPY_PLUGIN_VERSION_MAJOR		0
#define AMIPY_PLUGIN_VERSION_MINOR		1	//	0..99
#define AMIPY_PLUGIN_VERSION_RELEASE	0	//	0..99

// These are the only two lines you need to change
#define PLUGIN_NAME "AmiPy: AmiBroker-Python integration"
#define VENDOR_NAME "Kuba"
#define PLUGIN_VERSION (AMIPY_PLUGIN_VERSION_MAJOR*10000 +	\
						AMIPY_PLUGIN_VERSION_MINOR * 100 +	\
						AMIPY_PLUGIN_VERSION_RELEASE )

////////////////////////////////////////
// Data section
////////////////////////////////////////
static struct PluginInfo oPluginInfo =
{
		sizeof( struct PluginInfo ),
		1,
		PLUGIN_VERSION,
		0,
		PLUGIN_NAME,
		VENDOR_NAME,
		0,
		63000//64000
};

// the site interface for callbacks
struct SiteInterface gSite;

float VarArgTable[-VA_ARGS_ARG_CNT];
FunctionTag gFunctionTable[] = {
	{"PyLoadFromFile( fileName )",						{PyLoadFromFile, 0, 1, 0, 0, NULL} },
	{"PyEvalFunction( PyfunctionName, ... [fn args] )",	{PyEvalFunction, 0, 1, 0, (UBYTE)VA_ARGS_ARG_CNT, VarArgTable} },
};
int gFunctionTableSize = sizeof( gFunctionTable ) / sizeof( *gFunctionTable );

///////////////////////////////////////////////////////////
// Basic plug-in interface functions exported by DLL
///////////////////////////////////////////////////////////

PLUGINAPI int GetPluginInfo( struct PluginInfo *pInfo )
{
	*pInfo = oPluginInfo;

	return TRUE;
}


PLUGINAPI int SetSiteInterface( struct SiteInterface *pInterface )
{
	gSite = *pInterface;

	return TRUE;
}


PLUGINAPI int GetFunctionTable( FunctionTag **ppFunctionTable )
{
	for( int i = 0; i < -VA_ARGS_ARG_CNT; i++ )
		VarArgTable[i] = VA_ARGS_EMPTY_VAL;

	*ppFunctionTable = gFunctionTable;

	// must return the number of functions in the table
	return gFunctionTableSize;
}

PLUGINAPI int Init( void )
{
	// _PRINT_FN_NOAPI( "Init" );
	return 1;
	// return 1; 	 // default implementation does nothing

};

PLUGINAPI int Release( void )
{
	// _PRINT_FN_NOAPI( "Release" );
	SaveClose();
	return 1;
	// return 1; 	  // default implementation does nothing
};