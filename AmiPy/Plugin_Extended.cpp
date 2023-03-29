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
#include "Plugin_Extended.h"

#include "AmiPyFunctions.h"
#include "AmiPyConversions.h"
#include "AmiVar.h"
#include "Logger.h"

#define AMIPY_PLUGIN_VERSION_MAJOR		1
#define AMIPY_PLUGIN_VERSION_MINOR		0	//	0..99
#define AMIPY_PLUGIN_VERSION_RELEASE	3	//	0..99

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
		63000
};

// the site interface for callbacks
struct SiteInterface gSite;

float VarArgTable[VA_ARGS_ARG_CNT];
FunctionTag gFunctionTable[] = {
	{"PyLoadFromFile( context, fileName )",						{AmiPyLoadFromFile, 0, 2, 0, 0, NULL} },
	{"PyEvalFunction( context, PyfunctionName, ... [fn args] )",{AmiPyEvalFunction, 0, 2, 0, (UBYTE)-VA_ARGS_ARG_CNT, VarArgTable} },
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
	for( int i = 0; i < VA_ARGS_ARG_CNT; i++ )
		VarArgTable[i] = VA_ARGS_EMPTY_VAL;

	*ppFunctionTable = gFunctionTable;

	// must return the number of functions in the table
	return gFunctionTableSize;
}

PLUGINAPI int Init( void )
{
	return 1;
};

PLUGINAPI int Release( void )
{
	g_bIsClosing = true;

	bool res = SaveClose();
	if( !res )
	{
		gLogger( "could not safely close python => closing forcefully\n", Logger::MSG_ERROR );
		ForceClose();
	}

	return 1;
};