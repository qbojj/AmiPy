#include "StdAfx.h"
#include "AmiPyPython.h"

#include "AmiPyConversions.h"
#include "AmiPyFunctions.h"
#include "AmiVar.h"

typedef struct AmiPyInterpreterLocalData AmiPyInterpreterLocalData;

static PyObject *FormatTuple( PyObject *tuple )
{
	Py_ssize_t siz = PyTuple_Size( tuple );
	PyObject *tuple2 = PyTuple_New( siz );
	if( !tuple2 ) return NULL;

	for( int i = 0; i < siz; i++ )
	{
		PyObject *cur, *NewI = NULL;

		cur = PyTuple_GetItem( tuple, i );
		if( cur ) NewI = PyObject_Str( cur );

		if( !NewI )
		{
			Py_DecRef( tuple2 );
			return NULL;
		}

		if( PyTuple_SetItem( tuple2, i, NewI ) < 0 )
		{
			Py_DecRef( tuple2 );
			Py_DecRef( NewI );
			return NULL;
		}
	}

	PyObject *separator = PyUnicode_FromString( " " );
	if( !separator )
	{
		Py_DecRef( tuple2 );
		return NULL;
	}

	PyObject *joined = PyUnicode_Join( separator, tuple2 );

	Py_DecRef( separator );
	Py_DecRef( tuple2 );

	return joined;
}

static PyObject *AmiPyPython_GenericPrinter( PyObject *pArgs, void(*printer)(const char *) )
{
	assert( printer != NULL );
	assert( pArgs != NULL );

	if( !PyTuple_Check( pArgs ) )
	{
		PyErr_SetString( PyExc_TypeError, "functions second argument is not tuple" );
		return NULL;
	}

	PyObject *str = FormatTuple( pArgs );

	if( !str )
	{
		//PyErr_SetString( PyExc_TypeError, "cannot convert args to string" );
		return NULL;
	}

	ASCIICapsule data = ASCIICapsule::FromUnicode( str, true );
	if( !data ) return NULL;

	printer( data.Data() );

	return Py_IncRef(Py_None), Py_None;
}

static PyObject *
AmiPyPython_Print( PyObject *, PyObject *pArgs ) { return AmiPyPython_GenericPrinter( pArgs, AmiPrintStr ); }

static PyObject *
AmiPyPython_Error( PyObject *, PyObject *pArgs ) { return AmiPyPython_GenericPrinter( pArgs, AmiError ); }

static PyMethodDef AmiPyMethods[] = 
{
	{"Print", AmiPyPython_Print, METH_VARARGS, 
	PyDoc_STR("Print(...)\n\n"
			  "Print string to AFL debugger.")},

	{"Error", AmiPyPython_Error, METH_VARARGS, 
	PyDoc_STR("Error(...)\n\n"
			  "Print error string in AFL.")},

	{NULL, NULL, 0, NULL}
};

static PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"AmiPy", "AmiBroker plug-in python module",
	-1,
	AmiPyMethods
};

PyObject *PyInit_AmiPy( void )
{
	PyObject *pMod = PyModule_Create( &cModPyDem );

	return pMod;
}