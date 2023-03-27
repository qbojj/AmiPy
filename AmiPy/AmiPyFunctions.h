#pragma once
#include "AmiVar.h"

/*
class AmiPyException : public CException
{
public:
	AmiPyException( const char *message ) : m_szMessage( message ) {};
	~AmiPyException() {};

	virtual BOOL GetErrorMessage(
		LPTSTR	lpszError,
		UINT	nMaxError,
		UINT	*pnHelpContext = NULL ) const;

protected:
	CStringA m_szMessage;
};*/

CStringA AmiPyPythonErrorToString();
void AmiPyPrintError(const char *BaseMsg);

AmiVar AmiPyLoadFromFile( int iNumArgs, AmiVar *pArgs );
AmiVar AmiPyEvalFunction( int iNumArgs, AmiVar *pArgs );

//AmiVar AmiPyChangeLoggingOptions( int iNumArgs, AmiVar *pArgs );