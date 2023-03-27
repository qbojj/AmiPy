#pragma once

/////////////////////////////////////////////////////////////////////////////////////
// enum TYPES { VALUE, ARRAY, TEXT, MATRIX };

void AmiPrintStr( const char * );
void AmiError( const char * );

inline AmiVar SetVal(float data);
inline AmiVar SetArr(const float *data);
inline AmiVar SetTxt(const char *data);
inline AmiVar SetMat(int rows, int cols, const float *data = NULL);

inline void FreeAmiVar(AmiVar);
inline void FreeAmiVarArray( int, AmiVar * );

inline int GetRealSize( int iNumArgs, AmiVar *pArgs );

//////////////////////////////////////////////////////////////////////////////////////
// implementations

inline AmiVar SetVal(float data)
{
	AmiVar v;
	v.type = VAR_FLOAT;
	v.val = data;
	return v;
}

inline AmiVar SetArr(const float* data)
{
	AmiVar v = gSite.AllocArrayResult();
	memcpy(v.array, data, gSite.GetArraySize() * sizeof(float));
	return v;
}

inline AmiVar SetTxt(const char* data)
{
	AmiVar v;
	v.type = VAR_STRING;
	int size = (int)strlen(data) + 1;
	v.string = (char*)gSite.Alloc(size * sizeof(char));
	memcpy(v.string, data, size);
	return v;
}

AmiVar SetMat(int rows, int cols, const float* data)
{
	AmiVar v;
	v.type = VAR_MATRIX;

	UINT DataSize = rows * cols * sizeof(float);
	Matrix* m = (Matrix*)gSite.Alloc(2 * sizeof(int) + DataSize);

	m->rows = rows;
	m->cols = cols;
	if (data) memcpy(m->data, data, DataSize);
	else memset(m->data, 0, DataSize);

	v.matrix = m;

	return v;
}

inline void FreeAmiVar(AmiVar v)
{
	if (v.type != VAR_FLOAT) gSite.Free(v.disp);
}

inline void FreeAmiVarArray(int s, AmiVar* a)
{
	for (int i = 0; i < s; i++)
		FreeAmiVar(a[i]);
}

inline int GetRealSize(int iNumArgs, AmiVar* pArgs)
{
	for (int i = 0; i < iNumArgs; i++)
		if (pArgs[i].type == VAR_FLOAT && pArgs[i].val == VA_ARGS_EMPTY_VAL)
			return i;

	return iNumArgs;
}