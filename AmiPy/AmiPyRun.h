#pragma once

int AmiPyRun_File(
	FILE *fh,
	const char *fileName,
	PyObject *dict,
	bool closeit = 0 );
