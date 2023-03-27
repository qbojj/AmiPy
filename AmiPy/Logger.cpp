#include "StdAfx.h"
#include "Logger.h"

#include "AmiVar.h"
#include "AmiPyConversions.h"

#include "inih.h"

Logger gLogger;

LARGE_INTEGER TimeMeasurer::m_oTimerFreq = { { 0, 0 } };

TimeMeasurer::TimeMeasurer( bool init )
{
	if( m_oTimerFreq.QuadPart == 0 )
		QueryPerformanceFrequency( &m_oTimerFreq );

	if( init ) start();
}

void TimeMeasurer::start()
{
	QueryPerformanceCounter( &m_oStartTime );
}

QWORD TimeMeasurer::getTimeDiff()
{
	LARGE_INTEGER t;
	QueryPerformanceCounter( &t );

	return (t.QuadPart - m_oStartTime.QuadPart) * 1000000 / m_oTimerFreq.QuadPart;
}

CStringA TimeMeasurer::getAsString()
{
	QWORD Elapsed = getTimeDiff();

	CStringA Ret;

	Ret.Format( "%llu.%3.3llu ms", Elapsed / 1000, Elapsed % 1000 );

	return Ret;
}

Logger::Logger() : m_oTM( false )
{
	m_pLogFile = NULL;
}

Logger::~Logger()
{
	if( m_pLogFile ) fclose( m_pLogFile );
}

const char DefaultINIfile[] = R"INI(
[Logging]
Path=AmiPy.log

error=true
warning=true
fn_start=false
fn_end=false
main_ev=true
debug=false
)INI";

bool Logger::_Initialize()
{
	if( m_pLogFile ) return true;

	bool bDefaultConfig = false;

	INIReader ini( INI_PATH );
	if( ini.ParseError() == -1 )
	{
		bDefaultConfig = true;
		FILE *f = fopen( INI_PATH, "w" );

		if( f )
		{
			fputs( DefaultINIfile, f );
			fclose( f );
		}

		ini = INIReader( INI_PATH );
	}

	if( ini.ParseError() == -1 )
	{
		OutputDebugStringA( "could not open or create config file (" INI_PATH ")\n" );
		AmiError( "could not open or create config file (" INI_PATH ")\n" );
		return false;
	}

	if( ini.ParseError() > 0 )
	{
		CStringA res;
		res.Format( "ini file parse error in line %d\n", ini.ParseError() );
		OutputDebugStringA( res );
		AmiError( res );
	}

	std::string logPath = ( ini.ParseError() >= 0 ) ? ini.Get( "Logging", "Path", DEFAULT_LOG_PATH ) : DEFAULT_LOG_PATH;

	m_pLogFile = fopen( logPath.c_str(), "w" );

	if( !m_pLogFile )
	{
		std::string msg = "could not open log file (" + logPath + ")\n";
		OutputDebugStringA( msg.c_str() );
		AmiError( msg.c_str() );
		return false;
	}

	std::string msgMaskStr;

	m_iMsgMask =
		(ini.GetBoolean( "Logging", "error", true ) ? MSG_ERROR : 0) |
		(ini.GetBoolean( "Logging", "warning", true ) ? MSG_WARNING : 0) |
		(ini.GetBoolean( "Logging", "fn_start", false ) ? MSG_FN_START : 0) |
		(ini.GetBoolean( "Logging", "fn_end", false ) ? MSG_FN_END : 0) |
		(ini.GetBoolean( "Logging", "main_ev", true ) ? MSG_MAIN_EVENTS : 0) |
		(ini.GetBoolean( "Logging", "debug", false ) ? MSG_DEBUG : 0);

	m_bFlush = true;//ini.GetBoolean("Logging", "flush", true);

	msgMaskStr =
		std::string() +
		(ini.GetBoolean( "Logging", "error", true ) ? "error " : "") +
		(ini.GetBoolean( "Logging", "warning", true ) ? "warning " : "") +
		(ini.GetBoolean( "Logging", "fn_start", false ) ? "fn_start " : "") +
		(ini.GetBoolean( "Logging", "fn_end", false ) ? "fn_end " : "") +
		(ini.GetBoolean( "Logging", "main_ev", true ) ? "main_ev " : "") +
		(ini.GetBoolean( "Logging", "debug", false ) ? "debug " : "");

	PluginInfo PI;
	GetPluginInfo( &PI );

	CTime StartTime = CTime::GetTickCount();

	AmiVar args[1] = { SetVal( 0 ) };

	AmiVar AmiVer = g_bIsClosing ? AmiVar{ VAR_NONE, 0 } : gSite.CallFunction("Version", 1, args);
	float fAmiVer = 0;
	if (AmiVer.type == VAR_FLOAT) fAmiVer = AmiVer.val;
	FreeAmiVar(AmiVer);

	fprintf(m_pLogFile,
		"AmiPy %d.%d.%d log:\n"
		"\tAB: %.2f\n"
		"\tPY: %s\n"
		"\tLogging message mask: %s\n"
		"%s"
		"Logging started %s\n",
		( PI.nVersion / 10000 ) % 100, ( PI.nVersion / 100 ) % 100, PI.nVersion % 100,
		fAmiVer,
		Py_GetVersion(),
		msgMaskStr.c_str(),
		!m_bFlush ? "\tLog file is NOT flushed: log file can be incomplete\n" : "",
		(LPCSTR)StartTime.Format( "%Y-%m-%d %H:%M:%S" ) );

	if( ini.ParseError() )
	{
		std::string err;
		switch(ini.ParseError())
		{
			case -1: err = "can't open ini file (" INI_PATH ")"; break;
			case -2: err = "cannot allocate line buffer"; break;
			default:
				err = "error in line " + std::to_string( ini.ParseError() );
		};

		fprintf( m_pLogFile,
			"Ini file parse error: %d - %s\n",
			ini.ParseError(), err.c_str() );
	}
	else if( bDefaultConfig )
	{
		fprintf( m_pLogFile, "Ini file wasn't present: using default configuration\n" );
	}

	fflush(m_pLogFile);

	m_oTM.start();

	return true;
}

bool Logger::Initialize()
{
	if( m_pLogFile ) return true;

	CSingleLock lock( &m_oLoggerCS );
	bool ok = false;

	if( lock.Lock( 500 ) )
	{
		ok = _Initialize();
		lock.Unlock();
	}
	else OutputDebugStringA( "could not lock Logger during initialization\n" ),
		AmiError( "could not lock Logger during initialization" );

	return ok;
}

void Logger::PushMessage( const char *message, int type )
{
	if( (type & m_iMsgMask) != type ) return; // pass only when all types are in MsgMask
	if( !m_pLogFile && !Initialize() ) return;

	QWORD Elapsed = m_oTM.getTimeDiff();

	CStringA msg;
	msg.Format( "%8lld.%2.2lld ms (TID: %8.8x) : %s\n",
		Elapsed / 1000,
		(Elapsed / 10) % 100,
		GetCurrentThreadId(),
		message );

	CSingleLock lock( &m_oLoggerCS );

	if( lock.Lock(100) ) // make sure that there is no race
	{
		fputs( msg, m_pLogFile );
		lock.Unlock();

		if( m_bFlush ) fflush( m_pLogFile );
	}
	else OutputDebugStringA( "could not lock Logger\n" ),
		 fputs( "could not lock Logger\n", stderr ),
		 AmiError( "could not lock Logger" );
}