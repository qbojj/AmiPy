#include "Logger.h"

#include "AmiPyConversions.h"
#include "AmiVar.h"

#include "inih.h"

#include <chrono>
#include <mutex>
#include <ratio>
#include <string>
#include <format>
#include <cstdio>
#include <thread>

Logger gLogger;

TimeMeasurer::TimeMeasurer(bool init) {
  if (init)
    start();
}

void TimeMeasurer::start() { m_oStartTime = clock::now(); }

std::uint64_t TimeMeasurer::getTimeDiff() {
  auto now = clock::now();
  auto diff = now - m_oStartTime;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(diff);
  return static_cast<std::uint64_t>(micros.count());
}

std::string TimeMeasurer::getAsString() {
  auto Elapsed = getTimeDiff();

  return std::format("{}.{:03} ms", Elapsed / 1000, Elapsed % 1000);
}

Logger::Logger() : m_oTM(false) { m_pLogFile = NULL; }

Logger::~Logger() {
  if (m_pLogFile)
    std::fclose(m_pLogFile);
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

bool Logger::_Initialize() {
  if (m_pLogFile)
    return true;

  bool bDefaultConfig = false;

  INIReader ini(INI_PATH);
  if (ini.ParseError() == -1) {
    bDefaultConfig = true;
    FILE *f = std::fopen(INI_PATH, "w");

    if (f) {
      std::fputs(DefaultINIfile, f);
      std::fclose(f);
    }

    ini = INIReader(INI_PATH);
  }

  if (ini.ParseError() == -1) {
#ifdef _WIN32
    OutputDebugStringA("could not open or create config file (" INI_PATH ")\n");
#endif
    AmiError("could not open or create config file (" INI_PATH ")\n");
    return false;
  }

  if (ini.ParseError() > 0) {
    std::string res;
    res = "ini file parse error in line " + std::to_string(ini.ParseError()) + "\n";
#ifdef _WIN32
    OutputDebugStringA(res.data());
#endif
    AmiError(res.data());
  }

  std::string logPath = (ini.ParseError() >= 0)
                            ? ini.Get("Logging", "Path", DEFAULT_LOG_PATH)
                            : DEFAULT_LOG_PATH;

  m_pLogFile = fopen(logPath.c_str(), "w");

  if (!m_pLogFile) {
    std::string msg = "could not open log file (" + logPath + ")\n";
#ifdef _WIN32
    OutputDebugStringA(msg.data());
#endif
    AmiError(msg.data());
    return false;
  }

  std::string msgMaskStr;

  m_iMsgMask =
      (ini.GetBoolean("Logging", "error", true) ? MSG_ERROR : 0) |
      (ini.GetBoolean("Logging", "warning", true) ? MSG_WARNING : 0) |
      (ini.GetBoolean("Logging", "fn_start", false) ? MSG_FN_START : 0) |
      (ini.GetBoolean("Logging", "fn_end", false) ? MSG_FN_END : 0) |
      (ini.GetBoolean("Logging", "main_ev", true) ? MSG_MAIN_EVENTS : 0) |
      (ini.GetBoolean("Logging", "debug", false) ? MSG_DEBUG : 0);

  m_bFlush = true; // ini.GetBoolean("Logging", "flush", true);

  msgMaskStr =
      std::string() +
      (ini.GetBoolean("Logging", "error", true) ? "error " : "") +
      (ini.GetBoolean("Logging", "warning", true) ? "warning " : "") +
      (ini.GetBoolean("Logging", "fn_start", false) ? "fn_start " : "") +
      (ini.GetBoolean("Logging", "fn_end", false) ? "fn_end " : "") +
      (ini.GetBoolean("Logging", "main_ev", true) ? "main_ev " : "") +
      (ini.GetBoolean("Logging", "debug", false) ? "debug " : "");

  PluginInfo PI;
  GetPluginInfo(&PI);

  auto startTime = std::chrono::system_clock::now();
  std::string StartTime = std::format("{:%Y-%m-%d %H:%M:%S}", startTime);

  AmiVar args[1] = {SetVal(0)};

  AmiVar AmiVer = g_bIsClosing ? AmiVar{VAR_NONE, 0}
                               : gSite.CallFunction("Version", 1, args);
  float fAmiVer = 0;
  if (AmiVer.type == VAR_FLOAT)
    fAmiVer = AmiVer.val;
  FreeAmiVar(AmiVer);

  std::fprintf(m_pLogFile,
          "AmiPy %d.%d.%d log:\n"
          "\tAB: %.2f\n"
          "\tPY: %s\n"
          "\tNp: compiled " NPY_FEATURE_VERSION_STRING "\n"
          "\tLogging message mask: %s\n"
          "%s"
          "Logging started %s\n",
          (PI.nVersion / 10000) % 100, (PI.nVersion / 100) % 100,
          PI.nVersion % 100, fAmiVer, Py_GetVersion(), msgMaskStr.c_str(),
          !m_bFlush ? "\tLog file is NOT flushed: log file can be incomplete\n"
                    : "",
          StartTime.data());

  if (ini.ParseError()) {
    std::string err;
    switch (ini.ParseError()) {
    case -1:
      err = "can't open ini file (" INI_PATH ")";
      break;
    case -2:
      err = "cannot allocate line buffer";
      break;
    default:
      err = "error in line " + std::to_string(ini.ParseError());
    };

    std::fprintf(m_pLogFile, "Ini file parse error: %d - %s\n", ini.ParseError(),
            err.c_str());
  } else if (bDefaultConfig) {
    std::fprintf(m_pLogFile,
            "Ini file wasn't present: using default configuration\n");
  }

  std::fflush(m_pLogFile);

  m_oTM.start();

  return true;
}

bool Logger::Initialize() {
  std::unique_lock<std::timed_mutex> lock(m_oLoggerCS, std::defer_lock);
  if (m_pLogFile)
    return true;

  bool ok = false;

  if (lock.try_lock_for(std::chrono::milliseconds(500))) {
    ok = _Initialize();
  } else {
#ifdef _WIN32
    OutputDebugStringA("could not lock Logger during initialization\n");
#endif
    AmiError("could not lock Logger during initialization");
  }

  return ok;
}

void Logger::PushMessage(std::string_view message, int type) {
  if ((type & m_iMsgMask) != type)
    return; // pass only when all types are in MsgMask
  if (!m_pLogFile && !Initialize())
    return;

  auto Elapsed = m_oTM.getTimeDiff();

  auto tid = 1;//std::this_thread::get_id();

  std::string msg;
  msg = std::format("{}.{:02} ms (TID: {}) : {}\n", Elapsed / 1000, (Elapsed / 10) % 100, tid, message);

  std::unique_lock<std::timed_mutex> lock(m_oLoggerCS, std::defer_lock);

  if (lock.try_lock_for(std::chrono::milliseconds(100))) {
    std::fputs(msg.c_str(), m_pLogFile);
    lock.unlock();

    if (m_bFlush)
      std::fflush(m_pLogFile);
  } else {
#ifdef _WIN32
    OutputDebugStringA("could not lock Logger\n");
#endif
    std::fputs("could not lock Logger\n", stderr);
    AmiError("could not lock Logger");
  }
}
