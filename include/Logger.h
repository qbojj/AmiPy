#pragma once

#include <AmiVar.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
#include <mutex>
#include <string>
#include <cstdint>
#include <mutex>
#include <string_view>

class TimeMeasurer {
public:
  using clock = std::chrono::steady_clock;

  TimeMeasurer(bool init = true);
  void start();

  // in us
  std::uint64_t getTimeDiff();

  // in ms ( 0.01 ms )
  std::string getAsString();

protected:
  clock::time_point m_oStartTime;
};

class Logger {
public:
  enum {
    MSG_ERROR = 1 << 0,
    MSG_WARNING = 1 << 1,     // w
    MSG_FN_START = 1 << 2,    // s
    MSG_FN_END = 1 << 3,      // e
    MSG_MAIN_EVENTS = 1 << 4, // m
    MSG_DEBUG = 1 << 5,       // d
  };

  Logger();
  ~Logger();

  inline void operator()(std::string_view message, int type) {
    PushMessage(message, type);
  };
  void PushMessage(std::string_view message, int type);

  bool Initialize();

protected:
  FILE *m_pLogFile;
  TimeMeasurer m_oTM;

  bool _Initialize();
  std::timed_mutex m_oLoggerCS;

  int m_iMsgMask = MSG_ERROR | MSG_WARNING | MSG_MAIN_EVENTS | MSG_FN_START;
  bool m_bFlush = true;
};

extern Logger gLogger;

inline std::string QUIET_ERROR(auto &&... args) {
  auto len = std::snprintf(nullptr, 0, std::forward<decltype(args)>(args)...);
  std::string message;
  message.resize(len);
  std::sprintf(message.data(), std::forward<decltype(args)>(args)...);
#ifdef _MSVC_VER
  OutputDebugStringA(message + "\n");
#else
  std::fprintf(stderr, "%s\n", message.data());
#endif
  return message; 
}

inline void WARNING(auto &&... args) {
  auto message = QUIET_ERROR(std::forward<decltype(args)>(args)...);
  gLogger("Warning: " + message, Logger::MSG_WARNING);
}

inline void PRINT_ERROR(auto &&... args) {
  auto message = QUIET_ERROR(std::forward<decltype(args)>(args)...);
  gLogger("Error: " + message, Logger::MSG_ERROR);
  AmiError(message.data());
}
