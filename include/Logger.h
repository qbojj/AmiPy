#pragma once

#define DEFAULT_LOG_PATH "AmiPy.log"
#define INI_PATH "AmiPy.ini"

class TimeMeasurer {
public:
  TimeMeasurer(bool init = true);
  void start();

  // in us
  QWORD getTimeDiff();

  // in ms ( 0.01 ms )
  CStringA getAsString();

protected:
  LARGE_INTEGER m_oStartTime;
  static LARGE_INTEGER m_oTimerFreq;
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

  inline void operator()(const char *message, int type) {
    PushMessage(message, type);
  };
  void PushMessage(const char *message, int type);

  bool Initialize();

protected:
  FILE *m_pLogFile;
  TimeMeasurer m_oTM;

  bool _Initialize();
  CCriticalSection m_oLoggerCS;

  int m_iMsgMask = MSG_ERROR | MSG_WARNING | MSG_MAIN_EVENTS | MSG_FN_START;
  bool m_bFlush = true;
};

extern Logger gLogger;