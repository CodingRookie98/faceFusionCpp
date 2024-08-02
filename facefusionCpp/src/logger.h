/**
 ******************************************************************************
 * @file           : logger.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-19
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_LOGGER_H_
#define FACEFUSIONCPP_SRC_LOGGER_H_

#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/async.h>

namespace Ffc {

class Logger {
public:
    enum LogLevel {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
        Critical
    };

    static std::shared_ptr<Logger> getInstance();

    void setLogLevel(const Logger::LogLevel &level);
    [[nodiscard]] LogLevel getLogLevel() const;
    void log(const Logger::LogLevel &level, const std::string &message) const;
    void trace(const std::string &message) const;
    void debug(const std::string &message) const;
    void info(const std::string &message) const;
    void warn(const std::string &message) const;
    void error(const std::string &message) const;
    void critical(const std::string &message) const;
    
    Logger();
    ~Logger() = default;
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(Logger &&) = delete;

private:
    std::shared_ptr<spdlog::logger> m_logger;
    LogLevel m_level;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_LOGGER_H_
