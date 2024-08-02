/**
 ******************************************************************************
 * @file           : progress_bar.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-28
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROGRESS_BAR_H_
#define FACEFUSIONCPP_SRC_PROGRESS_BAR_H_

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

using namespace indicators;

namespace Ffc {

class ProgressBar {
public:
    ProgressBar() = default;
    ~ProgressBar() = default;
    
    void setMaxProgress(const int64_t &max);
    void setPrefixText(const std::string& text);
    void setPostfixText(const std::string& text);
    void setProgress(const int &progress);
    void tick();
    void markAsCompleted();
    static void showConsoleCursor(bool show);
    
private:
    indicators::ProgressBar bar{
        option::BarWidth{50},
        option::MaxProgress{static_cast<int64_t>(0)},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{""},
        option::ForegroundColor{Color::green},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::ShowPercentage{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROGRESS_BAR_H_
