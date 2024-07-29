/**
 ******************************************************************************
 * @file           : progress_bar.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-28
 ******************************************************************************
 */

#include "progress_bar.h"

namespace Ffc {
void ProgressBar::setMaxProgress(const int64_t &max) {
    int64_t setMax = max;
    if (max < 0) {
        setMax = 0;
    } else if (max > 100) {
        setMax = 100;
    }
    bar.set_option(option::MaxProgress{setMax});
}

void ProgressBar::setPrefixText(const std::string &text) {
    bar.set_option(option::PrefixText{text});
}

void ProgressBar::setPostfixText(const std::string &text) {
    bar.set_option(option::PostfixText{text});
}

void ProgressBar::setProgress(const int &progress) {
    int setProgress = progress;
    if (progress < 0) {
        setProgress = 0;
    } else if (progress > 100) {
        setProgress = 100;
    }
    bar.set_progress(setProgress);
}

void ProgressBar::tick() {
    bar.tick();
}

void ProgressBar::markAsCompleted() {
    bar.mark_as_completed();
}

void ProgressBar::showConsoleCursor(bool show) {
    show_console_cursor(show);
}
} // namespace Ffc