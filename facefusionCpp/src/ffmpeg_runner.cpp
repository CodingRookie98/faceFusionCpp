/**
 ******************************************************************************
 * @file           : ffmpeg_runner.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-8-6
 ******************************************************************************
 */

#include <numeric>
#include "ffmpeg_runner.h"

namespace Ffc {

std::vector<std::string> FfmpegRunner::childProcess(const std::string &command) {
    std::vector<std::string> lines;
    try {
        // 使用 Boost.Process 启动进程并获取其输出
        std::string commandToRun = command;

        // 使用 Boost.Process 启动进程
        bp::ipstream pipeStream; // 用于读取进程输出
        bp::child c(commandToRun, bp::std_out > pipeStream);

        if (!c.valid()) {
            std::string error = "child process is valid : " + commandToRun;
            lines.emplace_back(error);
            return lines;
        }

        // 读取输出
        std::string line;
        while (pipeStream >> line) {
            lines.push_back(line);
        }
        // 等待进程结束
        c.wait();

        if (c.exit_code() != 0) {
            lines.emplace_back(std::format("Process exited with code {}, command: {}", c.exit_code(), command));
        }
    } catch (const std::exception &e) {
        lines.emplace_back(std::format("Exception: {}", e.what()));
    }

    return lines;
}

bool FfmpegRunner::isVideo(const std::string &videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, "Failed to open video file : " + videoPath));
        return false;
    }
    cap.release();
    return true;
}

bool FfmpegRunner::isAudio(const std::string &audioPath) {
    if (!bp::filesystem::exists(audioPath)) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, "Not a audio file : " + audioPath));
        return false;
    }
    
     AVFormatContext *formatContext = avformat_alloc_context();
    if (avformat_open_input(&formatContext, audioPath.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open input file." << std::endl;
        return false;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            return true;
        }
    }
    avformat_close_input(&formatContext);
    return false;
}

void FfmpegRunner::extractFrames(const std::string &videoPath, const std::string &outputImagePattern) {
    if (!isVideo(videoPath)) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, "Not a video file"));
        return;
    }

    if (!bp::filesystem::exists(bp::filesystem::path(outputImagePattern).parent_path())) {
        bp::filesystem::create_directory(bp::filesystem::path(outputImagePattern).parent_path());
    }

    std::string command = "ffmpeg -v error -i " + videoPath + " -q:v 0 -vsync 0 " + outputImagePattern;
    std::vector<std::string> results = childProcess(command);
    if (!results.empty()) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, std::accumulate(results.begin(), results.end(), std::string())));
    }
}

bool FfmpegRunner::cutVideoIntoSegments(const std::string &videoPath, const std::string &outputPath,
                                        const unsigned int &segmentDuration, const std::string &outputPattern) {
    if (!isVideo(videoPath)) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, "Not a video file : " + videoPath));
        return false;
    }
    if (!bp::filesystem::exists(outputPath)) {
        bp::filesystem::create_directory(outputPath);
    }

    std::string durationStr = std::to_string(segmentDuration);
    std::string command = "ffmpeg -v error -i " + videoPath + " -c:v copy -an -f segment -segment_time " + durationStr + " -reset_timestamps 1 -y " + outputPath + "/" + outputPattern;
    std::vector<std::string> results = childProcess(command);
    if (!results.empty()) {
        Logger::getInstance()->error(std::format("{} : {}", __FUNCTION__, std::accumulate(results.begin(), results.end(), std::string())));
        return false;
    }
    return true;
}

void FfmpegRunner::extractAudios(const std::string &videoPath, const std::string &outputDir,
                                 const FfmpegRunner::Audio_Codec &audioCodec) {
    // ffmpeg -i input.mp4 -vn -acodec copy output.aac
    if (!isVideo(videoPath)) {
        Logger::getInstance()->error("Not a video file : " + videoPath);
        return;
    }
    if (!bp::filesystem::is_directory(outputDir) && bp::filesystem::is_regular_file(outputDir)) {
        Logger::getInstance()->error("Output directory is not a directory : " + outputDir);
        return;
    }
    if (!bp::filesystem::exists(outputDir)) {
        bp::filesystem::create_directory(outputDir);
    }

    std::string audioCodecStr;
    std::string extension;
    switch (audioCodec) {
    case Codec_UNKNOWN:
    case FfmpegRunner::Audio_Codec::Codec_AAC:
        audioCodecStr = "aac";
        extension = ".aac";
        break;
    case FfmpegRunner::Audio_Codec::Codec_MP3:
        audioCodecStr = "libmp3lame";
        extension = ".mp3";
        break;
    case Codec_OPUS:
        audioCodecStr = "libopus";
        extension = ".opus";
        break;
    case Codec_VORBIS:
        audioCodecStr = "libvorbis";
        extension = ".ogg";
        break;
    }

    auto audioStreamsInfo = getAudioStreamsIndexAndCodec(videoPath);

    for (const auto &audioStreamInfo : audioStreamsInfo) {
        std::string index = std::to_string(audioStreamInfo.first);
        std::string command = "ffmpeg -v error -i " + videoPath + " -map 0:" + index + " -c:a " + audioCodecStr + " -vn -y " + outputDir + "/audio_" + index + extension;
        std::vector<std::string> results = childProcess(command);
        if (!results.empty()) {
            Logger::getInstance()->error(std::format("{} Failed to extract audio : {}", __FUNCTION__, command));
        }
    }
}

std::unordered_map<unsigned int, std::string> FfmpegRunner::getAudioStreamsIndexAndCodec(const std::string &videoPath) {
    if (!isVideo(videoPath)) {
        Logger::getInstance()->error("Not a video file : " + videoPath);
        return {};
    }

    AVFormatContext *formatContext = avformat_alloc_context();
    if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open input file." << std::endl;
        return {};
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        return {};
    }
    
    std::unordered_map<unsigned int, std::string> results;
    for (size_t i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            std::string codecName = avcodec_get_name(formatContext->streams[i]->codecpar->codec_id);
            results.emplace(i, codecName);
        }
    }
    avformat_close_input(&formatContext);
    
    return results;
}

bool FfmpegRunner::concatVideoSegments(const std::vector<std::string> &videoSegmentsPaths,
                                       const std::string &outputVideoPath, const VideoPrams &videoPrams) {
    if (bp::filesystem::is_regular_file(outputVideoPath) && bp::filesystem::exists(outputVideoPath)) {
        bp::filesystem::remove(outputVideoPath);
    }

    std::string parentPath = bp::filesystem::path(outputVideoPath).parent_path().string();
    if (bp::filesystem::is_directory(parentPath) && !bp::filesystem::exists(parentPath)) {
        bp::filesystem::create_directory(parentPath);
    }

    std::string listVideoFilePath;
    // get outputVideo base name
    std::string outputVideoBaseName = bp::filesystem::path(outputVideoPath).stem().string();
    std::string listFileName = outputVideoBaseName + "_segments.txt";
    if (bp::filesystem::is_directory(outputVideoPath)) {
        listVideoFilePath = outputVideoPath + "/" + listFileName;
    } else {
        listVideoFilePath = bp::filesystem::path(outputVideoPath).parent_path().string() + "/" + listFileName;
    }
    std::ofstream listFile(listVideoFilePath);
    if (!listFile.is_open()) {
        Logger::getInstance()->error(std::format("{} : Failed to create list file", __FUNCTION__));
        return false;
    }

    for (const auto &videoSegmentPath : videoSegmentsPaths) {
        if (!isVideo(videoSegmentPath)) {
            Logger::getInstance()->error(std::format("{} : {} is not a video file", __FUNCTION__, videoSegmentPath));
            return false;
        }
        listFile << "file '" << videoSegmentPath << "'" << std::endl;
    }
    listFile.close();
    std::string frameRate = std::to_string(videoPrams.frameRate);
    std::string outputRes = std::to_string(videoPrams.width) + "x" + std::to_string(videoPrams.height);

    std::string command;
    command = "ffmpeg -v error -f concat -safe 0 -r " + frameRate + " -i " + listVideoFilePath + " -s " + outputRes + " -c:v " + videoPrams.videoCodec + " ";
    command += getCompressionAndPresetCmd(videoPrams.quality, videoPrams.preset, videoPrams.videoCodec);
    if (bp::filesystem::is_directory(outputVideoPath)) {
        command += " -pix_fmt yuv420p -colorspace bt709 -y -r " + frameRate + " " + outputVideoPath + "/output.mp4";
    } else {
        command += " -pix_fmt yuv420p -colorspace bt709 -y -r " + frameRate + " " + outputVideoPath;
    }

    std::vector<std::string> results = childProcess(command);
    bp::filesystem::remove(listVideoFilePath);
    if (!results.empty()) {
        std::string error = std::accumulate(results.begin(), results.end(), std::string());
        Logger::getInstance()->error("Failed to concat video segments! Error: " + error);
        return false;
    }
    return true;
}

std::unordered_set<std::string> FfmpegRunner::filterVideoPaths(const std::unordered_set<std::string> &filePaths) {
    std::unordered_set<std::string> filteredPaths;
    std::for_each(filePaths.begin(), filePaths.end(), [&](const std::string &videoPath) {
        if (isVideo(videoPath)) {
            filteredPaths.insert(videoPath);
        }
    });
    return filteredPaths;
}

std::unordered_set<std::string> FfmpegRunner::filterAudioPaths(const std::unordered_set<std::string> &filePaths) {
    std::unordered_set<std::string> filteredPaths;
    std::for_each(filePaths.begin(), filePaths.end(), [&](const std::string &audioPath) {
        if (isAudio(audioPath)) {
            filteredPaths.insert(audioPath);
        }
    });
    return filteredPaths;
}

bool FfmpegRunner::addAudiosToVideo(const std::string &videoPath,
                                    const std::vector<std::string> &audioPaths,
                                    const std::string &outputVideoPath) {
    if (!isVideo(videoPath)) {
        Logger::getInstance()->error("Not a video file : " + videoPath);
        return false;
    }
    if (bp::filesystem::is_directory(outputVideoPath)) {
        Logger::getInstance()->error("Output path is a directory : " + outputVideoPath);
        return false;
    }
    if (!bp::filesystem::exists(bp::filesystem::path(outputVideoPath).parent_path().string())) {
        bp::filesystem::create_directory(bp::filesystem::path(outputVideoPath).parent_path().string());
    }

    if (audioPaths.empty()) {
        Logger::getInstance()->warn(std::format("{} No audio files to add", __FUNCTION__));
        bp::filesystem::copy(videoPath, outputVideoPath);
        return true;
    }

    std::string command = "ffmpeg -v error -i " + videoPath;
    for (const auto &audioPath : audioPaths) {
        command += " -i " + audioPath;
    }
    command += " -map 0:v:0";
    for (size_t i = 0; i < audioPaths.size(); ++i) {
        command += " -map " + std::to_string(i + 1) + ":a:0";
    }
    command += " -c:v copy -c:a copy -shortest -y " + outputVideoPath;
    std::vector<std::string> results = childProcess(command);
    if (!results.empty()) {
        Logger::getInstance()->error("Failed to add audios to video : " + command);
        return false;
    }
    return true;
}

bool FfmpegRunner::imagesToVideo(const std::string &inputImagePattern,
                                 const std::string &outputVideoPath,
                                 const FfmpegRunner::VideoPrams &videoPrams) {
    if (inputImagePattern.empty() || outputVideoPath.empty()) {
        Logger::getInstance()->error(std::format("{} : inputImagePattern or outputVideoPath is empty", __FUNCTION__));
        return false;
    }

    if (bp::filesystem::is_directory(outputVideoPath)) {
        Logger::getInstance()->error(std::format("{} : Output video path is a directory : {}", __FUNCTION__, outputVideoPath));
        return false;
    }
    if (bp::filesystem::is_regular_file(outputVideoPath)) {
        bp::filesystem::remove(outputVideoPath);
    }
    if (!bp::filesystem::exists(bp::filesystem::path(outputVideoPath).parent_path())) {
        bp::filesystem::create_directory(bp::filesystem::path(outputVideoPath).parent_path());
    }

    std::string frameRate = std::to_string(videoPrams.frameRate);
    std::string codec = videoPrams.videoCodec;
    std::string outputRes = std::to_string(videoPrams.width) + "x" + std::to_string(videoPrams.height);

    std::string command = "ffmpeg -v error -r " + frameRate + " -i " + inputImagePattern + " -s " + outputRes + " -c:v " + codec + " ";
    command += getCompressionAndPresetCmd(videoPrams.quality, videoPrams.preset, codec);
    command += " -pix_fmt yuv420p -colorspace bt709 -y -r " + frameRate + " " + outputVideoPath;

    std::vector<std::string> results = childProcess(command);
    if (!results.empty()) {
        Logger::getInstance()->error("Failed to create video from images : " + command);
        Logger::getInstance()->error(std::accumulate(results.begin(), results.end(), std::string()));
        return false;
    }
    return true;
}

std::string FfmpegRunner::map_NVENC_preset(const std::string &preset) {
    const std::unordered_set<std::string> fastPresets = {"ultrafast", "superfast", "veryfast", "faster", "fast"};
    const std::unordered_set<std::string> mediumPresets = {"medium"};
    const std::unordered_set<std::string> slowPresets = {"slow", "slower", "veryslow"};

    if (fastPresets.contains(preset)) {
        return "fast";
    } else if (mediumPresets.contains(preset)) {
        return "medium";
    } else if (slowPresets.contains(preset)) {
        return "slow";
    } else {
        Logger::getInstance()->warn(std::format("{} : Unknown preset: {}, using medium preset", __FUNCTION__, preset));
        return "medium";
    }
}

std::string FfmpegRunner::map_amf_preset(const std::string &preset) {
    const std::unordered_set<std::string> fastPresets = {"ultrafast", "superfast", "veryfast"};
    const std::unordered_set<std::string> mediumPresets = {"faster", "fast", "medium"};
    const std::unordered_set<std::string> slowPresets = {"slow", "slower", "veryslow"};

    if (fastPresets.contains(preset)) {
        return "speed";
    } else if (mediumPresets.contains(preset)) {
        return "balanced";
    } else if (slowPresets.contains(preset)) {
        return "quality";
    } else {
        Logger::getInstance()->warn(std::format("{} : Unknown preset: {}, using medium preset", __FUNCTION__, preset));
        return "balanced";
    }
}

std::string FfmpegRunner::getCompressionAndPresetCmd(const unsigned int &quality, const std::string &preset, const std::string &codec) {
    if (codec == "libx264" || codec == "libx265") {
        int crf = (int)std::round(51 - (float)(quality * 0.51));
        return "-crf " + std::to_string(crf) + " -preset " + preset;
    } else if (codec == "libvpx-vp9") {
        int crf = (int)std::round(63 - (float)(quality * 0.63));
        return "-cq " + std::to_string(crf);
    } else if (codec == "h264_nvenc" || codec == "hevc_nvenc") {
        int cq = (int)std::round(51 - (float)(quality * 0.51));
        return "-crf " + std::to_string(cq) + " -preset " + map_NVENC_preset(preset);
    } else if (codec == "h264_amf" || codec == "hevc_amf") {
        int qb_i = (int)std::round(51 - (float)(quality * 0.51));
        return "-qb_i " + std::to_string(qb_i) + " -qb_p " + std::to_string(qb_i) + " -quality " + map_amf_preset(preset);
    }
    return {};
}

FfmpegRunner::Audio_Codec FfmpegRunner::getAudioCodec(const std::string &codec) {
    if (codec == "aac") {
        return Audio_Codec::Codec_AAC;
    } else if (codec == "mp3") {
        return Audio_Codec::Codec_MP3;
    } else if (codec == "opus") {
        return Audio_Codec::Codec_OPUS;
    } else if (codec == "vorbis") {
        return Audio_Codec::Codec_VORBIS;
    } else {
        Logger::getInstance()->warn(std::format("{} : Unknown audio codec: {}", __FUNCTION__, codec));
        return Audio_Codec::Codec_UNKNOWN;
    }
}

FfmpegRunner::VideoPrams::VideoPrams(const std::string &videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        Logger::getInstance()->error(std::format("{} : Failed to open video : {}", __FUNCTION__, videoPath));
        return;
    }
    width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    frameRate = cap.get(cv::CAP_PROP_FPS);
    cap.release();
}
} // namespace Ffc