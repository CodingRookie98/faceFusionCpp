/**
 ******************************************************************************
 * @file           : ffmpeg_runner.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-8-6
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_FACEFUSIONCPP_FFMPEG_RUNNER_H_
#define FACEFUSIONCPP_FACEFUSIONCPP_FFMPEG_RUNNER_H_

#include <boost/process.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "logger.h"

namespace Ffc {
namespace bp = boost::process;
class FfmpegRunner {
public:
    enum AV_Codec {
        Codec_H264,
        Codec_H265,
        Codec_VP9,
        Codec_H264_NVENC,
        Codec_HEVC_NVENC,
    };
    enum Audio_Codec {
        Codec_AAC,
        Codec_MP3,
        Codec_OPUS,
        Codec_VORBIS,
        Codec_UNKNOWN
    };

    struct VideoPrams {
        std::string r_frame_rate;
        std::string avg_frame_rate;
        unsigned int width;
        unsigned int height;
        unsigned int quality;
        std::string preset;
        std::string videoCodec;

        void setPrams(const std::unordered_map<std::string, std::any> &mapPrams);
        void setPramsFromJson(const nlohmann::json &jsonPrams);
    };

    static std::vector<std::string> childProcess(const std::string &command);
    static bool isVideo(const std::string &videoPath);
    static bool isAudio(const std::string &videoPath);
    static bool hasAudio(const std::string &videoPath);
    static void extractFrames(const std::string &videoPath, const std::string &outputImagePattern);
    static void extractAudios(const std::string &videoPath, const std::string &outputDir, const Audio_Codec &audioCodec = Codec_AAC);
    static bool addAudiosToVideo(const std::string &videoPath, const std::vector<std::string> &audioPaths,
                                 const std::string &outputVideoPath);

    // no audio output
    static bool cutVideoIntoSegments(const std::string &videoPath, const std::string &outputPath,
                                     const unsigned int &segmentDuration, const std::string &outputPattern);

    static bool concatVideoSegments(const std::vector<std::string> &videoSegmentsPaths,
                                    const std::string &outputVideoPath, const VideoPrams &videoPrams);

    static bool getVideoInfoJsonFile(const std::string &videoPath, std::string &videoInfoJsonFilePath);
    static std::string getVideoInfoJson(const std::string &videoPath);

    static std::unordered_map<int, std::string> getAudioStreamsIndexAndCodec(const std::string &videoPath);

    static std::unordered_set<std::string> filterVideoPaths(const std::unordered_set<std::string> &filePaths);
    static std::unordered_set<std::string> filterAudioPaths(const std::unordered_set<std::string> &filePaths);

    static bool imagesToVideo(const std::string &inputImagePattern, const std::string &outputVideoPath,
                              const VideoPrams &videoPrams);
    static Audio_Codec getAudioCodec(const std::string &codec);

private:
    static std::string map_NVENC_preset(const std::string &preset);
    static std::string map_amf_preset(const std::string &preset);
    static std::string getCompressionAndPresetCmd(const unsigned int &quality, const std::string &preset, const std::string &codec);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_FACEFUSIONCPP_FFMPEG_RUNNER_H_
