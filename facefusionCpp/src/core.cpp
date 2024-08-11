/**
 ******************************************************************************
 * @file           : core.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-19
 ******************************************************************************
 */

#include "core.h"

namespace Ffc {
Core::Core() {
    m_env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    m_config = Config::getInstance();
    m_logger = Logger::getInstance();
    m_modelsInfoJson = std::make_shared<nlohmann::json>();
    std::ifstream file("./modelsInfo.json");
    if (file.is_open()) {
        file >> *m_modelsInfoJson;
        file.close();
    } else {
        m_logger->error(std::format("Failed to open file: {}", FileSystem::resolveRelativePath("./modelsInfo.json")));
        std::exit(EXIT_FAILURE);
    }
    m_faceAnalyser = std::make_shared<Ffc::FaceAnalyser>(m_env, m_modelsInfoJson, m_config);
    m_faceMasker = std::make_shared<Ffc::FaceMasker>(m_env, m_modelsInfoJson, m_config);
}

void Core::forceDownload() {
    std::unordered_set<std::string> modelUrls;

    auto faceAnalyserModels = m_modelsInfoJson->at("faceAnalyserModels");
    for (const auto &model : faceAnalyserModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceMaskerModels = m_modelsInfoJson->at("faceMaskerModels");
    for (const auto &model : faceMaskerModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceSwapperModels = m_modelsInfoJson->at("faceSwapperModels");
    for (const auto &model : faceSwapperModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceEnhancerModels = m_modelsInfoJson->at("faceEnhancerModels");
    for (const auto &model : faceEnhancerModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    Downloader::batchDownload(modelUrls, "./models");
}

void Core::run() {
    m_logger->setLogLevel(m_config->m_logLevel);
    auto removeTempFunc = []() {
        FileSystem::removeDirectory(FileSystem::getTempPath());
    };
    if (std::atexit(removeTempFunc) != 0) {
        m_logger->warn("Failed to register exit function");
    }

    if (m_config->m_forceDownload) {
        this->forceDownload();
    }

    if (!preCheck() || !m_faceAnalyser->preCheck() || !m_faceMasker->preCheck()) {
        return;
    }

    // Start the frame processors
    for (const auto &processor : *getFrameProcessors()) {
        processor->preCheck();
    }

    conditionalProcess();
}

void Core::conditionalProcess() {
    for (const auto &processor : *getFrameProcessors()) {
        while (!processor->postCheck()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        processor->preProcess();
    }

    conditionalAppendReferenceFaces();

    m_logger->info("[Core] Filtering images...");
    std::unordered_set<std::string> targetImagePaths = FileSystem::filterImagePaths(m_config->m_targetPaths);
    if (!targetImagePaths.empty()) {
        processImages(targetImagePaths);
    }

    if (targetImagePaths.size() == m_config->m_targetPaths.size()) {
        return;
    }

    m_logger->info("[Core] Filtering videos...");
    std::unordered_set<std::string> targetVideoPaths = FfmpegRunner::filterVideoPaths(m_config->m_targetPaths);
    if (!targetVideoPaths.empty()) {
        processVideos(targetVideoPaths);
    }

    FileSystem::removeDirectory(FileSystem::getTempPath());
}

void Core::createFrameProcessors() {
    if (m_frameProcessors == nullptr) {
        m_frameProcessors = std::make_shared<std::vector<std::shared_ptr<ProcessorBase>>>();
    } else {
        m_frameProcessors->clear();
    }

    for (const auto &processor : m_config->m_frameProcessors) {
        switch (processor) {
        case Typing::EnumFrameProcessor::FaceSwapper:
            if (m_frameProcessorMap.contains(Typing::EnumFrameProcessor::FaceSwapper)) {
                m_frameProcessors->emplace_back(m_frameProcessorMap.at(Typing::EnumFrameProcessor::FaceSwapper));
            } else {
                m_frameProcessorMap.emplace(Typing::EnumFrameProcessor::FaceSwapper, std::make_shared<Ffc::FaceSwapper>(m_env, m_faceAnalyser, m_faceMasker, m_modelsInfoJson, m_config));
                m_frameProcessors->emplace_back(m_frameProcessorMap.at(Typing::EnumFrameProcessor::FaceSwapper));
            }
            break;
        case Typing::EnumFrameProcessor::FaceEnhancer:
            if (m_frameProcessorMap.contains(Typing::EnumFrameProcessor::FaceEnhancer)) {
                m_frameProcessors->emplace_back(m_frameProcessorMap.at(Typing::EnumFrameProcessor::FaceEnhancer));
            } else {
                m_frameProcessorMap.emplace(Typing::EnumFrameProcessor::FaceEnhancer, std::make_shared<Ffc::FaceEnhancer>(m_env, m_faceAnalyser, m_faceMasker, m_modelsInfoJson, m_config));
                m_frameProcessors->emplace_back(m_frameProcessorMap.at(Typing::EnumFrameProcessor::FaceEnhancer));
            }
            break;
        }
    }
}

std::shared_ptr<std::vector<std::shared_ptr<ProcessorBase>>> Core::getFrameProcessors() {
    createFrameProcessors();
    return m_frameProcessors;
}

bool Core::preCheck() const {
    return true;
}

void Core::conditionalAppendReferenceFaces() {
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference
        && m_faceStore->getReferenceFaces().empty()) {
        std::unordered_set<std::string> sourcePaths = m_config->m_sourcePaths;
        if (sourcePaths.empty()) {
            m_logger->warn("[Core] No image found in the source paths.");
        }

        if (m_config->m_referenceFacePath.empty()) {
            m_logger->error("[Core] Reference face path is empty.");
            std::exit(1);
        }

        if (!FileSystem::isImage(m_config->m_referenceFacePath)) {
            m_logger->error("[Core] Reference face path is not a valid image file.");
            std::exit(1);
        }

        auto sourceFrame = Vision::readStaticImages(sourcePaths);
        auto sourceAverageFace = m_faceAnalyser->getAverageFace(sourceFrame);
        auto referenceFrame = Vision::readStaticImage(m_config->m_referenceFacePath);
        auto referenceFace = m_faceAnalyser->getOneFace(referenceFrame, m_config->m_referenceFacePosition);

        if (referenceFace == nullptr || referenceFace->isEmpty()) {
            m_logger->error("[Core] No face found in the reference image.");
            std::exit(1);
        }

        m_faceStore->appendReferenceFace("origin", *referenceFace);
        if (!sourceAverageFace->isEmpty() && !referenceFace->isEmpty()) {
            for (const auto &processor : *getFrameProcessors()) {
                auto abstractFrame = processor->getReferenceFrame(*sourceAverageFace, *referenceFace, referenceFrame);
                if (!abstractFrame.empty()) {
                    m_faceStore->appendReferenceFace(typeid(*processor).name(), *sourceAverageFace);
                }
            }
        } else {
            m_logger->error("[Core] Source face or reference face is empty.");
        }
    }
}

void Core::processImages(std::unordered_set<std::string> imagePaths) {
    std::vector<std::string> targetImagePaths(imagePaths.begin(), imagePaths.end());
    std::vector<std::string> normedOutputPaths;

    std::string tempPath = FileSystem::getTempPath();

    m_logger->info("[Core] Coping images to temp...");
    if (!FileSystem::copyImagesToTemp(targetImagePaths, m_config->m_outputImageResolution)) {
        m_logger->error("[Core] Copy target images to temp path failed.");
        return;
    }

    std::vector<std::string> tempTargetImagePaths;
    for (const auto &targetImagePath : targetImagePaths) {
        auto tempTargetImagePath = tempPath + "/" + FileSystem::getFileName(targetImagePath);
        tempTargetImagePaths.emplace_back(tempTargetImagePath);
    }
    std::string outputPath = FileSystem::resolveRelativePath(m_config->m_outputPath);

    if (!FileSystem::directoryExists(outputPath)) {
        m_logger->info("[Core] Create output directory: " + outputPath);
        FileSystem::createDirectory(outputPath);
    }

    normedOutputPaths = FileSystem::normalizeOutputPaths(tempTargetImagePaths, outputPath);
    targetImagePaths.clear();

    auto processors = getFrameProcessors();
    for (auto processor = processors->begin(); processor != processors->end();) {
        (*processor)->processImages(m_config->m_sourcePaths, tempTargetImagePaths, tempTargetImagePaths);
        processor = processors->erase(processor);
        m_faceStore->clearStaticFaces();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Don't remove this line, it will cause a bug
    }

    m_logger->info("[Core] Finalizing images...");
    if (!FileSystem::finalizeImages(tempTargetImagePaths, tempTargetImagePaths, m_config->m_outputImageResolution, m_config->m_outputImageQuality)) {
        m_logger->warn("[Core] Some images skipped finalization!");
    }

    m_logger->info("[Core] Moving processed images to output path...");
    FileSystem::moveFiles(tempTargetImagePaths, normedOutputPaths);

    m_logger->info(std::format("[Core] All images processed successfully. Output path: {}", FileSystem::resolveRelativePath(m_config->m_outputPath)));
}

void Core::processVideos(std::unordered_set<std::string> videoPaths) {
    if (videoPaths.empty()) {
        m_logger->warn("[Core] No video found in the source paths.");
        return;
    }
    std::vector<std::string> targetVideoPaths(videoPaths.begin(), videoPaths.end());
    std::vector<std::string> tempTargetVideoPaths;
    for (const auto &targetVideoPath : targetVideoPaths) {
        std::string tempTargetVideoDir = FileSystem::getTempPath() + "/videos/" + FileSystem::getBaseName(targetVideoPath);
        while (FileSystem::directoryExists(tempTargetVideoDir)) {
            tempTargetVideoDir += "-" + FileSystem::generateRandomString(8);
        }
        std::string tempTargetVideoPath = tempTargetVideoDir + "/" + FileSystem::getFileName(targetVideoPath);
        tempTargetVideoPaths.emplace_back(tempTargetVideoPath);
    }

    m_logger->info("[Core] Coping videos to temp...");
    FileSystem::copyFiles(targetVideoPaths, tempTargetVideoPaths);

    std::string outputPath = FileSystem::resolveRelativePath(m_config->m_outputPath);
    if (!FileSystem::directoryExists(outputPath)) {
        m_logger->info("[Core] Create output directory: " + outputPath);
        FileSystem::createDirectory(outputPath);
    }
    std::vector<std::string> normedOutputPaths = FileSystem::normalizeOutputPaths(tempTargetVideoPaths, m_config->m_outputPath);

    for (size_t i = 0; i < targetVideoPaths.size(); ++i) {
        if (m_config->m_videoSegmentDuration > 0) {
            if (!processVideoInSegments(tempTargetVideoPaths[i], normedOutputPaths[i], m_config->m_videoSegmentDuration)) {
                m_logger->error(std::format("[Core] Process video {}/{} in segments failed : {}", i + 1, targetVideoPaths.size(), targetVideoPaths[i]));
            } else {
                m_logger->info(std::format("[Core] Process video {}/{} in segments successfully. Output path: {}", i + 1, targetVideoPaths.size(), normedOutputPaths[i]));
            }
        } else {
            if (!processVideo(tempTargetVideoPaths[i], normedOutputPaths[i])) {
                m_logger->error(std::format("[Core] Process video failed : {}", targetVideoPaths[i]));
            } else {
                m_logger->info(std::format("[Core] Process video {}/{} successfully. Output path: {}", i + 1, targetVideoPaths.size(), normedOutputPaths[i]));
            }
        }
        FileSystem::removeFile(tempTargetVideoPaths[i]);
    }
}

bool Core::processVideo(const std::string &videoPath, const std::string &outputVideoPath) {
    std::filesystem::path pathVideo(videoPath);

    std::string audiosDir = pathVideo.parent_path().string() + "/audios";
    if (!m_config->m_skipAudio) {
        FfmpegRunner::Audio_Codec audioCodec = FfmpegRunner::getAudioCodec(m_config->m_outputAudioEncoder);
        if (audioCodec == FfmpegRunner::Audio_Codec::Codec_UNKNOWN) {
            m_logger->warn("[Core] Unsupported audio codec. Use Default: aac");
            audioCodec = FfmpegRunner::Audio_Codec::Codec_AAC;
        }
        m_logger->info(std::format("[Core] Extract Audios for {}", videoPath));
        FfmpegRunner::extractAudios(videoPath, audiosDir, audioCodec);
    }

    m_logger->info(std::format("[Core] Extract Frames for {}", videoPath));
    std::string pattern = "frame_%06d." + m_config->m_tempFrameFormat;
    std::string videoFramesOutputDir = pathVideo.parent_path().string() + "/" + Ffc::FileSystem::getBaseName(videoPath);
    std::string outputPattern = videoFramesOutputDir + "/" + pattern;
    Ffc::FfmpegRunner::extractFrames(videoPath, outputPattern);
    std::unordered_set<std::string> framePaths = FileSystem::listFilesInDirectory(videoFramesOutputDir);
    framePaths = Ffc::FileSystem::filterImagePaths(framePaths);
    std::vector<std::string> framePathsVec(framePaths.begin(), framePaths.end());

    auto processors = getFrameProcessors();
    for (auto processor = processors->begin(); processor != processors->end(); ++processor) {
        (*processor)->processImages(m_config->m_sourcePaths, framePathsVec, framePathsVec);
        m_faceStore->clearStaticFaces();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Don't remove this line, it will cause a bug
    }

    Ffc::FfmpegRunner::VideoPrams videoPrams(videoPath);
    videoPrams.quality = m_config->m_outputVideoQuality;
    videoPrams.preset = m_config->m_outputVideoPreset;
    videoPrams.videoCodec = m_config->m_outputVideoEncoder;

    std::string inputImagePattern = videoFramesOutputDir + "/" + pattern;
    std::string outputVideo_NA_Path = pathVideo.parent_path().string() + "/" + Ffc::FileSystem::getBaseName(videoPath) + "_processed_NA" + Ffc::FileSystem::getExtension(videoPath);
    Ffc::Logger::getInstance()->info("[Core] Images to video...");
    if (!Ffc::FfmpegRunner::imagesToVideo(inputImagePattern, outputVideo_NA_Path, videoPrams)) {
        Ffc::Logger::getInstance()->error("[Core] images to video failed!");
        FileSystem::removeDirectory(videoFramesOutputDir);
        FileSystem::removeFile(outputVideo_NA_Path);
        return false;
    }

    std::unordered_set<std::string> audioPaths = Ffc::FileSystem::listFilesInDirectory(audiosDir);
    audioPaths = Ffc::FfmpegRunner::filterAudioPaths(audioPaths);
    std::vector<std::string> audioPathsVec(audioPaths.begin(), audioPaths.end());

    Ffc::Logger::getInstance()->info("[Core] Add audios to video ...");
    if (!Ffc::FfmpegRunner::addAudiosToVideo(outputVideo_NA_Path, audioPathsVec, outputVideoPath)) {
        Ffc::Logger::getInstance()->warn("[Core] Add audios to Video failed. The output video will be without audio.");
    }

    Ffc::FileSystem::removeDirectory(videoFramesOutputDir);
    Ffc::FileSystem::removeDirectory(audiosDir);
    Ffc::FileSystem::removeFile(outputVideo_NA_Path);
    return true;
}

bool Core::processVideoInSegments(const std::string &videoPath, const std::string &outputVideoPath, const unsigned int &duration) {
    std::filesystem::path pathVideo(videoPath);

    std::string audiosDir = pathVideo.parent_path().string() + "/audios";
    if (!m_config->m_skipAudio) {
        FfmpegRunner::Audio_Codec audioCodec = FfmpegRunner::getAudioCodec(m_config->m_outputAudioEncoder);
        if (audioCodec == FfmpegRunner::Audio_Codec::Codec_UNKNOWN) {
            m_logger->warn("[Core] Unsupported audio codec. Use Default: aac");
            audioCodec = FfmpegRunner::Audio_Codec::Codec_AAC;
        }
        m_logger->info(std::format("[Core] Extract Audios for {}", videoPath));
        FfmpegRunner::extractAudios(videoPath, audiosDir, audioCodec);
    }

    std::string videoSegmentsDir = pathVideo.parent_path().string() + "/videoSegments";
    std::string videoSegmentPattern = "segment_%03d" + FileSystem::getExtension(videoPath);
    Ffc::Logger::getInstance()->info(std::format("[Core] Divide the video into segments of {} seconds each....", duration));
    if (!Ffc::FfmpegRunner::cutVideoIntoSegments(videoPath, videoSegmentsDir, duration, videoSegmentPattern)) {
        Ffc::Logger::getInstance()->error("The attempt to cut the video into segments was failed!");
        FileSystem::removeDirectory(audiosDir);
        return false;
    }

    std::unordered_set<std::string> videoSegmentsPaths = Ffc::FileSystem::listFilesInDirectory(videoSegmentsDir);
    videoSegmentsPaths = Ffc::FfmpegRunner::filterVideoPaths(videoSegmentsPaths);

    std::vector<std::string> processedVideoSegmentsPaths;
    std::string processedVideoSegmentsDir = pathVideo.parent_path().string() + "/videoSegments_processed";
    size_t segmentIndex = 0;
    auto processors = getFrameProcessors();
    for (const auto &videoSegmentPath : videoSegmentsPaths) {
        m_logger->info(std::format("[Core] Processing video segment {}/{}", segmentIndex + 1, videoSegmentsPaths.size()));
        segmentIndex++;

        std::string pattern = "frame_%06d." + m_config->m_tempFrameFormat;
        std::string videoFramesOutputDir = std::filesystem::path(videoSegmentPath).parent_path().string() + "/" + Ffc::FileSystem::getBaseName(videoSegmentPath);
        std::string outputPattern = videoFramesOutputDir + "/" + pattern;
        Ffc::FfmpegRunner::extractFrames(videoSegmentPath, outputPattern);
        std::unordered_set<std::string> framePaths = Ffc::FileSystem::listFilesInDirectory(videoFramesOutputDir);
        framePaths = Ffc::FileSystem::filterImagePaths(framePaths);
        std::vector<std::string> framePathsVec(framePaths.begin(), framePaths.end());

        for (auto processor = processors->begin(); processor != processors->end(); ++processor) {
            (*processor)->processImages(m_config->m_sourcePaths, framePathsVec, framePathsVec);
            m_faceStore->clearStaticFaces();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Don't remove this line, it will cause a bug
        }

        std::string outputVideoSegmentPath = processedVideoSegmentsDir + "/" + Ffc::FileSystem::getFileName(videoSegmentPath);
        const std::string &inputImagePattern = outputPattern;
        Ffc::FfmpegRunner::VideoPrams videoPrams(videoSegmentPath);
        videoPrams.quality = m_config->m_outputVideoQuality;
        videoPrams.preset = m_config->m_outputVideoPreset;
        videoPrams.videoCodec = m_config->m_outputVideoEncoder;

        Ffc::Logger::getInstance()->info("[Core] Images to video " + outputVideoSegmentPath);
        if (Ffc::FfmpegRunner::imagesToVideo(inputImagePattern, outputVideoSegmentPath, videoPrams)) {
            processedVideoSegmentsPaths.push_back(outputVideoSegmentPath);
        } else {
            Ffc::Logger::getInstance()->error("[Core] images to video failed!");
            FileSystem::removeDirectory(videoSegmentsDir);
            FileSystem::removeDirectory(videoFramesOutputDir);
            FileSystem::removeDirectory(processedVideoSegmentsDir);
            return false;
        }

        Ffc::FileSystem::removeFile(videoSegmentPath);
        Ffc::FileSystem::removeDirectory(videoFramesOutputDir);
    }
    Ffc::FileSystem::removeDirectory(videoSegmentsDir);

    Ffc::FfmpegRunner::VideoPrams videoPrams(videoPath);
    videoPrams.quality = m_config->m_outputVideoQuality;
    videoPrams.preset = m_config->m_outputVideoPreset;
    videoPrams.videoCodec = m_config->m_outputVideoEncoder;

    std::string outputVideo_NA_Path = pathVideo.parent_path().string() + "/" + Ffc::FileSystem::getBaseName(videoPath) + "_processed_NA" + Ffc::FileSystem::getExtension(videoPath);
    std::vector<std::string> videoSegmentsPathsVec(videoSegmentsPaths.begin(), videoSegmentsPaths.end());
    Ffc::Logger::getInstance()->info("[Core] concat video segments...");
    if (!Ffc::FfmpegRunner::concatVideoSegments(processedVideoSegmentsPaths, outputVideo_NA_Path, videoPrams)) {
        Ffc::Logger::getInstance()->error("[Core] Failed concat video segments for : " + videoPath);
        FileSystem::removeDirectory(processedVideoSegmentsDir);
        return false;
    }

    std::unordered_set<std::string> audioPaths = Ffc::FileSystem::listFilesInDirectory(audiosDir);
    audioPaths = Ffc::FfmpegRunner::filterAudioPaths(audioPaths);
    std::vector<std::string> audioPathsVec(audioPaths.begin(), audioPaths.end());

    Ffc::Logger::getInstance()->info("[Core] Add audios to video...");
    if (!Ffc::FfmpegRunner::addAudiosToVideo(outputVideo_NA_Path, audioPathsVec, outputVideoPath)) {
        Ffc::Logger::getInstance()->warn("[Core] Add audios to Video failed. The output video will be without audio.");
    }

    Ffc::FileSystem::removeDirectory(audiosDir);
    Ffc::FileSystem::removeFile(outputVideo_NA_Path);
    Ffc::FileSystem::removeDirectory(processedVideoSegmentsDir);
    return true;
}

} // namespace Ffc