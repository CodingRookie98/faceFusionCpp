/**
 ******************************************************************************
 * @file           : face_yolov_8.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#include "face_detector_yolo.h"

namespace Ffc {
using namespace Typing;
std::tuple<std::vector<float>, float, float>
FaceDetectorYolo::preProcess(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    const int faceDetectorHeight = faceDetectorSize.height;
    const int faceDetectorWidth = faceDetectorSize.width;

    cv::Mat tempVisionFrame = Vision::resizeFrameResolution(visionFrame, faceDetectorSize);
    float ratioHeight = (float)visionFrame.rows / (float)tempVisionFrame.rows;
    float ratioWidth = (float)visionFrame.cols / (float)tempVisionFrame.cols;

    // 创建一个指定尺寸的全零矩阵
    cv::Mat detectVisionFrame = cv::Mat::zeros(faceDetectorHeight, faceDetectorWidth, CV_32FC3);
    // 将输入的图像帧复制到全零矩阵的左上角
    tempVisionFrame.copyTo(detectVisionFrame(cv::Rect(0, 0, tempVisionFrame.cols, tempVisionFrame.rows)));

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(detectVisionFrame, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int imageArea = faceDetectorWidth * faceDetectorHeight;
    std::vector<float> inputData;
    inputData.resize(3 * imageArea);
    const size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputData.data(), (float *)bgrChannels[0].data, singleChnSize);
    memcpy(inputData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(inputData.data() + imageArea * 2, (float *)bgrChannels[2].data, singleChnSize);
    return std::make_tuple(inputData, ratioHeight, ratioWidth);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceDetectorYolo::detect(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize,
                         const float &scoreThreshold) {
    std::vector<float> inputData;
    float ratioHeight, ratioWidth;
    std::tie(inputData, ratioHeight, ratioWidth) = this->preProcess(visionFrame, faceDetectorSize);

    std::vector<int64_t> inputImgShape = {1, 3, faceDetectorSize.height, faceDetectorSize.width};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(m_memoryInfo, inputData.data(), inputData.size(), inputImgShape.data(), inputImgShape.size());
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = m_session->Run(runOptions, m_inputNames.data(), &inputTensor, 1, m_outputNames.data(), m_outputNames.size());

    // 不需要手动释放 pdata，它由 Ort::Value 管理
    float *pdata = ortOutputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int numBox = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];

    std::vector<BoundingBox> boundingBoxRaw;
    std::vector<float> scoreRaw;
    std::vector<Ffc::Typing::FaceLandmark> landmarkRaw;
    for (int i = 0; i < numBox; i++) {
        const float score = pdata[4 * numBox + i];
        if (score > scoreThreshold) {
            float xmin = (pdata[i] - 0.5 * pdata[2 * numBox + i]) * ratioWidth;           ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[numBox + i] - 0.5 * pdata[3 * numBox + i]) * ratioHeight; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * numBox + i]) * ratioWidth;           ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[numBox + i] + 0.5 * pdata[3 * numBox + i]) * ratioHeight; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图

            // 坐标的越界检查保护
            xmin = std::max(0.0f, std::min(xmin, static_cast<float>(visionFrame.cols)));
            ymin = std::max(0.0f, std::min(ymin, static_cast<float>(visionFrame.rows)));
            xmax = std::max(0.0f, std::min(xmax, static_cast<float>(visionFrame.cols)));
            ymax = std::max(0.0f, std::min(ymax, static_cast<float>(visionFrame.rows)));

            boundingBoxRaw.emplace_back(BoundingBox{xmin, ymin, xmax, ymax});
            scoreRaw.emplace_back(score);

            // 剩下的5个关键点坐标的计算
            Ffc::Typing::FaceLandmark faceLandmark;
            for (int j = 5; j < 20; j += 3) {
                cv::Point2f point2F;
                point2F.x = pdata[j * numBox + i] * ratioWidth;
                point2F.y = pdata[(j + 1) * numBox + i] * ratioHeight;
                faceLandmark.emplace_back(point2F);
            }
            landmarkRaw.emplace_back(faceLandmark);
        }
    }

    auto result = std::make_shared<std::tuple<std::vector<Typing::BoundingBox>,
                                              std::vector<Typing::FaceLandmark>,
                                              std::vector<Typing::Score>>>(boundingBoxRaw, landmarkRaw, scoreRaw);
    return result;
}

FaceDetectorYolo::FaceDetectorYolo(const std::shared_ptr<Ort::Env> &env,
                                   const std::shared_ptr<const nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_yoloface").at("path");

    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_yoloface").at("url"),
                                                    "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    this->createSession(modelPath);
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];
}

} // namespace Ffc
