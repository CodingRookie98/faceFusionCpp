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
void FaceDetectorYolo::preProcess(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];
    // Todo
    //    this->m_inputHeight = faceDetectorSize.height;
    //    this->m_inputWidth = faceDetectorSize.width;

    const int height = visionFrame.rows;
    const int width = visionFrame.cols;
    // Todo
    //    cv::Mat tempImage = Vision::resizeFrameResolution(visionFrame, Globals::faceDetectorSize);
    cv::Mat tempImage = visionFrame.clone();
    if (height > this->m_inputHeight || width > this->m_inputWidth) {
        const float scale = std::min((float)this->m_inputHeight / height, (float)this->m_inputWidth / width);
        cv::Size newSize = cv::Size(int(width * scale), int(height * scale));
        resize(visionFrame, tempImage, newSize);
    }
    this->m_ratioHeight = (float)height / (float)tempImage.rows;
    this->m_ratioWidth = (float)width / (float)tempImage.cols;
    cv::Mat inputImg;
    cv::copyMakeBorder(tempImage, inputImg, 0, this->m_inputHeight - tempImage.rows, 0, this->m_inputWidth - tempImage.cols, cv::BORDER_CONSTANT, 0);

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(inputImg, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int imageArea = this->m_inputHeight * this->m_inputWidth;
    this->m_inputImage.resize(3 * imageArea);
    const size_t singleChnSize = imageArea * sizeof(float);
    const float *srcPtrs[] = {
        reinterpret_cast<const float *>(bgrChannels[0].data),
        reinterpret_cast<const float *>(bgrChannels[1].data),
        reinterpret_cast<const float *>(bgrChannels[2].data)};

    float *dstPtr = this->m_inputImage.data();
    for (auto &srcPtr : srcPtrs) {
        std::memcpy(dstPtr, srcPtr, singleChnSize);
        dstPtr += imageArea;
    }
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceDetectorYolo::detect(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    this->preProcess(visionFrame, faceDetectorSize);

    std::vector<int64_t> inputImgShape = {1, 3, this->m_inputHeight, this->m_inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(this->m_memoryInfo, this->m_inputImage.data(), this->m_inputImage.size(), inputImgShape.data(), inputImgShape.size());
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = this->m_session->Run(runOptions, this->m_inputNames.data(), &inputTensor, 1, this->m_outputNames.data(), m_outputNames.size());

    // 不需要手动释放 pdata，它由 Ort::Value 管理
    float *pdata = ortOutputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int numBox = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];

    std::vector<BoundingBox> boundingBoxRaw;
    std::vector<float> scoreRaw;
    std::vector<Ffc::Typing::FaceLandmark> landmarkRaw;
    for (int i = 0; i < numBox; i++) {
        const float score = pdata[4 * numBox + i];
        if (score > Ffc::Globals::faceDetectorScore) {
            float xmin = (pdata[i] - 0.5 * pdata[2 * numBox + i]) * this->m_ratioWidth;           ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[numBox + i] - 0.5 * pdata[3 * numBox + i]) * this->m_ratioHeight; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * numBox + i]) * this->m_ratioWidth;           ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[numBox + i] + 0.5 * pdata[3 * numBox + i]) * this->m_ratioHeight; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图

            // 坐标的越界检查保护，可以添加一下
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
                point2F.x = pdata[j * numBox + i] * this->m_ratioWidth;
                point2F.y = pdata[(j + 1) * numBox + i] * this->m_ratioHeight;
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
                                   const std::shared_ptr<nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = "./models/yoloface_8n.onnx";

    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::downloadFileFromURL(m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_yoloface").at("url"),
                                                               "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    this->createSession(modelPath);
}

} // namespace Ffc
