/**
 ******************************************************************************
 * @file           : face_landmarker_68.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#include "face_landmarker_68.h"

namespace Ffc {
FaceLandmarker68::FaceLandmarker68(const std::shared_ptr<Ort::Env> &env) :
    FaceAnalyserSession(env, "./models/2dfan4.onnx") {
}

std::shared_ptr<std::tuple<Typing::FaceLandmark, Typing::Score>>
FaceLandmarker68::detect(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    this->preProcess(visionFrame, boundingBox);

    std::vector<int64_t> input_img_shape = {1, 3, this->m_inputHeight, this->m_inputWidth};
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(this->m_memoryInfo, this->m_inputImage.data(), this->m_inputImage.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ort_outputs = this->m_session->Run(runOptions, this->m_inputNames.data(), &input_tensor_, 1, this->m_outputNames.data(), this->m_outputNames.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 68, 3), 每一行的长度是3，表示一个关键点坐标x,y和置信度
    const int numPoints = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    std::vector<cv::Point2f> faceLandmark68(numPoints);
    std::vector<Typing::Score> scores(numPoints);
    for (int i = 0; i < numPoints; i++) {
        float x = pdata[i * 3] / 64.0 * 256.0;
        float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        float score = pdata[i * 3 + 2];
        faceLandmark68[i] = cv::Point2f(x, y);
        scores[i] = score;
    }
    cv::transform(faceLandmark68, faceLandmark68, this->m_invAffineMatrix);

    float sum = 0.0;
    for (int i = 0; i < numPoints; i++) {
        sum += scores[i];
    }
    float meanScore = sum / (float)numPoints;
    return std::make_shared<std::tuple<Typing::FaceLandmark,
                                       Typing::Score>>(std::make_tuple(faceLandmark68, meanScore));
}

void FaceLandmarker68::preProcess(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    m_inputHeight = m_inputNodeDims[0][2];
    m_inputWidth = m_inputNodeDims[0][3];

    float sub_max = std::max(boundingBox.xmax - boundingBox.xmin, boundingBox.ymax - boundingBox.ymin);
    const float scale = 195.f / sub_max;
    const std::vector<float> translation = {(256.f - (boundingBox.xmax + boundingBox.xmin) * scale) * 0.5f, (256.f - (boundingBox.ymax + boundingBox.ymin) * scale) * 0.5f};
   
    auto cropVisionFrameAndAffineMat = FaceHelper::warpFaceByTranslation(visionFrame, translation,
                                                                         scale, cv::Size{256, 256});
    cv::Mat cropImg = std::get<0>(*cropVisionFrameAndAffineMat);
    cv::Mat affineMatrix = std::get<1>(*cropVisionFrameAndAffineMat);
    cv::invertAffineTransform(affineMatrix, this->m_invAffineMatrix);

    std::vector<cv::Mat> bgrChannels(3);
    split(cropImg, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.0);
    }

    const int image_area = this->m_inputHeight * this->m_inputWidth;
    this->m_inputImage.resize(3 * image_area);
    const size_t single_chn_size = image_area * sizeof(float);
    const float *src_ptrs[] = {
        reinterpret_cast<const float *>(bgrChannels[0].data),
        reinterpret_cast<const float *>(bgrChannels[1].data),
        reinterpret_cast<const float *>(bgrChannels[2].data)};

    float *dst_ptr = this->m_inputImage.data();
    for (auto &src_ptr : src_ptrs) {
        std::memcpy(dst_ptr, src_ptr, single_chn_size);
        dst_ptr += image_area;
    }
}
} // namespace Ffc