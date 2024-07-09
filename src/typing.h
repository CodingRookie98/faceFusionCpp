/**
 ******************************************************************************
 * @file           : data_types.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_TYPING_H_
#define FACEFUSIONCPP_SRC_TYPING_H_

#include <vector>
#include <opencv2/imgproc.hpp>

namespace Ffc {
namespace Typing {

typedef struct BoudingBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} BoundingBox;

typedef std::vector<float> Embedding;
typedef std::vector<cv::Point2f> FaceLandmark;
typedef float Score;

typedef struct Face {
    BoundingBox boundingBox;
    FaceLandmark faceLandmark5;
    FaceLandmark faceLandmark68;
    FaceLandmark faceLandMark5_68;
    FaceLandmark faceLandmark68_5;
    
    Embedding embedding;
    Embedding normedEmbedding;
    Score detectorScore;
    Score landmarkerScore;

    bool isEmpty() const {
        return faceLandmark68.empty();
    }
} Face;

typedef std::vector<Face> Faces;

typedef cv::Mat VisionFrame;

typedef struct {
    int width;
    int height;
} Resolution;
typedef std::tuple<int, int, int, int> Padding;
}
} // namespace Ffc::Typing
#endif // FACEFUSIONCPP_SRC_TYPING_H_
