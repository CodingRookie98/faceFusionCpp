/**
 ******************************************************************************
 * @file           : face_store.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-20
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_STORE_H_
#define FACEFUSIONCPP_SRC_FACE_STORE_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <openssl/sha.h>
#include <iomanip>
#include <shared_mutex>
#include "typing.h"

namespace Ffc {

class FaceStore {
public:
    FaceStore();
    ~FaceStore() = default;
    FaceStore(const FaceStore &) = delete;
    FaceStore &operator=(const FaceStore &) = delete;
    FaceStore(FaceStore &&) = delete;
    FaceStore &operator=(FaceStore &&) = delete;

    static std::shared_ptr<FaceStore> getInstance();
    void appendReferenceFace(const std::string &name, const Typing::Face &face);
    std::unordered_map<std::string, Typing::Faces> getReferenceFaces();
    void clearReferenceFaces();
    void setStaticFaces(const Typing::VisionFrame &visionFrame, const Typing::Faces &faces);
    Typing::Faces getStaticFaces(const Typing::VisionFrame &visionFrame);
    void clearStaticFaces();

private:
    std::shared_ptr<std::unordered_map<std::string, Typing::Faces>> m_staticFaces;
    std::shared_ptr<std::unordered_map<std::string, Typing::Faces>> m_referenceFaces;
    std::shared_mutex m_rwMutexForStatic;
    std::shared_mutex m_rwMutexForReference;

    static std::string createFrameHash(const Typing::VisionFrame &visionFrame);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_STORE_H_
