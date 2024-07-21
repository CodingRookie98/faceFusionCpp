/**
 ******************************************************************************
 * @file           : face_store.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-20
 ******************************************************************************
 */

#include "face_store.h"

namespace Ffc {
std::shared_ptr<FaceStore> FaceStore::getInstance() {
    static std::shared_ptr<FaceStore> instance;
    static std::once_flag flag;
    std::call_once(flag, [&]() { instance = std::make_shared<FaceStore>(); });
    return instance;
}

FaceStore::FaceStore() {
    m_staticFaces  = std::make_shared<std::unordered_map<std::string, Typing::Faces>>();
    m_referenceFaces = std::make_shared<std::unordered_map<std::string, Typing::Faces>>();
}

void FaceStore::appendReferenceFace(const std::string &name, const Typing::Face &face) {
    (*m_referenceFaces)[name].push_back(face);
}

std::unordered_map<std::string, Typing::Faces> FaceStore::getReferenceFaces() const {
    return *m_referenceFaces;
}

void FaceStore::clearReferenceFaces() {
    m_referenceFaces->clear();
}

void FaceStore::setStaticFaces(const Typing::VisionFrame &visionFrame, const Typing::Faces &faces) {
    (*m_staticFaces)[createFrameHash(visionFrame)] = faces;
}

Typing::Faces FaceStore::getStaticFaces(const Typing::VisionFrame &visionFrame) const {
    auto it = m_staticFaces->find(createFrameHash(visionFrame));
    if (it != m_staticFaces->end()) {
        return it->second;
    }
    return Typing::Faces{};
}

void FaceStore::clearStaticFaces() {
    m_staticFaces->clear();
}
std::string FaceStore::createFrameHash(const Typing::VisionFrame &visionFrame) {
    // 获取 Mat 数据的指针和大小
    const uchar *data = visionFrame.data;
    size_t dataSize = visionFrame.total() * visionFrame.elemSize();

    // 使用 std::hash 计算哈希值
    std::hash<std::string> hasher;
    std::string hashString(reinterpret_cast<const char *>(data), dataSize);
    auto hashValue = hasher(hashString);
    return std::to_string(hashValue);
}
} // namespace Ffc