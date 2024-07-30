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
    m_staticFaces = std::make_shared<std::unordered_map<std::string, Typing::Faces>>();
    m_referenceFaces = std::make_shared<std::unordered_map<std::string, Typing::Faces>>();
}

void FaceStore::appendReferenceFace(const std::string &name, const Typing::Face &face) {
    if (name.empty() || face.isEmpty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(m_rwMutexForReference);
    (*m_referenceFaces)[name].push_back(face);
}

std::unordered_map<std::string, Typing::Faces> FaceStore::getReferenceFaces() {
    std::shared_lock<std::shared_mutex> lock(m_rwMutexForReference);
    return *m_referenceFaces;
}

void FaceStore::clearReferenceFaces() {
    std::unique_lock<std::shared_mutex> lock(m_rwMutexForReference);
    m_referenceFaces->clear();
}

void FaceStore::setStaticFaces(const Typing::VisionFrame &visionFrame, const Typing::Faces &faces) {
    if (faces.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(m_rwMutexForStatic);
    (*m_staticFaces)[createFrameHash(visionFrame)] = faces;
}

Typing::Faces FaceStore::getStaticFaces(const Typing::VisionFrame &visionFrame) {
    std::shared_lock<std::shared_mutex> lock(m_rwMutexForStatic);
    auto it = m_staticFaces->find(createFrameHash(visionFrame));
    if (it != m_staticFaces->end()) {
        return it->second;
    }
    return Typing::Faces{};
}

void FaceStore::clearStaticFaces() {
    std::unique_lock<std::shared_mutex> lock(m_rwMutexForStatic);
    m_staticFaces->clear();
}

std::string FaceStore::createFrameHash(const Typing::VisionFrame &visionFrame) {
    // 获取 Mat 数据的指针和大小
    const uchar *data = visionFrame.data;
    size_t dataSize = visionFrame.total() * visionFrame.elemSize();

    // 最终计算并获取结果
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char *>(data), dataSize, hash);
    // 将哈希结果转换为十六进制字符串
    std::ostringstream oss;
    for (unsigned char i : hash) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)i;
    }

    return oss.str();
}
} // namespace Ffc