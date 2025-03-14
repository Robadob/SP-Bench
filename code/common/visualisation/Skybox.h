#ifndef __Skybox_h__
#define __Skybox_h__

#include "Entity.h"
#include "texture/TextureCubeMap.h"

class Skybox : public Entity
{
public:
    Skybox(const char *texturePath = TextureCubeMap::SKYBOX_PATH, float yOffset = 0.0f);
    void render() override;
    using Entity::setModelViewMatPtr;
    void setModelViewMatPtr(const Camera *camera) override;
    void setYOffset(float yOffset);
};
#endif //ifndef __Skybox_h__