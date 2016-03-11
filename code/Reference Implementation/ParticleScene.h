#pragma once


#include "visualisation/VisualisationScene.h"
#include "visualisation/Entity.h"
#include "visualisation/Shaders.h"
#include <driver_types.h>
#ifdef _3D
#define DIMENSIONS 3
#else
#define DIMENSIONS 2
#endif
class ParticleScene : public VisualisationScene
{
public:
    ParticleScene(Camera* camera = nullptr);
    ~ParticleScene();

    void update() override;
    void reload() override;
    void render(glm::mat4 projection) override;
    void generate() override;
    void setCount(unsigned int count);
    //void setMax(unsigned int max);
    void setTex(GLuint* const tex);

private:
    //void createTextureBufferObject(GLuint *tbo, GLuint *tex, GLuint size, GLuint type);
   // void deleteTextureBufferObject(GLuint *tbo, GLuint *tex);
    Entity entity;
    Shaders shaders;
    unsigned int count;// , max;
    GLuint tex[DIMENSIONS];
};

