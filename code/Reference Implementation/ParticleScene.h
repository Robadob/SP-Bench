#ifndef __ParticleScene_h__
#define __ParticleScene_h__

#include "visualisation/Scene.h"
#include "visualisation/Entity.h"
#include "Circles.cuh"

#ifdef _3D
#define DIMENSIONS 3
#else
#define DIMENSIONS 2
#endif

template<class T>
class ParticleScene : protected Scene
{
public:

    ParticleScene(Visualisation &visualisation, Circles<T> &model);

    void render() override;
    void reload() override;
    void update() override;
    void setCount(unsigned int count);

private:
    void renderPBM();
    ~ParticleScene();//Private to prevent stack allocation
    void setTex(const GLuint *tex);//Sets the instance data arrays

    std::shared_ptr<Entity> icosphere;
    unsigned int count;// Number of agents within the texture buffer
    GLuint tex[DIMENSIONS];//Array of texture arrays containing our instance data to be rendered

    Circles<T> &model;
};

#include "ParticleScene.hpp"

#endif __ParticleScene_h__
