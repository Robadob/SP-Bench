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
    void renderPBM();
    void setCount(unsigned int count);

private:
    ~ParticleScene();//Private to prevent stack allocation
    void setTex(const GLuint *tex);

    Entity entity;
    unsigned int count;// , max;
    GLuint tex[DIMENSIONS];

    Circles<T> &model;
};

#include "ParticleScene.hpp"

#endif __ParticleScene_h__
