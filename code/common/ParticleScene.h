#ifndef __ParticleScene_h__
#define __ParticleScene_h__

#include "visualisation/Scene.h"
#include "visualisation/Entity.h"
#include "benchmarks/circles/Circles.cuh"
#include <memory>

#ifdef _3D
#define DIMENSIONS 3
#else
#define DIMENSIONS 2
#endif

class ParticleScene : protected Scene
{
public:

    ParticleScene(Visualisation &visualisation, std::shared_ptr<Model> model);
    ~ParticleScene() override;
    void render() override;
    void reload() override;
    void update(unsigned int frameTime) override;
    bool keypress(SDL_Keycode keycode, int x, int y) override;
    void setCount(unsigned int count);

private:
    bool drawPBM;
    void renderPBM();
    void setTex(const GLuint *tex);//Sets the instance data arrays

    std::shared_ptr<Entity> icosphere;
    unsigned int count;// Number of agents within the texture buffer
    GLuint tex[DIMENSIONS];//Array of texture arrays containing our instance data to be rendered

    std::shared_ptr<Model> model;
};

#endif __ParticleScene_h__
