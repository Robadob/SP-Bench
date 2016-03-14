#include "ParticleScene.h"

#include "visualisation/GLcheck.h"

template<class T>
ParticleScene<T>::ParticleScene(Visualisation &visualisation, Circles<T> &model)
    : Scene(visualisation, new Shaders("../shaders/instanced.vert", "../shaders/instanced.frag"))
    , entity("../models/icosphere.obj", 1.0f, shaders)
    , count(model.agentMax)
    , model(model)
{
    this->visualisation.setWindowTitle("Circles Benchmark");
    this->visualisation.setRenderAxis(true);
    setTex(model.getPartition()->getLocationTexNames());
}
template<class T>
ParticleScene<T>::~ParticleScene()
{
    delete shaders;
}
/*
Sets the texture buffers that the shaders should use
@param Pointer to array of DIMENSIONS texture names
*/
template<class T>
void ParticleScene<T>::setTex(const GLuint* tex)
{
    memcpy(this->tex, tex, DIMENSIONS*sizeof(GLuint));
    this->shaders->addTextureUniform(tex[0], "tex_locX");
    this->shaders->addTextureUniform(tex[1], "tex_locY");
#ifdef _3D
    this->shaders->addTextureUniform(tex[2], "tex_locZ");
#endif
#ifdef _GL
    this->shaders->addTextureUniform(model.getPartition()->getCountTexName(), "tex_count");
#endif
}
/*
Sets the number of instances to be rendered
*/
template<class T>
void ParticleScene<T>::setCount(unsigned int count)
{
    this->count = count;
}
/*
Steps the model
@note This is currently done externally
*/
template<class T>
void ParticleScene<T>::update()
{
    //Update agent count
    setCount(this->model.getPartition()->getLocationCount());
}
/*
Refreshes shaders
*/
template<class T>
void ParticleScene<T>::reload()
{
    this->shaders->reload();
}
/*
Renders a frame
*/
template<class T>
void ParticleScene<T>::render()
{
    if (this->count <= 0)
        return;
    this->shaders->useProgram();
    this->entity.renderInstances(count);
}
