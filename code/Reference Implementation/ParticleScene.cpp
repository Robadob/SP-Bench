#include "ParticleScene.h"
#include <cuda_runtime_api.h>
#include <glm/gtc/type_ptr.hpp>

ParticleScene::ParticleScene(Camera* camera)
    :VisualisationScene(camera)
    , entity("../models/icosphere.obj", 1)
    , shaders("../shaders/instanced.vert", "../shaders/instanced.frag")
    , count(0)
    //, max(0)
{
    //Init after 3rd party has set max
}
ParticleScene::~ParticleScene()
{
}
//    if (this->max != 0)
//    {
//        deleteTextureBufferObject(&this->tbo[0], &this->tex[0]);
//        deleteTextureBufferObject(&this->tbo[1], &this->tex[1]);
//        deleteTextureBufferObject(&this->tbo[2], &this->tex[2]);
//    }
//}
//void ParticleScene::deleteTextureBufferObject(GLuint *tbo, GLuint *tex)
//{
//    glBindBuffer(1, *tbo);
//    glDeleteBuffers(1, tbo);
//    glDeleteTextures(1, tex);
//    *tbo = 0;
//    *tex = 0;
//}
//void ParticleScene::setMax(unsigned int max)
//{
//    if (this->max!=0)
//    {
//        //free existing if not first call
//        deleteTextureBufferObject(&this->tbo[0], &this->tex[0]);
//        deleteTextureBufferObject(&this->tbo[1], &this->tex[1]);
//        deleteTextureBufferObject(&this->tbo[2], &this->tex[2]);
//    }
//    createTextureBufferObject(&this->tbo[0], &this->tex[0], max*sizeof(float), GL_R32F);//Location-x
//    createTextureBufferObject(&this->tbo[1], &this->tex[1], max*sizeof(float), GL_R32F);//Location-y
//    createTextureBufferObject(&this->tbo[2], &this->tex[2], max*sizeof(float), GL_R32F);//Location-z
//}
//void ParticleScene::createTextureBufferObject(GLuint *tbo, GLuint *tex, GLuint size, GLuint type)
//{
//    glGenTextures(1, tex);
//    glGenBuffers(1, tbo);
//
//    glBindBuffer(GL_TEXTURE_BUFFER, *tbo);
//    glBufferData(GL_TEXTURE_BUFFER, size, 0, GL_STATIC_DRAW);
//
//    glBindTexture(GL_TEXTURE_BUFFER, *tex);
//    glTexBuffer(GL_TEXTURE_BUFFER, type, *tbo);//GL_RGBA32F==float4, GL_RG32UI==uint2, GL_R32F==float1
//    glBindBuffer(GL_TEXTURE_BUFFER, 0);
//    glBindTexture(GL_TEXTURE_BUFFER, 0);
//}

void ParticleScene::setTex(GLuint* const tex)
{
    memcpy(this->tex, tex, DIMENSIONS*sizeof(GLuint));
}
void ParticleScene::setCount(unsigned int count)
{
    this->count = count;
}
void ParticleScene::update()
{
    //Call function pointer to timestep model
}
void ParticleScene::reload()
{
    this->shaders.reloadShaders();
}
void ParticleScene::render(glm::mat4 projection)
{
    if (count == 0)
        return;
    this->shaders.useProgram();
    //Set texture buffers
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tex[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, tex[1]);
#ifdef _3D
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, tex[2]);
#endif
    //Pass texture locations to shader
    glUniform1i(glGetUniformLocation(this->shaders.getProgram(), "tex_locX"), 0);
    glUniform1i(glGetUniformLocation(this->shaders.getProgram(), "tex_locY"), 1);
#ifdef _3D
    glUniform1i(glGetUniformLocation(this->shaders.getProgram(), "tex_locZ"), 2);
#endif
    //Set matrices
    this->shaders.setUniformMatrix4fv(0, &this->camera->view()[0][0]);
    this->shaders.setUniformMatrix4fv(1, &projection[0][0]);
 //   glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(projection));
  //  glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(this->camera->view()));
    glBindFragDataLocation(this->shaders.getProgram(), 0, "outColor");
    //Draw
    this->entity.renderInstances(count);
    this->shaders.clearProgram();
}
void ParticleScene::generate()
{
    //No generation required
}
