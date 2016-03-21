#include "ParticleScene.h"

#include "visualisation/GLcheck.h"

template<class T>
ParticleScene<T>::ParticleScene(Visualisation &visualisation, Circles<T> &model)
    : Scene(visualisation)
    , icosphere(new Entity(Stock::Models::ICOSPHERE, 1.0f, std::make_shared<Shaders>("../shaders/instanced.vert","../shaders/instanced.frag")))
    , count(model.agentMax)
    , model(model)
{
    registerEntity(icosphere);
    this->visualisation.setWindowTitle("Circles Benchmark");
    this->visualisation.setRenderAxis(true);
    setTex(model.getPartition()->getLocationTexNames());
}
/*
Sets the texture buffers that the shaders should use
@param Pointer to array of DIMENSIONS texture names
*/
template<class T>
void ParticleScene<T>::setTex(const GLuint* tex)
{
    memcpy(this->tex, tex, DIMENSIONS*sizeof(GLuint));
    this->icosphere->getShaders()->addTextureUniform(tex[0], "tex_locX");
    this->icosphere->getShaders()->addTextureUniform(tex[1], "tex_locY");
#ifdef _3D
    this->icosphere->getShaders()->addTextureUniform(tex[2], "tex_locZ");
#endif
#ifdef _GL
    this->icosphere->getShaders()->addTextureUniform(model.getPartition()->getCountTexName(), "tex_count");
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
//Restart model?
}
/*
Renders a frame
*/
template<class T>
void ParticleScene<T>::render()
{
    if (this->count <= 0)
        return;
    this->renderPBM();
    this->icosphere->renderInstances(count);
}
/*
Renders a frame
*/
template<class T>
void ParticleScene<T>::renderPBM()
{
    DIMENSIONS_IVEC dim = this->model.getPartition()->getGridDim();
    DIMENSIONS_VEC envMin = this->model.getPartition()->getEnvironmentMin();
    DIMENSIONS_VEC envMax = this->model.getPartition()->getEnvironmentMax();
    float cellSize = this->model.getPartition()->getCellSize();
    glUseProgram(0); //Use default shader
    GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
    glPushMatrix();
    glDisable(GL_LIGHTING);
    glPushAttrib(GL_ENABLE_BIT);
    glBegin(GL_LINES);
    // X lines
    for (int y = 0; y <= dim.y; y++)
        for (int z = 0; z <= dim.z; z++)
        {
            glColor3f(0.5, y/(float)dim.y, z/(float)dim.z);
            glVertex3f(envMin.x, y*cellSize, z*cellSize);
            glVertex3f(envMax.x, y*cellSize, z*cellSize);
        }

    //// Y axis
    for (int x = 0; x <= dim.x; x++)
        for (int z = 0; z <= dim.z; z++)
        {
        glColor3f(x / (float)dim.x, 0.5, z / (float)dim.z);
            glVertex3f(x*cellSize, envMin.y, z*cellSize);
            glVertex3f(x*cellSize, envMax.y, z*cellSize);
        }

    //// Z axis
    for (int x = 0; x <= dim.x; x++)
        for (int y = 0; y <= dim.y; y++)
        {
        glColor3f(x / (float)dim.x, y / (float)dim.y, 0.5);
            glVertex3f(x*cellSize, y*cellSize, envMin.z);
            glVertex3f(x*cellSize, y*cellSize, envMax.z);
        }
    glEnd();
    GL_CHECK();
    glPopAttrib();
    glPopMatrix();
    GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
}
