#include "ParticleScene.h"

#include "visualisation/GLcheck.h"

ParticleScene::ParticleScene(Visualisation &visualisation, std::shared_ptr<CoreModel> model)
    : Scene(visualisation)
    , icosphere(std::make_shared<Entity>(Stock::Models::ICOSPHERE, 1.0f, std::make_shared<Shaders>("../shaders/instanced.vert","../shaders/instanced.frag")))
    , count(model->agentMax)
    , model(model)
    , drawPBM(true)
{
    registerEntity(icosphere);
    this->visualisation.setWindowTitle("Circles Benchmark");
    this->visualisation.setRenderAxis(true);
    setTex(model->getPartition()->getLocationTexNames());
}
ParticleScene::~ParticleScene()
{

}
/*
Sets the texture buffers that the shaders should use
@param Pointer to array of DIMENSIONS texture names
*/
void ParticleScene::setTex(const GLuint* tex)
{
    memcpy(this->tex, tex, DIMENSIONS*sizeof(GLuint));
    this->icosphere->getShaders()->addTextureUniform(tex[0], "tex_locX");
    this->icosphere->getShaders()->addTextureUniform(tex[1], "tex_locY");
#ifdef _3D
    this->icosphere->getShaders()->addTextureUniform(tex[2], "tex_locZ");
#endif
    this->icosphere->getShaders()->addTextureUniform(model->getPartition()->getCountTexName(), "tex_count");
}
/*
Sets the number of instances to be rendered
*/
void ParticleScene::setCount(unsigned int count)
{
    this->count = count;
}
/*
Steps the model
@note This is currently done externally
*/
void ParticleScene::update(unsigned int frameTime)
{
    //Update agent count
    setCount(this->model->getPartition()->getLocationCount());
}
/*
Steps the model
@note This is currently done externally
*/
bool ParticleScene::keypress(SDL_Keycode keycode, int x, int y)
{
    switch (keycode)
    {
    case SDLK_p:
        this->drawPBM = !this->drawPBM;
        break;
    default:
        //Only permit the keycode to be processed if we haven't handled personally
        return true;
    }
    return false;
}
/*
Refreshes shaders
*/
void ParticleScene::reload()
{
//Restart model?
}
/*
Renders a frame
*/
void ParticleScene::render()
{
    if (this->count <= 0)
        return;
    if (this->drawPBM)
        this->renderPBM();
    this->icosphere->renderInstances(count);
}
/*
Renders a frame
*/
void ParticleScene::renderPBM()
{
    DIMENSIONS_IVEC dim = this->model->getPartition()->getGridDim();
    DIMENSIONS_VEC envMin = this->model->getPartition()->getEnvironmentMin();
    DIMENSIONS_VEC envMax = this->model->getPartition()->getEnvironmentMax();
    float cellSize = this->model->getPartition()->getCellSize();
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
