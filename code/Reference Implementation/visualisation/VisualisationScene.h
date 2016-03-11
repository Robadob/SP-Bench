#pragma once

#include <stdio.h>
#include "gl/glew.h"
#include "SDL/SDL.h"
#include "SDL/SDL_opengl.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Camera.h"
#include "Shaders.h"

class VisualisationScene
{
public:

    VisualisationScene(Camera* camera = nullptr);
    virtual ~VisualisationScene(){};

    /*
    Update any scene animations here
    */
    virtual void update() = 0;
    /*
    Reload any shaders/models here
    */
    virtual void reload() = 0;
    /*
    Perform any render calls here
    */
    virtual void render(glm::mat4 projection) = 0;
    /*
    Generate any scene content
    */
    virtual void generate() {};

protected:
    Camera* camera;

};

