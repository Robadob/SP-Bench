#include "VisualisationScene.h"

VisualisationScene::VisualisationScene(Camera* camera)
    : camera(camera)
{
    this->generate();
}
