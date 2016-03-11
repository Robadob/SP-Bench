#include "Visualisation.h"//Unused

#include <math.h>
#include <string>
#include <sstream>
#include "../GLcheck.h"


#define FOVY 60.0f
#define NEAR_CLIP 0.001f
#define FAR_CLIP 500.0f
#define DELTA_THETA_PHI 0.01f
#define MOUSE_SPEED 0.001f
#define SHIFT_MULTIPLIER 5.0f

#define MOUSE_SPEED_FPS 0.05f
#define DELTA_MOVE 0.1f
#define DELTA_STRAFE 0.1f
#define DELTA_ASCEND 0.1f
#define DELTA_ROLL 0.01f
#define ONE_SECOND_MS 1000
#define VSYNC 1

template<class T>
Visualisation<T>::Visualisation(char* windowTitle, int windowWidth, int windowHeight)
    : isInitialised(false)
    , quit(false)
    , windowTitle(windowTitle)
    , windowWidth(windowWidth)
    , windowHeight(windowHeight)
    , camera(glm::vec3(10, 10, 10))
    , renderAxisState(false)
    , axis(0.5)
    , renderThread(0)
{
    this->isInitialised = this->init();
    skybox = 0;// new Skybox();
}

template<class T>
Visualisation<T>::~Visualisation(){
    delete this->scene;
    //delete this->skybox;
}

template<class T>
bool Visualisation<T>::init(){
    bool result = true;

    SDL_Init(SDL_INIT_VIDEO);

    this->window = SDL_CreateWindow
        (
        this->windowTitle,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        this->windowWidth,
        this->windowHeight,
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL //| SDL_WINDOW_BORDERLESS
        );

    if (this->window == NULL){
        printf("window failed to init");
        result = false;
    }
    else {
        SDL_GetWindowPosition(window, &this->windowedBounds.x, &this->windowedBounds.y);
        SDL_GetWindowSize(window, &this->windowedBounds.w, &this->windowedBounds.h);

        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 5);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);

        // Get context
        this->context = SDL_GL_CreateContext(window);

        // Enable MSAA
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8);

        int swapIntervalResult = SDL_GL_SetSwapInterval(VSYNC);
        if (swapIntervalResult == -1){
            printf("Swap Interval Failed: %s\n", SDL_GetError());
        }

        // Init glew.
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            /* Problem: glewInit failed, something is seriously wrong. */
            fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
            exit(1);
        }

        // Create the scene - need to be done after glew is init
        this->scene = new T(&this->camera);


        // Setup gl stuff
        GL_CALL(glEnable(GL_DEPTH_TEST));
        GL_CALL(glEnable(GL_CULL_FACE));
        GL_CALL(glCullFace(GL_BACK));
        GL_CALL(glShadeModel(GL_SMOOTH));
        GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
        GL_CALL(glEnable(GL_LIGHTING));
        GL_CALL(glEnable(GL_LIGHT0));
        GL_CALL(glEnable(GL_COLOR_MATERIAL));
        GL_CALL(glEnable(GL_NORMALIZE));

        // Setup the projection matrix
        this->resizeWindow();
    }
    return result;
}

template<class T>
void Visualisation<T>::handleKeypress(SDL_Keycode keycode, int x, int y){

    switch (keycode){
    case SDLK_ESCAPE:
        this->setQuit(true);
        break;
    case SDLK_F11:
        this->toggleFullScreen();
        break;
    case SDLK_F5:
        //this->skybox->reload();
        this->scene->reload();
        break;
    default:
        // Do nothing?
        break;
    }
}


template<class T>
void Visualisation<T>::close(){
    SDL_GL_DeleteContext(this->context);
    SDL_DestroyWindow(this->window);
    this->window = NULL;
    SDL_Quit();
    if (renderThread)
    {
        renderThread->join();
        renderThread = 0;
    }
}

template<class T>
void Visualisation<T>::run(){
    if (!this->isInitialised){
        printf("Visulisation not initialised yet.");
    }
    else {
        SDL_Event e;
        SDL_StartTextInput();
        while (!this->quit){
            renderStep();
        }
        SDL_StopTextInput();

    }

    this->close();
}
template<class T>
void Visualisation<T>::renderStep()
{

    //SDL_GL_MakeCurrent(this->window, this->context);
    SDL_Event e;
    SDL_StartTextInput();
    GL_CHECK();
    // Update the fps
    this->updateFPS();

    // Handle continues press keys (movement)
    const Uint8 *state = SDL_GetKeyboardState(NULL);
    float turboMultiplier = state[SDL_SCANCODE_LSHIFT] ? SHIFT_MULTIPLIER : 1.0f;
    if (state[SDL_SCANCODE_W]) {
        this->camera.move(DELTA_MOVE*turboMultiplier);
    }
    if (state[SDL_SCANCODE_A]) {
        this->camera.strafe(-DELTA_STRAFE*turboMultiplier);
    }
    if (state[SDL_SCANCODE_S]) {
        this->camera.move(-DELTA_MOVE*turboMultiplier);
    }
    if (state[SDL_SCANCODE_D]) {
        this->camera.strafe(DELTA_STRAFE*turboMultiplier);
    }
    if (state[SDL_SCANCODE_Q]) {
        this->camera.roll(-DELTA_ROLL);
    }
    if (state[SDL_SCANCODE_E]) {
        this->camera.roll(DELTA_ROLL);
    }
    if (state[SDL_SCANCODE_SPACE]) {
        this->camera.ascend(DELTA_ASCEND*turboMultiplier);
    }
    if (state[SDL_SCANCODE_LCTRL]) {
        this->camera.ascend(-DELTA_ASCEND*turboMultiplier);
    }


    // handle each event on the queue
    while (SDL_PollEvent(&e) != 0){
        switch (e.type){
        case SDL_QUIT:
            this->setQuit(true);
            break;
        case SDL_KEYDOWN:
        {
            int x = 0;
            int y = 0;
            SDL_GetMouseState(&x, &y);
            this->handleKeypress(e.key.keysym.sym, x, y);
        }
            break;
            //case SDL_MOUSEWHEEL:

            //break;

        case SDL_MOUSEMOTION:
            this->handleMouseMove(e.motion.xrel, e.motion.yrel);
            break;
        case SDL_MOUSEBUTTONDOWN:
            this->toggleMouseMode();
            break;

        }
    }

    // update
    this->scene->update();
    // render
    this->clearFrame();
    //this->skybox->render(&camera, this->frustum);
    this->defaultProjection();
    if (this->renderAxisState)
        this->axis.render();
    this->defaultLighting();
    this->scene->render(this->frustum);
    // update the screen
    SDL_GL_SwapWindow(window);

    SDL_StopTextInput();
}
template<class T>
void Visualisation<T>::runAsync()
{
    SDL_GL_MakeCurrent(this->window, NULL);
    renderThread = new std::thread(&Visualisation<T>::run, this);
}
template<class T>
void Visualisation<T>::clearFrame()
{
    GL_CALL(glViewport(0, 0, this->windowWidth, this->windowHeight));
    GL_CALL(glClearColor(0, 0, 0, 1));
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}
template<class T>
void Visualisation<T>::defaultProjection()
{
    GL_CALL(glEnable(GL_CULL_FACE));
    GL_CALL(glMatrixMode(GL_PROJECTION));
    GL_CALL(glLoadIdentity());
    GL_CALL(glLoadMatrixf(glm::value_ptr(this->frustum)));
    //GL_CALL(glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(this->frustum)));
    GL_CALL(glMatrixMode(GL_MODELVIEW));
    GL_CALL(glLoadIdentity());
    GL_CALL(glLoadMatrixf(glm::value_ptr(this->camera.view())));
    //GL_CALL(glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(this->camera.view())));
}
template<class T>
void Visualisation<T>::defaultLighting()
{
    GL_CALL(glEnable(GL_LIGHT0));
    GL_CALL(glm::vec3 eye = this->camera.getEye());
    float lightPosition[4] = { eye.x, eye.y, eye.z, 1 };
    float amb[4] = { 0.8f, 0.8f, 0.8f, 1 };
    float diffuse[4] = { 0.2f, 0.2f, 0.2f, 1 };
    float white[4] = { 1, 1, 1, 1 };
    GL_CALL(glLightfv(GL_LIGHT0, GL_POSITION, lightPosition));
    GL_CALL(glLightfv(GL_LIGHT0, GL_AMBIENT, amb));
    GL_CALL(glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse));
    GL_CALL(glLightfv(GL_LIGHT0, GL_SPECULAR, white));

    // Spotlight stuff
    //float angle = 180.0f;
    //glm::vec3 look = this->camera.getLook();
    // float direction[4] = { look.x, look.y, look.z, 0 };
    //GL_CALL(glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, angle));
    //GL_CALL(glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, direction));
}
template<class T>
void Visualisation<T>::renderAxis()
{
    glPushMatrix();
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    //Axis
    glLineWidth(1);
    glPushMatrix();
    glColor4f(1.0, 1.0, 1.0, 1.0);//White-x
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(100, 0, 0);
    glEnd();
    glPopMatrix();
    glPushMatrix();
    glColor4f(0.0, 1.0, 0.0, 1.0);//Green-y
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 100, 0);
    glEnd();
    glPopMatrix();
    glPushMatrix();
    glColor4f(0.0, 0.0, 1.0, 1.0);//Blue-z
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 100);
    glEnd();
    glPopMatrix();
    glPopMatrix();
    GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
}
template<class T>
void Visualisation<T>::setRenderAxis(bool state)
{
    this->renderAxisState = state;
}
template<class T>
char* Visualisation<T>::getWindowTitle(){
    return this->windowTitle;
}

template<class T>
void Visualisation<T>::setWindowTitle(char* windowTitle){
    this->windowTitle = windowTitle;
}

template<class T>
void Visualisation<T>::setQuit(bool quit){
    this->quit = quit;
}

template<class T>
void Visualisation<T>::toggleFullScreen(){
    if (this->isFullscreen()){
        // Update the window using the stored windowBounds
        SDL_SetWindowBordered(this->window, SDL_TRUE);
        SDL_SetWindowSize(this->window, this->windowedBounds.w, this->windowedBounds.h);
        SDL_SetWindowPosition(this->window, this->windowedBounds.x, this->windowedBounds.y);
    }
    else {
        // Store the windowedBounds for later
        SDL_GetWindowPosition(window, &this->windowedBounds.x, &this->windowedBounds.y);
        SDL_GetWindowSize(window, &this->windowedBounds.w, &this->windowedBounds.h);
        // Get the window bounds for the current screen
        int displayIndex = SDL_GetWindowDisplayIndex(this->window);
        SDL_Rect displayBounds;
        SDL_GetDisplayBounds(displayIndex, &displayBounds);
        // Update the window
        SDL_SetWindowBordered(this->window, SDL_FALSE);
        SDL_SetWindowPosition(this->window, displayBounds.x, displayBounds.y);
        SDL_SetWindowSize(this->window, displayBounds.w, displayBounds.h);
    }
    this->resizeWindow();
}

template<class T>
void Visualisation<T>::toggleMouseMode(){
    if (SDL_GetRelativeMouseMode()){
        SDL_SetRelativeMouseMode(SDL_FALSE);
    }
    else {
        SDL_SetRelativeMouseMode(SDL_TRUE);
    }
}

template<class T>
void Visualisation<T>::resizeWindow(){
    // Use the sdl drawable size
    SDL_GL_GetDrawableSize(this->window, &this->windowWidth, &this->windowHeight);

    float fAspect = static_cast<float>(this->windowWidth) / static_cast<float>(this->windowHeight);
    double fovy = FOVY;

    glViewport(0, 0, this->windowWidth, this->windowHeight);
    float top = static_cast<float>(tan(glm::radians(fovy * 0.5)) * NEAR_CLIP);
    float bottom = -top;
    float left = fAspect * bottom;
    float right = fAspect * top;
    this->frustum = glm::frustum<float>(left, right, bottom, top, NEAR_CLIP, FAR_CLIP);
}

template<class T>
void Visualisation<T>::handleMouseMove(int x, int y){
    if (SDL_GetRelativeMouseMode()){
        this->camera.turn(x * MOUSE_SPEED, y * MOUSE_SPEED);
    }
}

template<class T>
bool Visualisation<T>::isFullscreen(){
    // Use window borders as a toggle to detect fullscreen.
    return (SDL_GetWindowFlags(this->window) & SDL_WINDOW_BORDERLESS) == SDL_WINDOW_BORDERLESS;
}

// Super simple fps counter imoplementation
template<class T>
void Visualisation<T>::updateFPS(){
    // Update the current time
    this->currentTime = SDL_GetTicks();
    // Update frame counter
    this->frameCount += 1;
    // If it's been more than a second, do something.
    if (this->currentTime > this->previousTime + ONE_SECOND_MS){
        // Calculate average fps.
        double fps = this->frameCount / double(this->currentTime - this->previousTime) * ONE_SECOND_MS;
        // Update the title to include FPS at the end.
        std::ostringstream newTitle;
        newTitle << this->windowTitle << " (" << std::to_string(static_cast<int>(std::ceil(fps))) << " fps)";
        SDL_SetWindowTitle(this->window, newTitle.str().c_str());

        // reset values;
        this->previousTime = this->currentTime;
        this->frameCount = 0;
    }
}

template<class T>
Camera *Visualisation<T>::getCamera()
{
    return &this->camera;
}
template<class T>
T *Visualisation<T>::getScene() const
{
    return this->scene;
}
