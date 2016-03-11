#ifndef GLCHECK_H
#define GLCHECK_H

#ifdef _GL
#include <cstdlib>
#include <cstdio>
#include <GL\glew.h>
//Cuda call
static void HandleGLError(const char *file, int line) {
    GLuint error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("%s(%i) GL Error Occurred;\n%s\n", file, line, gluErrorString(error));
        getchar();
        exit(1);
    }
}
//#define GL_CALL( err ) { #err, err ## ;HandleGLError(__FILE__, __LINE__))
#define GL_CALL( err ) err ;HandleGLError(__FILE__, __LINE__)
#define GL_CHECK() (HandleGLError(__FILE__, __LINE__))
#endif

#endif
