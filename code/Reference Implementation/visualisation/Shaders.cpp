#define  _CRT_SECURE_NO_WARNINGS
#include "Shaders.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <GL/glew.h>

#define EXIT_ON_ERROR 0



Shaders::Shaders(char* vertexShaderPath, char* fragmentShaderPath, char* geometryShaderPath) 
    : compileSuccessFlag(true)
    , programId(0)
{
    // This is a bit pointless but why not.
    this->vertexShaderPath = vertexShaderPath;
    this->fragmentShaderPath = fragmentShaderPath;
    this->geometryShaderPath = geometryShaderPath;
    // Create the shaders
    this->createShaders();
    //this->useProgram();
    this->checkGLError(__FILE__, __LINE__);

}

Shaders::~Shaders(){
    this->destroyProgram();
}

bool Shaders::hasVertexShader(){
    return this->vertexShaderPath!=0;
}

bool Shaders::hasFragmentShader(){
    return this->fragmentShaderPath != 0;
}

bool Shaders::hasGeometryShader(){
    return this->geometryShaderPath != 0;
}


void Shaders::createShaders(){
    // Reset the flag
    this->compileSuccessFlag = true;
    // Load shader files
    const char* vertexSource = loadShaderSource(this->vertexShaderPath);
    const char* fragmentSource = loadShaderSource(this->fragmentShaderPath);
    const char* geometrySource = loadShaderSource(this->geometryShaderPath);
    //Check for shaders that didn't load correctly
    if (vertexSource == 0) this->vertexShaderPath = 0;
    if (fragmentSource == 0) this->fragmentShaderPath = 0;
    if (geometrySource == 0) this->geometryShaderPath = 0;

    // For each shader we have been able to read.
    // Create the empty shader handle
    // Atrtempt to compile the shader
    // Check compilation
    // If it fails, bail out.
    if (hasVertexShader()){
        this->vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
        //printf("\n>>vsi: %d\n", this->vertexShaderId);
        glShaderSource(this->vertexShaderId, 1, &vertexSource, 0);
        this->checkGLError(__FILE__, __LINE__);
        glCompileShader(this->vertexShaderId);
        this->checkGLError(__FILE__, __LINE__);
        this->checkShaderCompileError(this->vertexShaderId, this->vertexShaderPath);

    }
    if (hasFragmentShader()){
        this->fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(this->fragmentShaderId, 1, &fragmentSource, 0);
        this->checkGLError(__FILE__, __LINE__);
        glCompileShader(this->fragmentShaderId);
        this->checkGLError(__FILE__, __LINE__);
        this->checkShaderCompileError(this->fragmentShaderId, this->fragmentShaderPath);
    }
    if (hasGeometryShader()){
        this->geometryShaderId = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(this->geometryShaderId, 1, &geometrySource, 0);
        this->checkGLError(__FILE__, __LINE__);
        glCompileShader(this->geometryShaderId);
        this->checkGLError(__FILE__, __LINE__);
        this->checkShaderCompileError(this->geometryShaderId, this->geometryShaderPath);
    }

    // Only attempt to link the program if the compilation of each individual shader was successful.
    if (this->compileSuccessFlag){

        // Create the program
        int newProgramId = glCreateProgram();
        this->checkGLError(__FILE__, __LINE__);

        // Attach each included shader
        if (this->hasVertexShader()){
            glAttachShader(newProgramId, this->vertexShaderId);
            this->checkGLError(__FILE__, __LINE__);
        }
        if (this->hasFragmentShader()){
            glAttachShader(newProgramId, this->fragmentShaderId);
            this->checkGLError(__FILE__, __LINE__);
        }
        if (this->hasGeometryShader()){
            glAttachShader(newProgramId, this->geometryShaderId);
            this->checkGLError(__FILE__, __LINE__);
        }
        // Link the program and Ensure the program compiled correctly;
        glLinkProgram(newProgramId);
        this->checkGLError(__FILE__, __LINE__);

        this->checkProgramCompileError(newProgramId);
        this->checkGLError(__FILE__, __LINE__);
        // If the program compiled ok, then we update the instance variable (for live reloading
        if (this->compileSuccessFlag){
            // Destroy the old program
            this->destroyProgram();
            this->checkGLError(__FILE__, __LINE__);
            // Update the class var for the next usage.
            this->programId = newProgramId;
            this->checkGLError(__FILE__, __LINE__);
        }
    }
    this->checkGLError(__FILE__, __LINE__);

    // Clean up any shaders
    this->destroyShaders();
    // and delete sources
    delete vertexSource;
    delete fragmentSource;
    delete geometrySource;
}

void Shaders::reloadShaders(){
    fprintf(stdout, "Reloading Shaders\n");
    this->createShaders();
}

void Shaders::useProgram(){
    glUseProgram(this->programId);

    glBindAttribLocation(this->programId, 0, "in_position");
    //this->checkGLError(__FILE__, __LINE__);

   // glBindAttribLocation(this->programId, 1, "in_normal");
    //this->checkGLError(__FILE__, __LINE__);

    //GLfloat model[16];
    //glGetFloatv(GL_MODELVIEW_MATRIX, model);
    //this->setUniformMatrix4fv(0, model);
    //glUniformMatrix4fv(1, 1, GL_FALSE, model);
    //this->checkGLError(__FILE__, __LINE__);
    //glGetFloatv(GL_PROJECTION_MATRIX, model);
    //this->setUniformMatrix4fv(1, model);
    //glUniformMatrix4fv(2, 1, GL_FALSE, model);
   // this->checkGLError(__FILE__, __LINE__);
}

void Shaders::clearProgram(){
    glUseProgram(0);
}

int Shaders::getProgram() const
{
    return this->programId;
}
void Shaders::setUniformi(int location, int value){
    if (location >= 0){
        glUniform1i(location, value);
    }
}

void Shaders::setUniformMatrix4fv(int location, GLfloat* value){
    if (location >= 0){
        // Must be false and length with most likely just be 1. Can add an extra parameter version if required.
        glUniformMatrix4fv(location, 1, GL_FALSE, value);
    }
}

char* Shaders::loadShaderSource(char* file){
    // If file path is 0 it is being omitted. kinda gross
    if (file != 0){
        FILE* fptr = fopen(file, "rb");
        if (!fptr){
            printf("Shader not found: %s\n", file);
#if EXIT_ON_ERROR == 1
            //@todo exit maybe?
            system("pause"); // @temp for pausing on output.
            exit(1);
#endif
            return 0;
        }
        fseek(fptr, 0, SEEK_END);
        long length = ftell(fptr);
        char* buf = (char*)malloc(length + 1); // Allocate a buffer for the entire length of the file and a null terminator
        fseek(fptr, 0, SEEK_SET);
        fread(buf, length, 1, fptr);
        fclose(fptr);
        buf[length] = 0; // Null terminator
        return buf;
    }
    else {
        return 0;
    }
}


void Shaders::destroyShaders(){
    // Destroy the shaders and program
    if (this->hasVertexShader()){
        glDeleteShader(this->vertexShaderId);
        this->checkGLError(__FILE__, __LINE__);
    }
    if (this->hasFragmentShader()){
        glDeleteShader(this->fragmentShaderId);
        this->checkGLError(__FILE__, __LINE__);
    }
    if (this->hasGeometryShader()){
        glDeleteShader(this->geometryShaderId);
        this->checkGLError(__FILE__, __LINE__);
    }
}

void Shaders::destroyProgram(){
    if (programId!=0)
        glDeleteProgram(this->programId);
}

void Shaders::checkGLError(const char *file, int line){
    GLuint error = glGetError();
    if (error != GL_NO_ERROR)
    {
        const char* errMessage = (const char*)gluErrorString(error);
        fprintf(stderr, "%s(%i) OpenGL Error #%d: %s\n", file, line, error, errMessage);
    }
}

void Shaders::checkShaderCompileError(int shaderId, char* shaderPath){
    this->checkGLError(__FILE__, __LINE__);

    GLint status;
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE){
        // Get the length of the info log
        GLint len;
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &len);
        // Get the contents of the log message
        char* log = new char[len + 1];
        glGetShaderInfoLog(shaderId, len, &len, log);
        // Print the message
        printf("Shader compilation error (%s) :\n", shaderPath);
        printf("%s\n", log);
        delete log;
        this->compileSuccessFlag = false;
#if EXIT_ON_ERROR == 1
        //@todo exit maybe?
        system("pause"); // @temp for pausing on output.
        exit(1);
#endif
    }
    
}



void Shaders::checkProgramCompileError(int programId){
    int status;
    this->checkGLError(__FILE__, __LINE__);
    glGetProgramiv(programId, GL_LINK_STATUS, &status);
    this->checkGLError(__FILE__, __LINE__);
    if (status == GL_FALSE){
        // Get the length of the info log
        GLint len=0;
        glGetProgramiv(this->programId, GL_INFO_LOG_LENGTH, &len);
        // Get the contents of the log message
        char* log = new char[len + 1];
        glGetProgramInfoLog(this->programId, len, &len, log);
        // Print the message
        printf("Program compilation error:\n");
        printf("%s\n", log);
        this->compileSuccessFlag = false;
        this->checkGLError(__FILE__, __LINE__);
#if EXIT_ON_ERROR == 1
        //@todo exit maybe?
        system("pause"); // @temp for pausing on output.
        exit(1);
#endif
    }
}
