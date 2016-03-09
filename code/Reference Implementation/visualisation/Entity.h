#ifndef __Entity_h__
#define __Entity_h__

#include "Material.h"
#include "GL\glew.h"
#include <glm\glm.hpp>

class Entity
{
public:
    /**
     * @param modelPath Path to .obj format model file
     * @param modelScale World size to scale the longest direction (in the x, y or z) axis of the model to fit
    **/
    Entity(const char *modelPath, float modelScale=1.0);
    ~Entity();
    /**
     * Calls the necessary code to render the entities model
    **/
    void render();
    void renderInstances(int instanceCount);
    void setColor(glm::vec3 color);
    void setLocation(glm::vec3 location);
    void clearMaterial();
protected:
    //World scale of the longest side (in the axis x, y or z)
    const float SCALE;
    //Model vertex and face counts
    int v_count, f_count;
    //Primitive data
    glm::vec3 *vertices, *normals;
    glm::ivec3 *faces;
    //Vertex Buffer Objects for rendering
    GLuint vertices_vbo;
    GLuint faces_vbo;
    // Material properties
    Material *material;
    //Color
    glm::vec3 color;
    //Location
    glm::vec3 location;
    

    /**
     * Creates a vertex buffer object of the specified size
    **/
    void createVertexBufferObject(GLuint *vbo, GLenum target, GLuint size);
    /**
     * Deallocates the provided vertex buffer
    **/
    void deleteVertexBufferObject(GLuint *vbo);
    /**
     * Binds model primitive data to the vertex buffer objects
    **/
    void bindVertexBufferData();
    /**
     * Loads and scales the specified model into this class's primitive storage
     * @note Loading of vertex normals was disabled to suit some models that have non matching counts of vertices and vertex normals
    **/
    void loadModelFromFile(const char *path, float modelScale);
    void loadMaterialFromFile(const char *objPath, const char *materialFilename, const char *materialName);
    /**
     * Allocates the storage for model primitives
    **/
    void allocateModel();
    /**
     * Deallocates the storage for model primitives
    **/
    void freeModel();
    /**
     * Scales the vertices to fit the provided scale
    **/
    void scaleModel(float modelScale);
    void checkGLError();
};

#endif