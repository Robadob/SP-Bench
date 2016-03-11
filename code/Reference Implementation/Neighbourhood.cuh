#ifndef __Neighbourhood_cuh__
#define __Neighbourhood_cuh__

#include "header.cuh"
#include "glm/gtx/component_wise.hpp"

#ifdef _GL
#include <GL\glew.h>
#endif


#ifdef _3D
#define DIMENSIONS 3
#define DIMENSIONS_VEC glm::vec3
#define DIMENSIONS_IVEC glm::ivec3
#else
#define DIMENSIONS 2
#define DIMENSIONS_VEC glm::vec2
#define DIMENSIONS_IVEC glm::ivec2
#endif
//#include "NeighbourhoodKernels.cuh"

/**
* Structure used to maintain search state in each thread during neigbourhood search
**/
struct BinState
{
#ifdef _3D
    glm::ivec2 relative;
    glm::ivec3 location;
#else
    int relative;
    glm::vec2 location;
#endif
    unsigned int binIndexMax;//Last pbm index
    unsigned int binIndex;//Current loaded message pbm index
};

/**
 * Structure data is returned in when performing neighbour search
**/
struct LocationMessage
{
    unsigned int id;
    BinState state;
#ifdef _3D
    glm::vec3 location;
#else
    glm::vec2 location;
#endif
};


struct LocationMessages
{
public:
    float *locationX;
    float *locationY;
#ifdef _3D
    float *locationZ;
#endif

    __device__ LocationMessage *getFirstNeighbour(DIMENSIONS_VEC location);
    __device__ LocationMessage *getNextNeighbour(LocationMessage *message);

private:
    __device__ bool nextBin();
    //Load the next desired message into shared memory
    __device__ LocationMessage *loadNextMessage();

};
class SpatialPartition
{
public:
    SpatialPartition(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad);
    ~SpatialPartition();
    //Getters
    inline unsigned int *d_getPBM() { return d_PBM; }
    inline LocationMessages *d_getLocationMessages() { return d_locationMessages; }
    inline LocationMessages *d_getLocationMessagesSwap() { return d_locationMessages_swap; }
    inline unsigned int getLocationCount(){ return locationMessageCount; }
    //Setters
    void SpatialPartition::setLocationCount(unsigned int);
    //Util
    void buildPBM();
    void swap();
#ifdef _GL
    GLuint *SpatialPartition::getLocationTexNames()
    {
        return gl_tex;
    }
#endif
private:
    //Allocators
    void deviceAllocateLocationMessages(LocationMessages **d_locMessage);
    void deviceAllocatePBM(unsigned int **d_PBM_t);
    void deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals);
    void deviceAllocateTextures();
    void deviceAllocateTexture_float(unsigned int i);
    void deviceAllocateTexture_int();//allocate pbm tex
#ifdef _GL
    void deviceAllocateGLTexture_float(unsigned int i);
#endif
    //Deallocators
    void deviceDeallocateLocationMessages(LocationMessages *d_locMessage);
    void deviceDeallocatePBM(unsigned int *d_PBM_t);
    void deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals);
    void deviceDeallocateTextures();
    
    void fillTextures();

    unsigned int getBinCount();//(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))
    const unsigned int maxAgents;
    float interactionRad;//Radius of agent interaction
    //Kernel launchers
    void launchHashLocationMessages();
    void launchReorderLocationMessages();
    //Device pointers
    unsigned int *d_PBM; //Each int points to the end index of the relevant bin index
    LocationMessages *d_locationMessages;
    LocationMessages *d_locationMessages_swap;
    //Device primitive pointers (used by thrust/CUB methods)
    unsigned int *d_keys;
    unsigned int *d_vals;
#ifndef THRUST
    unsigned int *d_keys_swap;
    unsigned int *d_vals_swap;
#endif
    //Local copies of device constants
    unsigned int locationMessageCount;

    const DIMENSIONS_IVEC gridDim;
    const DIMENSIONS_VEC  environmentMin;
    const DIMENSIONS_VEC  environmentMax;
#ifdef _DEBUG
    unsigned int PBM_isBuilt;
#endif
    //Textures
    cudaTextureObject_t tex_location[DIMENSIONS];
    float* tex_loc_ptr[DIMENSIONS];
    cudaTextureObject_t tex_PBM;
    unsigned int* tex_PBM_ptr;
    //GL Textures
#ifdef _GL
    GLuint gl_tex[DIMENSIONS];
    GLuint gl_tbo[DIMENSIONS];
    cudaGraphicsResource_t gl_gRes[DIMENSIONS];
#endif
};

//Device constants
extern __device__ __constant__ unsigned int d_locationMessageCount;
extern __device__ __constant__ unsigned int d_binCount;
extern __device__ __constant__ float d_interactionRad;

extern __device__ __constant__ DIMENSIONS_IVEC d_gridDim;
extern __device__ __constant__ DIMENSIONS_VEC d_gridDim_float;
extern __device__ __constant__ DIMENSIONS_VEC  d_environmentMin;
extern __device__ __constant__ DIMENSIONS_VEC  d_environmentMax;

#ifdef _DEBUG
extern __device__ __constant__ unsigned int d_PBM_isBuilt;
#endif

extern __device__ __constant__ cudaTextureObject_t d_tex_location[DIMENSIONS];
extern __device__ __constant__ cudaTextureObject_t d_tex_PBM;

#endif
