#ifndef __Neighbourhood_cuh__
#define __Neighbourhood_cuh__

#include "results.h"
#include "header.cuh"
#include "glm/gtx/component_wise.hpp"

#ifdef _GL
#include "visualisation/GLcheck.h"
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
#if defined(STRIPS) && !defined (MORTON)
#if defined (_3D)
    glm::ivec2 relative;
	glm::ivec3 location;
#endif
#if !defined (_3D)
    int relative;
    glm::ivec2 location;
#endif
#else
#if defined (_3D)
	glm::ivec3 relative;
	glm::ivec3 location;
#endif
#if !defined (_3D)
	glm::ivec2 relative;
	glm::ivec2 location;
#endif
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
	DIMENSIONS_VEC location;
};


struct LocationMessages
{
public:
    float *locationX;
	float *locationY;
#ifdef _3D
	float *locationZ;
#endif
#if defined(_GL) || defined(_DEBUG)
    float *count;
#endif
    __device__ LocationMessage *getFirstNeighbour(DIMENSIONS_VEC location);
    __device__ LocationMessage *getNextNeighbour(LocationMessage *message);

private:
	__device__ bool nextBin(LocationMessage *sm_message);
    //Load the next desired message into shared memory
	__device__ LocationMessage *loadNextMessage(LocationMessage *message);

};

extern __host__ __device__ unsigned int morton3D(DIMENSIONS_IVEC pos);
extern __host__ __device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos);
class SpatialPartition
{
public:
    SpatialPartition(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad);
    ~SpatialPartition();
    //Getters
    unsigned int *d_getPBM() { return d_PBM; }
    LocationMessages *d_getLocationMessages() { return d_locationMessages; }
    LocationMessages *d_getLocationMessagesSwap() { return d_locationMessages_swap; }
    unsigned int getLocationCount() { return locationMessageCount; }
    //Setters
    void SpatialPartition::setLocationCount(unsigned int);
    //Util
    void buildPBM();
    void swap();
    DIMENSIONS_IVEC getGridDim() const { return gridDim; }
    DIMENSIONS_VEC getEnvironmentMin() const { return environmentMin; }
    DIMENSIONS_VEC getEnvironmentMax() const { return environmentMax; }
    float getCellSize() const { return interactionRad;  }
#ifdef _DEBUG
    bool isValid(DIMENSIONS_IVEC bin) const;
	DIMENSIONS_IVEC getPos(unsigned int hash);
	int getHash(DIMENSIONS_IVEC gridPos);
    void assertSearch();
    void launchAssertPBMIntegerity();
#endif
#ifdef _GL
    const GLuint *SpatialPartition::getLocationTexNames() const { return gl_tex; }
    GLuint SpatialPartition::getCountTexName() const { return gl_tex_count; }
#endif
private:
	void setBinCount();
    //Allocators
    void deviceAllocateLocationMessages(LocationMessages **d_locMessage, LocationMessages *hd_locMessage);
    void deviceAllocatePBM(unsigned int **d_PBM_t);
	void deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals);
#ifndef THRUST
	void deviceAllocateCUBTemp(void **d_CUB_temp, size_t &d_cub_temp_bytes);
#endif
    void deviceAllocateTextures();
    void deviceAllocateTexture_float(unsigned int i);
    void deviceAllocateTexture_int();//allocate pbm tex
#ifdef _GL
    void deviceAllocateGLTexture_float(unsigned int i);
    void deviceAllocateGLTexture_float2();//Allocate float to be passed to shader
    GLuint gl_tex_count;
    GLuint gl_tbo_count;
    cudaGraphicsResource_t gl_gRes_count;
    cudaTextureObject_t tex_location_count;
    float *tex_location_ptr_count;
#endif
    //Deallocators
    void deviceDeallocateLocationMessages(LocationMessages *d_locMessage, LocationMessages hd_locMessage);
    void deviceDeallocatePBM(unsigned int *d_PBM_t);
    void deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals);
	void deviceDeallocateTextures();
#ifndef THRUST
	void deviceDeallocateCUBTemp(void *d_CUB_temp);
#endif
    
    void fillTextures();

    unsigned int getBinCount() const;//(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))
	unsigned int binCount;
	unsigned int binCountMax;
	const unsigned int maxAgents;
    float interactionRad;//Radius of agent interaction
    //Kernel launchers
    void launchHashLocationMessages();
    void launchReorderLocationMessages();
    //Device pointers
    unsigned int *d_PBM; //Each int points to the first message index of the relevant bin index
    LocationMessages *d_locationMessages, *d_locationMessages_swap;
    LocationMessages hd_locationMessages, hd_locationMessages_swap;
    //Device primitive pointers (used by thrust/CUB methods)
    unsigned int *d_keys;
    unsigned int *d_vals;
#ifndef THRUST
    unsigned int *d_keys_swap;
    unsigned int *d_vals_swap;
	void *d_CUB_temp_storage;
	size_t d_CUB_temp_storage_bytes;
#endif
    //Local copies of device constants
    unsigned int locationMessageCount;

    const DIMENSIONS_VEC  environmentMin;
    const DIMENSIONS_VEC  environmentMax;
    const DIMENSIONS_IVEC gridDim;
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
#if defined(_GL) || defined(_DEBUG)
extern __device__ __constant__ LocationMessages *d_locationMessagesA;
extern __device__ __constant__ LocationMessages *d_locationMessagesB;
#endif
extern __device__ __constant__ cudaTextureObject_t d_tex_location[DIMENSIONS];
extern __device__ __constant__ cudaTextureObject_t d_tex_PBM;

#endif
