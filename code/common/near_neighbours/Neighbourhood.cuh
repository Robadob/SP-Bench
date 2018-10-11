#include <host_defines.h>
#ifndef __Neighbourhood_cuh__
#define __Neighbourhood_cuh__

#include "results.h"
#include "header.cuh"
#include "glm/gtx/component_wise.hpp"

#ifdef _GL
#include "visualisation/GLcheck.h"
#endif


#if defined(_3D) && !defined(DIMENSIONS)
#define DIMENSIONS 3
#define DIMENSIONS_VEC glm::vec3
#define DIMENSIONS_IVEC glm::ivec3
#define DIMENSIONS_IVEC_MINUS1 glm::ivec2
#elif defined(_2D) && !defined(DIMENSIONS)
#define DIMENSIONS 2
#define DIMENSIONS_VEC glm::vec2
#define DIMENSIONS_IVEC glm::ivec2
#define DIMENSIONS_IVEC_MINUS1 int
#endif
//#include "NeighbourhoodKernels.cuh"


/**
* Structure used to maintain search state in each thread during neigbourhood search
**/
struct BinState
{
    DIMENSIONS_IVEC location;
    unsigned int binIndexMax;//Last pbm index
    unsigned int binIndex;//Current loaded message pbm index
#ifdef STRIDED_MESSAGES
    unsigned int binOffset;//Temp for testing
#endif
    /**
    * Default: 6 bits
    * Strips: 4 bits
    * Modular: 15 bits
    * Hybrid : 11 bits
    * Due to structure alignment/packing rules, this will always round upto nearest 4 bytes
    * Pragma pack(1) allows BITFIELDS_V2 to reduce bytes to 1 alignment, however this won't change register usage
    **/
#ifdef BITFIELDS_V2
#ifdef _DEBUG
#define ASSERT_BITS(input,field) assert((input&~field)==0)
#else
#define ASSERT_BITS(a,b)
#endif
#if (defined(MODULAR_STRIPS)||defined(MODULAR))
public:
    __device__ unsigned char offsetX(){ return (offsetBits & (3 << 0)) >> 0;}
    __device__ unsigned char offsetY(){ return (offsetBits & (3 << 2)) >> 2;}
    __device__ unsigned char offsetZ(){ return (offsetBits & (3 << 4)) >> 4;}
    __device__ void offsetXpp(){ offsetX(offsetX() + 1); }
    __device__ void offsetYpp(){ offsetY(offsetY() + 1); }
    __device__ void offsetZpp(){ offsetZ(offsetZ() + 1); }
    __device__ bool cont(){return offsetBits&1<<7;}
    __device__ void offsetX(char x){ offsetBits = ((offsetBits & ~(3 << 0)) | (x << 0)); ASSERT_BITS(x,(3 << 0)); }//0b00000011
    __device__ void offsetY(char y){ offsetBits = ((offsetBits & ~(3 << 2)) | (y << 2)); ASSERT_BITS(y,(3 << 0)); }//0b00001100
    __device__ void offsetZ(char z){ offsetBits = ((offsetBits & ~(3 << 4)) | (z << 4)); ASSERT_BITS(z,(3 << 0)); }//0b00110000
    __device__ void cont(bool c)   { offsetBits = ((offsetBits & ~(1 << 7)) | (c << 7)); ASSERT_BITS(c,(1 << 0)); }//0b01000000
private:
    unsigned char offsetBits;
#endif
public:
    __device__ char relativeX(){ return ((relativeBits & (3 << 0)) >> 0)-2; }//Output in range -2 to +1
    __device__ char relativeY(){ return ((relativeBits & (3 << 2)) >> 2)-1; }//Output in range -1 to +1
    __device__ char relativeZ(){ return ((relativeBits & (3 << 4)) >> 4)-1; }//Output in range -1 to +1
    __device__ void relativeXpp(){ relativeX(relativeX() + 1); }
    __device__ void relativeYpp(){ relativeY(relativeY() + 1); }
    __device__ void relativeZpp(){ relativeZ(relativeZ() + 1); }
    __device__ void relativeX(char x){ relativeBits = ((relativeBits & ~(3 << 0)) | ((x + 2) << 0)); ASSERT_BITS((x + 2), (3 << 0)); }//0b00000011
    __device__ void relativeY(char y){ relativeBits = ((relativeBits & ~(3 << 2)) | ((y + 1) << 2)); ASSERT_BITS((y + 1), (3 << 0)); }//0b00001100
    __device__ void relativeZ(char z){ relativeBits = ((relativeBits & ~(3 << 4)) | ((z + 1) << 4)); ASSERT_BITS((z + 1), (3 << 0)); }//0b00110000
private:
    char relativeBits;
#else
#if (defined(MODULAR_STRIPS)||defined(MODULAR))
#if defined(_3D)
    char blockRelativeX : 2;
    char blockRelativeY : 2;
    unsigned char offsetX : 2;//0-3
    unsigned char offsetY : 2;//0-3
#if defined(MODULAR)
    char blockRelativeZ : 2;
    unsigned char offsetZ : 2;//0-3
#endif
#elif defined(_2D)
    char blockRelativeX : 2;
    unsigned char offsetX : 2;//0-3
#if defined(MODULAR)
    char blockRelativeY : 2;
    unsigned char offsetY : 2;//0-3
#endif
#endif
    unsigned short blockContinue : 1;
#else
    char relativeX : 3;//-2 to 1
#if defined(_3D) || !defined(STRIPS)
    char relativeY : 2;//-1 to 1
#if defined(_3D) && !defined(STRIPS)
    char relativeZ : 2;//-1 to 1
#endif
#endif
#endif
#endif
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
#ifdef AOS_MESSAGES
    DIMENSIONS_VEC *location;
#else
    float *locationX;
	float *locationY;
#ifdef _3D
	float *locationZ;
#endif
#endif
#if defined(_GL) || defined(_DEBUG)
    float *count;
#endif

#if !defined(SHARED_BINSTATE)
#if !(defined(MODULAR) || defined(MODULAR_STRIPS))
    __device__ void getFirstNeighbour(DIMENSIONS_VEC location, LocationMessage *message);
    __device__ void getFirstMessage(unsigned int binId, LocationMessage *message);
#endif
    __device__ bool getNextNeighbour(LocationMessage *message);
#if defined(MODULAR) || defined(MODULAR_STRIPS)
    __device__ void firstBin(DIMENSIONS_VEC location, LocationMessage *message);
    __device__ void selectBin(unsigned int id, LocationMessage *message);
    __device__ bool nextBin(LocationMessage *sm_message);
#endif
#else
#if !(defined(MODULAR) || defined(MODULAR_STRIPS))
    __device__ LocationMessage *getFirstNeighbour(DIMENSIONS_VEC location);
    __device__ LocationMessage *getFirstMessage(unsigned int binId);
#endif
    __device__ LocationMessage *getNextNeighbour(LocationMessage *message);
#if defined(MODULAR) || defined(MODULAR_STRIPS)
    __device__ LocationMessage *firstBin(DIMENSIONS_VEC location);
    __device__ LocationMessage *selectBin(DIMENSIONS_VEC location);
    __device__ bool nextBin(LocationMessage *sm_message);
#endif
#endif
private:
#if !(defined(MODULAR) || defined(MODULAR_STRIPS))
	__device__ bool nextBin(LocationMessage *sm_message);
#endif

#if !defined(SHARED_BINSTATE)
    __device__ bool loadNextMessage(LocationMessage *message);
#else
    //Load the next desired message into shared memory
	__device__ LocationMessage *loadNextMessage(LocationMessage *message);
#endif

};
/* //These structures would be used if we were passing more data with neighbours, than simply location
template<class T>
struct LocationMessageExt<T>
{
    unsigned int id;
    BinState state;
	DIMENSIONS_VEC location;
    T data;
};
template<class T>
struct LocationMessagesExt<T>
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
    T *data;
    __device__ LocationMessage *getFirstNeighbour(DIMENSIONS_VEC location);
    __device__ LocationMessage *getNextNeighbour(LocationMessageExt<T> *message);

private:
	__device__ bool nextBin(LocationMessageExt<T> *sm_message);
    //Load the next desired message into shared memory
	__device__ LocationMessage *loadNextMessage(LocationMessageExt<T> *message);

};
*/
extern __device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos);
class SpatialPartition
{
public:
    SpatialPartition(unsigned int binCount, unsigned int maxAgents);
    SpatialPartition(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad);
    ~SpatialPartition();
    //Getters
    unsigned int *d_getPBMIndex() { return d_PBM_index; }
    unsigned int *d_getPBMCount() { return d_PBM_count; }
    LocationMessages *d_getLocationMessages() { return d_locationMessages; }
    LocationMessages *d_getLocationMessagesSwap() { return d_locationMessages_swap; }
    unsigned int getLocationCount() { return locationMessageCount; }
    //Setters
    void setLocationCount(unsigned int);
    //Util
    PBM_Time buildPBM();
    virtual void swap();
    DIMENSIONS_IVEC getGridDim() const { return gridDim; }
    DIMENSIONS_VEC getEnvironmentMin() const { return environmentMin; }
    DIMENSIONS_VEC getEnvironmentMax() const { return environmentMax; }
    DIMENSIONS_VEC getEnvironmentDimensions() const { return environmentMax - environmentMin; }
    float getCellSize() const { return interactionRad; }
    bool isValid(DIMENSIONS_IVEC bin) const;
    DIMENSIONS_IVEC getPos(unsigned int hash);
    DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos);
    unsigned int getHash(DIMENSIONS_IVEC gridPos);
    static int requiredSM(int blockSize);
#ifdef _DEBUG
    void assertSearch();
    void launchAssertPBMIntegerity();
#endif
    NeighbourhoodStats getNeighbourhoodStats();
#ifdef _GL
    const GLuint *getLocationTexNames() const { return gl_tex; }
    GLuint getCountTexName() const { return gl_tex_count; }
#endif
protected:
	void setBinCount();
    //Allocators
    void deviceAllocateLocationMessages(LocationMessages **d_locMessage, LocationMessages *hd_locMessage);
    void deviceAllocatePBM(unsigned int **d_PBM_index_t, unsigned int **d_PBM_count_t);
	void deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals);
#ifndef THRUST
	void deviceAllocateCUBTemp(void **d_CUB_temp, size_t &d_cub_temp_bytes);
#endif
    void deviceAllocateTextures();

#if !defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES)
    void deviceAllocateTexture_float(unsigned int i);
#endif
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
    void deviceAllocateTexture_int();//allocate pbm tex
#endif
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
    void deviceDeallocatePBM(unsigned int *d_PBM_index_t, unsigned int *d_PBM_count_t);
    void deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals);
	void deviceDeallocateTextures();
#ifndef THRUST
	void deviceDeallocateCUBTemp(void *d_CUB_temp);
#endif
    
    void fillTextures();

    unsigned int getBinCount() const;//(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))
	unsigned int binCount;
	unsigned int binCountMax;
    unsigned int binCountBits;
#if defined(MORTON) || defined(HILBERT) || defined(PEANO) || defined(MORTON_COMPUTE)
    unsigned int gridExponent;//The exponent of the grid, when using morton/hilbert encoding, which requires 2^n grid dims (3^n for Peano)
#endif
	const unsigned int maxAgents;
    float interactionRad;//Radius of agent interaction
    //Kernel launchers
    virtual void launchReorderLocationMessages();
#ifdef ATOMIC_PBM
    void launchAtomicHistogram();
#else
    void launchHashLocationMessages();
#endif
    //Device pointers
    //Under atomic count is the count
    //Under non atomic, count is the PBM shifted along one
    unsigned int *d_PBM_index, *d_PBM_count; //Each index points to the first message index of the relevant bin index
    LocationMessages *d_locationMessages, *d_locationMessages_swap;
    LocationMessages hd_locationMessages, hd_locationMessages_swap;
    //Device primitive pointers (used when building PBM)
    unsigned int *d_keys;//Atomic PBM uses this as bin index
    unsigned int *d_vals;//Atomic PBM uses this as bin sub-index
#ifndef THRUST
#ifndef ATOMIC_PBM
    unsigned int *d_keys_swap;
    unsigned int *d_vals_swap;
#endif
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
#if defined(_GL) ||(!defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES))
    cudaTextureObject_t tex_location[DIMENSIONS];
    float* tex_loc_ptr[DIMENSIONS];
#endif
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
    cudaTextureObject_t tex_PBM_index;
    cudaTextureObject_t tex_PBM_count;
    unsigned int* tex_PBM_index_ptr;
    unsigned int* tex_PBM_count_ptr;
#endif
    //GL Textures
#ifdef _GL
    GLuint gl_tex[DIMENSIONS];
    GLuint gl_tbo[DIMENSIONS];
    cudaGraphicsResource_t gl_gRes[DIMENSIONS];
#endif
};
/*Not required by RDC=false
//Device constants
extern __device__ __constant__ unsigned int d_locationMessageCount;
extern __device__ __constant__ unsigned int d_binCount;
extern __device__ __constant__ float d_interactionRad;

extern __device__ __constant__ DIMENSIONS_IVEC d_gridDim;
extern __device__ __constant__ DIMENSIONS_VEC d_gridDim_float;
extern __device__ __constant__ DIMENSIONS_VEC  d_environmentMin;
extern __device__ __constant__ DIMENSIONS_VEC  d_environmentMax;

#if defined(GLOBAL_PBM) || defined(LDG_PBM)
extern __device__ __constant__  unsigned int *d_pbm;
#else
extern __device__ __constant__ cudaTextureObject_t d_tex_PBM;
#endif

#ifdef _DEBUG
extern __device__ __constant__ unsigned int d_PBM_isBuilt;
#endif
#if defined(_GL) || defined(_DEBUG)
extern __device__ __constant__ LocationMessages *d_locationMessagesA;
extern __device__ __constant__ LocationMessages *d_locationMessagesB;
#endif

#if defined(GLOBAL_MESSAGES) ||defined(LDG_MESSAGES)
extern __device__ __constant__ LocationMessages *d_messages;
#endif
//We need tex storage if using opengl for rendering
#if defined(_GL) || !(defined(GLOBAL_MESSAGES) ||defined(LDG_MESSAGES))
extern __device__ __constant__ cudaTextureObject_t d_tex_location[DIMENSIONS];
#endif
*/
#endif
