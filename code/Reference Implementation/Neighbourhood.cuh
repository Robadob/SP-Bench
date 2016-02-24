#ifndef __Neighbourhood_cuh__
#define __Neighbourhood_cuh__

#include "header.cuh"
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#include "glm/glm.hpp"
#include "glm/gtx/component_wise.hpp"
/**
 * Structure used to maintain search state in each thread during neigbourhood search
**/
struct SearchState
{

};
/**
 * Structure data is returned in when performing neighbour search
**/
struct LocationMessage
{
#ifdef _3D
    glm::vec3 location;
#else
    glm::vec2 location;
#endif
};
struct LocationMessages
{
    float *locationX;
    float *locationY;
#ifdef _3D
    float *locationZ;
    __device__ LocationMessage *neighbours(glm::vec3 location, SearchState* state = 0)
    {
        return neighbours(location.x, location.y, location.z, state);
    }
#else
    __device__ LocationMessage *neighbours(glm::vec3 location, SearchState* state = 0)
    {
        return neighbours(location.x, location.y, location.z, state);
    }
    __device__ LocationMessage *neighbours(float locX, float locY, SearchState* state = 0)
#endif
#ifdef _3D
    __device__ LocationMessage *neighbours(float locX, float locY, float locZ, SearchState* state = 0)
#endif
    {
        //Do some kind of neighbour search
        //float x = tex1Dfetch(tex, i);
        return 0;
    }
};

class SpatialPartition
{
public:
#ifdef _3D
    SpatialPartition(glm::vec3  environmentMin, glm::vec3 environmentMax, unsigned int maxAgents, float neighbourRad);
#else
    SpatialPartition(glm::vec2  environmentMin, glm::vec2 environmentMax, unsigned int maxAgents, float neighbourRad);
#endif
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
private:
    SpatialPartition(unsigned int maxAgents, float neighbourRad);
    //Allocators
    void deviceAllocateLocationMessages(LocationMessages **d_locMessage);
    void deviceAllocatePBM(unsigned int **d_PBM_t);
    void deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals);
    void deviceAllocateTextures();
    void deviceAllocateTexture_float(cudaTextureObject_t *tex, float* d_data, const int size);
    void deviceAllocateTexture_int(cudaTextureObject_t *tex, unsigned int* d_data, const int size);
    //Deallocators
    void deviceDeallocateLocationMessages(LocationMessages *d_locMessage);
    void deviceDeallocatePBM(unsigned int *d_PBM_t);
    void deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals);
    void deviceDeallocateTextures();

    unsigned int getBinCount();//(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))
    unsigned int maxAgents;
    float neighbourRad;
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
#ifdef _3D
    const glm::ivec2 gridDim;
    const glm::vec3  environmentMin;
    const glm::vec3  environmentMax;
#else
    const glm::ivec2 gridDim;
    const glm::vec2  environmentMin;
    const glm::vec2  environmentMax;
#endif
    //Textures
    cudaTextureObject_t tex_locationX;
    cudaTextureObject_t tex_locationY;
#ifdef _3D
    cudaTextureObject_t tex_locationZ;
#endif
    cudaTextureObject_t tex_PBM;
};

//Device constants
extern __device__ __constant__ unsigned int d_locationMessageCount;
#ifdef _3D
extern __device__ __constant__ glm::ivec3 d_gridDim;
extern __device__ __constant__ glm::vec3  d_environmentMin;
extern __device__ __constant__ glm::vec3  d_environmentMax;
#else
extern __device__ __constant__ glm::ivec2 d_gridDim;
extern __device__ __constant__ glm::vec2  d_environmentMin;
extern __device__ __constant__ glm::vec2  d_environmentMax;
#endif
//
///**
//* PARTITION_GRID_BIN_COUNT *MUST* be manually calculated if the above #defines are changed
//* It is used for defining the size of static arrays, so must be calculated pre compile.
//* PARTITION_GRID_BIN_COUNT = (ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))
//**/
//#define PARTITION_GRID_BIN_COUNT 8000
///**
//* This namespace provides static access to the means for handling spatial partitioning messages
//* The namespace allows us to hide private things statically (C++ doesn't allow classes to have private static members/methods)
//* Really we should use a singleton pattern for this, however at the time of implementation this was unknown to us
//* The Partition Boundary Matrix algorithm behind this is based off that from FlameGPU
//**/
//namespace CollisionCore
//{
//    //CUDA Constant to store the number of partition messages for kernels
//    extern __constant__ int MESSAGE_COUNT__;
//    /**
//    * This is used to pass Location Messages back to kernels
//    **/
//    struct __align__(16) LocationMessage
//    {
//        //Internal vars
//        int3 _central_bin;		//Central bin of neighbour search
//        int3 _relative_bin;		//Bin offset from _grid_bin, range -1 to 1
//        uint _bin_index_max;	//Index of boundary within current bin
//        uint _bin_index;		//Index within current bin
//        //Message vars
//        float3 position;		//Location
//        float bounding_radius;	//Radius of bounding sphere
//        int ent_id;				//Entity who spawned the message
//        int team;				//Team identifier of the unit that spawned the message
//    };
//    /**
//    * Holds the start and end index's of each bin
//    **/
//    struct __align__(16) PartitionBoundaryMatrix
//    {
//        int start[PARTITION_GRID_BIN_COUNT];
//        int end[PARTITION_GRID_BIN_COUNT];
//    };
//    /**
//    * A buffer for storing all Location Messages (SoA for efficiency)
//    **/
//    struct __align__(16) LocationMessageList
//    {
//        float position_x[LOCATION_MESSAGE_MAX];
//        float position_y[LOCATION_MESSAGE_MAX];
//        float position_z[LOCATION_MESSAGE_MAX];
//        float bounding_radius[LOCATION_MESSAGE_MAX];
//        int   ent_id[LOCATION_MESSAGE_MAX];
//        int   team[LOCATION_MESSAGE_MAX];
//    };
//    /**
//    * Resets the buffer containing Location Messages
//    **/
//    void clearBuffer();
//    /**
//    * Constructs a Partition Boundary Matrix to map to the current Location Messages
//    **/
//    void buildPBM();
//    /**
//    * Copys the relevant collision data to texture memory for efficient access
//    **/
//    PartitionBoundaryMatrix *allocCollisionBuffers();
//    /**
//    * Frees the texture buffers we stored collision data in
//    **/
//    void freeCollisionBuffers();
//    /**
//    * Overloads @see getFirstLocationMessage(float3 pos)
//    **/
//    __device__ LocationMessage *getFirstLocationMessage(float x, float y, float z);
//    /**
//    * Returns the first location message within the search area of the provided position
//    * 0 is returned if no messages are present
//    * @note entities will recieve their own location message
//    **/
//    __device__ LocationMessage *getFirstLocationMessage(float3 pos);
//    /**
//    * Pass this method the previously returned LocationMessage* from getFirstLocationMessage() or this
//    * to recieve the next location message.
//    * 0 is returned when there are no messages left to iterate
//    * @note entities will recieve their own location message
//    **/
//    __device__ LocationMessage *getNextLocationMessage(LocationMessage *message);
//    __device__ bool loadNextLocationMessage(int3 relative_bin, uint bin_index_max, int3 agent_grid_bin, int bin_index);
//    __device__ int3 getGridPosition(float3 location);
//    __device__ int getHash(int3 gridPos);
//
//    /**
//    * Called to allocate the static members
//    **/
//    void alloc();
//    /**
//    * Called to deallocate the static members
//    * @note This would be much better implemented in a singleton pattern like @see ColourPickMgr or @see MessageCore
//    **/
//    void free();
//    /**
//    * @return a pointer to the host copy of message count
//    **/
//    int *getMessageCount();
//    /**
//    * Sets the MESSAGE_COUNT__ device constant to h_messageCount
//    **/
//    void updateDMessageCount();
//    /**
//    * Template method used by implementations of GPUEntMgr to provide their LocationMessage's
//    * to the collision buffer.
//    **/
//    template <class T_EntityList>
//    void postLocationMessages(T_EntityList *d_entities, int count, int offset);
//    /**
//    * Anonymous namespace enabled private access to static members
//    **/
//    namespace
//    {
//        /**
//        * Internal method for finding the next spatial partitioning bin to traverse
//        **/
//        __device__ int getNextBin(int3* relative_bin);
//        //host copy of the message count for messages stored in buffer
//        int h_messageCount = 0;
//        //Pointer to partition boundary matrix on device
//        PartitionBoundaryMatrix *d_PBM = 0;
//        //Pointers to Location Message buffer on device
//        LocationMessageList *d_messages = 0;
//        //Pointers to Location Message buffer on device, used for swapping sorted messages efficiently
//        LocationMessageList *d_messages_swap = 0;
//    }
//}
/**
* This simply provides a compiler warning if it detects spatial partitioning bounds have changed by the bin count has not
**/
//#if	WORLD_X_MIN!=-400||WORLD_X_MAX!=400||WORLD_Y_MIN!=-400||WORLD_Y_MAX!=400||WORLD_Z_MIN!=-400||WORLD_Z_MAX!=400||SEARCH_RADIUS!=40
//
//#if PARTITION_GRID_BIN_COUNT == 8000
//#pragma message( "World Defines Have Been Changed Without Updating PARTITION_GRID_BIN_COUNT!")
//#pragma message( "PARTITION_GRID_BIN_COUNT *MUST* be manually calculated if the world bound #defines or SEARCH_RADIUS #defines are changed!")
//#pragma message( "It is used for defining the size of the static arrays within the PBM, so must be calculated pre-compile.")
//#pragma message( "Its value should equal the result of the below formula;")
//#pragma message( "(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))")
//#pragma message( "(ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))")
//#else
//#pragma message( "It appears the world bound #defines or SEARCH_RADIUS defines have been changed without updating this message.")
//#pragma message( "Please update the conditions of this warning :)")
//#endif
//
//#endif

//PARTITION_GRID_BIN_COUNT *MUST* be manually calculated if the above #defines are changed
//It is used for defining the size of static arrays, so must be calculated pre compile.
//PARTITION_GRID_BIN_COUNT = (ceil((X_MAX-X_MIN)/SEARCH_RADIUS)*ceil((Y_MAX-Y_MIN)/SEARCH_RADIUS)*ceil((Z_MAX-Z_MIN)/SEARCH_RADIUS))

#endif