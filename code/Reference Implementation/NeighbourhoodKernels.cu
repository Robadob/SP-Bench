#include "Neighbourhood.cuh"
#ifdef _3D
__device__ glm::ivec3 getGridPosition(glm::vec3 worldPos)
#else
__device__ glm::ivec2 getGridPosition(glm::vec2 worldPos)
#endif
{
    return floor((worldPos-d_environmentMin)/(d_environmentMax-d_environmentMin));
//#ifdef _3D
//    glm::ivec3 gridPos;
//#else
//    glm::ivec2 gridPos;
//#endif
//    gridPos.x = floor(d_gridDim.x * (worldPos.x - d_environmentMin.x) / (d_environmentMax.x - d_environmentMin.x));
//    gridPos.y = floor(d_gridDim.y * (worldPos.y - d_environmentMin.y) / (d_environmentMax.y - d_environmentMin.y));
//#ifdef _3D
//    gridPos.z = floor(d_gridDim.z * (worldPos.z - d_environmentMin.z) / (d_environmentMax.z - d_environmentMin.z));
//#endif
//
//    return gridPos;
}
#ifdef _3D
__device__ int getHash(glm::ivec3 gridPos)
#else
__device__ int getHash(glm::ivec2 gridPos)
#endif
{
    //Bound gridPos to gridDimensions
    //Cheaper to bound without mod
    gridPos.x = (gridPos.x<0) ? d_gridDim.x - 1 : gridPos.x;
    gridPos.x = (gridPos.x >= d_gridDim.x) ? 0 : gridPos.x;
    gridPos.y = (gridPos.y<0) ? d_gridDim.y - 1 : gridPos.y;
    gridPos.y = (gridPos.y >= d_gridDim.y) ? 0 : gridPos.y;
#ifdef _3D
    gridPos.z = (gridPos.z<0) ? d_gridDim.z - 1 : gridPos.z;
    gridPos.z = (gridPos.z >= d_gridDim.z) ? 0 : gridPos.z;
#endif

    //Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return
#ifdef _3D
        __umul24(__umul24(gridPos.z, d_gridDim.y), d_gridDim.x) //z
#endif
        + __umul24(gridPos.y, d_gridDim.x)						//y
        + gridPos.x; 	                                        //x
}
__global__ void hashLocationMessages(unsigned int* keys, unsigned int* vals, LocationMessages* messageBuffer)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    //Kill excess threads
    if (index >= d_locationMessageCount) return;
#ifdef _3D
    glm::ivec3 gridPos;
    glm::vec3 worldPos(
#else
    glm::ivec2 gridPos;
    glm::vec2 worldPos(
#endif
         messageBuffer->locationX[index]
        ,messageBuffer->locationY[index]
#ifdef _3D
        ,messageBuffer->locationZ[index]
#endif
        );
        gridPos = getGridPosition(worldPos);
        unsigned int hash = getHash(gridPos);
        keys[index] = hash;
        vals[index] = index;
}

//For-each location message in memory
//Check whether preceding key is the same
__global__ void reorderLocationMessages(
    unsigned int *keys, 
    unsigned int *vals, 
    unsigned int *pbm, 
    LocationMessages *unordered_messages, 
    LocationMessages *ordered_messages
    )
{
    extern __shared__ int sm_data[];

    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    //Load current key and copy it into shared
    unsigned int key;
    unsigned int old_pos;
    if (index < d_locationMessageCount)
    {//Don't go out of bounds when buffer is at max capacity
        key = keys[index];
        old_pos = vals[index];
        //Every valid thread put hash into shared memory
        sm_data[threadIdx.x] = key;
    }
    __syncthreads();
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

    //Load previous key
    unsigned int prev_key=0;
    //If thread 0, no prev in warp, goto global
    if (threadIdx.x == 0)
    {
        //Skip if first thread globally
        if (index != 0)
            prev_key = keys[index - 1];
    }
    else
    {
        prev_key = sm_data[threadIdx.x - 1];
    }

    //Set partition boundaries
    //if (index == 0)
    //{//First message, set first bin start
    //    pbm->start[key] = index;
    //}
    //else 
    if (prev_key != key)
    {//Boundary message, update (//start and) ends of boundary
    //    pbm->start[key] = index;
        pbm[prev_key] = index;
    }
    if (index == (d_locationMessageCount - 1))
    {//Last message, set last bin end
        pbm[key] = index + 1;
    }

    //Order messages into swap space
    ordered_messages->locationX[index] = unordered_messages->locationX[old_pos];
    ordered_messages->locationY[index] = unordered_messages->locationY[old_pos]; 
#ifdef _3D
    ordered_messages->locationZ[index] = unordered_messages->locationZ[old_pos];
#endif
}