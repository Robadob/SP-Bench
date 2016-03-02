#ifndef __NeighbourhoodKernels_cuh__
#define __NeighbourhoodKernels_cuh__

#include "Neighbourhood.cuh"


__device__ bool getNextBin(glm::ivec3 *relative);
#ifdef _3D
__device__ glm::ivec3 getGridPosition(glm::vec3 worldPos);
__device__ int getHash(glm::ivec3 gridPos);
#else
__device__ glm::ivec2 getGridPosition(glm::vec2 worldPos);
__device__ int getHash(glm::ivec2 gridPos);
#endif

__global__ void hashLocationMessages(unsigned int* keys, unsigned int* vals, LocationMessages* messageBuffer);

//For-each location message in memory
//Check whether preceding key is the same
__global__ void reorderLocationMessages(
    unsigned int *keys,
    unsigned int *vals,
    unsigned int *pbm,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
    );

#endif