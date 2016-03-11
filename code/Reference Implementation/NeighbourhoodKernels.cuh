#ifndef __NeighbourhoodKernels_cuh__
#define __NeighbourhoodKernels_cuh__

#include "Neighbourhood.cuh"


__device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos);
__device__ int getHash(DIMENSIONS_IVEC gridPos);


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
