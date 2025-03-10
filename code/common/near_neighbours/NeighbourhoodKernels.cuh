#ifndef __NeighbourhoodKernels_cuh__
#define __NeighbourhoodKernels_cuh__

#include "Neighbourhood.cuh"


__device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos);
__device__ unsigned int getHash(DIMENSIONS_IVEC gridPos);
//__device__ unsigned int getHash(unsigned int x, unsigned int y);

#ifdef ATOMIC_PBM
__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, LocationMessages* messageBuffer);
__global__ void reorderLocationMessages(
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
    );
#else
__global__ void hashLocationMessages(unsigned int* keys, unsigned int* vals, LocationMessages* messageBuffer);

//For-each location message in memory
//Check whether preceding key is the same
__global__ void reorderLocationMessages(
    unsigned int *keys,
    unsigned int *vals,
    unsigned int *pbm_index,
    unsigned int *pbm_count,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
    );
#endif
__global__ void assertPBMIntegrity();
#endif
