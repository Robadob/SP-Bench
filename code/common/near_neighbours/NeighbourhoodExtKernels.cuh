#ifndef __NeighbourhoodExtKernels_cuh__
#define __NeighbourhoodExtKernels_cuh__

#include "NeighbourhoodExt.cuh"

#ifdef ATOMIC_PBM
template<class T>
__global__ void reorderLocationMessagesExt(
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages,
    T *unordered_ext,
    T *ordered_ext
    );
#else
//For-each location message in memory
//Check whether preceding key is the same
template<class T>
__global__ void reorderLocationMessagesExt(
    unsigned int *keys,
    unsigned int *vals,
    unsigned int *pbm_index,
    unsigned int *pbm_count,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages,
    T *unordered_ext,
    T *ordered_ext
    );
#endif
#endif //__NeighbourhoodExtKernels_cuh__
