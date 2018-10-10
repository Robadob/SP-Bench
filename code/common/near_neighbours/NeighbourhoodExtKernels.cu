#include "NeighbourhoodExtKernels.cuh"
#include <cuda_runtime.h>

#ifdef ATOMIC_PBM

template<class T>
__global__ void reorderLocationMessagesExt(
    unsigned int* bin_index, 
    unsigned int* bin_sub_index,
    unsigned int *pbm_index,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages,
    T *unordered_ext,
    T *ordered_ext
    )
{
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

    unsigned int i = bin_index[index];
    unsigned int sorted_index = pbm_index[i] + bin_sub_index[index];

    //Order messages into swap space
#ifdef AOS_MESSAGES
    ordered_messages->location[sorted_index] = unordered_messages->location[index];
#else
    ordered_messages->locationX[sorted_index] = unordered_messages->locationX[index];
    ordered_messages->locationY[sorted_index] = unordered_messages->locationY[index];
#ifdef _3D
    ordered_messages->locationZ[sorted_index] = unordered_messages->locationZ[index];
#endif
#endif
    ordered_ext[sorted_index] = unordered_ext[index];
}
#else //ifdef-ATOMIC_PBM

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
    )
{
    extern __shared__ int sm_data[];

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    //Load current key and copy it into shared
    unsigned int key;
    unsigned int old_pos;
    if (index < d_locationMessageCount)
    {//Don't go out of bounds when buffer is at max capacity
        key = keys[index];
        old_pos = vals[index];
        //Every valid thread put key into shared memory
        sm_data[threadIdx.x] = key;
    }
    __syncthreads();
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

    //Load next key
    unsigned int prev_key;
    //if thread is final thread
    if (index == 0)
    {
        prev_key = UINT_MAX;//?
    }
    //If thread is first in block, no next in SM, goto global
    else if (threadIdx.x == 0)
    {
        prev_key = keys[index - 1];
    }
    else
    {
        prev_key = sm_data[threadIdx.x - 1];
    }
    //Boundary message
    if (prev_key != key)
    {
        pbm_index[key] = index;
        if (index > 0)
        {
            pbm_count[key - 1] = index;
        }
    }
    if (index == d_locationMessageCount - 1)
    {
        pbm_count[key] = d_locationMessageCount;
    }
#ifdef _DEBUG
    if (old_pos >= d_locationMessageCount)
    {
        printf("ERROR: PBM generated an out of range old_pos (%i >= %i).\n", old_pos, d_locationMessageCount);
        assert(0);
    }
#endif

#ifdef AOS_MESSAGES
    ordered_messages->location[index] = unordered_messages->location[old_pos];
#else
    //Order messages into swap space
    ordered_messages->locationX[index] = unordered_messages->locationX[old_pos];
    ordered_messages->locationY[index] = unordered_messages->locationY[old_pos];
#ifdef _3D
    ordered_messages->locationZ[index] = unordered_messages->locationZ[old_pos];
#endif
#endif
    ordered_ext[index] = unordered_ext[old_pos];

#ifdef _DEBUG
    //Check these rather than ordered in hopes of memory coealesce
    if (ordered_messages->locationX[index] == NAN
        || ordered_messages->locationY[index] == NAN
#ifdef _3D
        || ordered_messages->locationZ[index] == NAN
#endif
        )
    {
        printf("ERROR: Location containing NaN detected.\n");
    }
#endif
}
#endif //ifdef-else-ATOMIC_PBM