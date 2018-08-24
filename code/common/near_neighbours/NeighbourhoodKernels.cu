#include "NeighbourhoodKernels.cuh"
#include <cuda_runtime.h>
//getHash already clamps.
//#define SP_NO_CLAMP_GRID //Clamp grid coords to within grid (if it's possible for model to go out of bounds)

__device__ __forceinline__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos)
{
#ifndef SP_NO_CLAMP_GRID
    //Clamp each grid coord to 0<=x<dim
	return clamp(floor(((worldPos - d_environmentMin) / (d_environmentMax - d_environmentMin))*d_gridDim_float), DIMENSIONS_VEC(0), d_gridDim_float - DIMENSIONS_VEC(1));
#else
	return floor(((worldPos - d_environmentMin) / (d_environmentMax - d_environmentMin))*d_gridDim_float);
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
#endif
}

__device__ __forceinline__ unsigned int getHash(DIMENSIONS_IVEC gridPos)
{
    //Bound gridPos to gridDimensions
    gridPos = clamp(gridPos, DIMENSIONS_IVEC(0), d_gridDim - DIMENSIONS_IVEC(1));
#if defined(MORTON)
    return d_mortonEncode(gridPos);
#elif defined(HILBERT)
    return d_hilbertEncode(gridPos);
#elif defined(PEANO)
    return d_peanoEncode(gridPos);
#elif defined(MORTON_COMPUTE)
    return mortonComputeEncode(gridPos);
#else
    //Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
#ifdef _3D
        (gridPos.z * d_gridDim.y * d_gridDim.x) +   //z
#endif
        (gridPos.y * d_gridDim.x) +					//y
        gridPos.x); 	                            //x
#endif
}

#ifdef ATOMIC_PBM
__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, LocationMessages* messageBuffer)
{
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

#ifdef AOS_MESSAGES
    DIMENSIONS_IVEC gridPos = getGridPosition(messageBuffer->location[index]);
#else
    DIMENSIONS_VEC worldPos(
        messageBuffer->locationX[index]
        , messageBuffer->locationY[index]
#ifdef _3D
        , messageBuffer->locationZ[index]
#endif
        );
    DIMENSIONS_IVEC gridPos = getGridPosition(worldPos);
#endif

    unsigned int hash = getHash(gridPos);
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}
__global__ void reorderLocationMessages(
    unsigned int* bin_index, 
    unsigned int* bin_sub_index,
    unsigned int *pbm_index,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
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
}
#else //ifdef-ATOMIC_PBM
__global__ void hashLocationMessages(unsigned int* keys, unsigned int* vals, LocationMessages* messageBuffer)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

    DIMENSIONS_IVEC gridPos;
    DIMENSIONS_VEC worldPos(
        messageBuffer->locationX[index]
        , messageBuffer->locationY[index]
#ifdef _3D
        , messageBuffer->locationZ[index]
#endif
        );
    gridPos = getGridPosition(worldPos);
    unsigned int hash = getHash(gridPos);
    keys[index] = hash;
    vals[index] = index;
}
__global__ void reorderLocationMessages(
    unsigned int *keys,
    unsigned int *vals,
    unsigned int *pbm_index,
    unsigned int *pbm_count,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
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
#ifdef _DEBUG
__global__ void assertPBMIntegrity()
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int prev = 0, me = 0, next = d_locationMessageCount;
    // (tex->local)x3, is faster than (tex->local)x1 (local->shared)x1 (shared->local)x2 right?
    if (index < d_binCount)
    {
        if (index > 0)
            prev = tex1Dfetch<unsigned int>(d_tex_PBM_index, index - 1);

        me = tex1Dfetch<unsigned int>(d_tex_PBM_index, index);
        if (index < d_binCount-1)
            next = tex1Dfetch<unsigned int>(d_tex_PBM_index, index + 1);
    }

    //Assert Order
    if (prev>me || me>next)
    {
        printf("ERROR: PBM contains values which are out of order.\nid:%i, prev:%i, me:%i, next:%i, count:%i\n", index, prev, me, next, d_binCount);
        assert(0);
    }
    //Assert Range
    if (me > d_locationMessageCount)
    {
        printf("ERROR: PBM contains out of range values.\nid:%i, prev:%i, me:%i, next:%i, count:%i\n", index, prev, me, next, d_binCount);
        assert(0);
    }
}
#endif

#if !defined(SHARED_BINSTATE)
__device__ __forceinline__ bool LocationMessages::getNextNeighbour(LocationMessage *sm_message)
#else
__device__ __forceinline__ LocationMessage *LocationMessages::getNextNeighbour(LocationMessage *sm_message)
#endif
{
	return loadNextMessage(sm_message);
}
__device__ __forceinline__ bool invalidBinXYZ(DIMENSIONS_IVEC bin)
{
	if (
		bin.x<0 || bin.x >= d_gridDim.x
		|| bin.y<0 || bin.y >= d_gridDim.y
#ifdef _3D
		|| bin.z<0 || bin.z >= d_gridDim.z
#endif
		)
	{
		return true;
	}
	return false;
}
__device__ __forceinline__ bool invalidBinYZ(DIMENSIONS_IVEC bin)
{
    if (
        bin.y<0 || bin.y >= d_gridDim.y
#ifdef _3D
        || bin.z<0 || bin.z >= d_gridDim.z
#endif
        )
    {
        return true;
    }
    return false;
}
__device__ __forceinline__ bool invalidBinY(DIMENSIONS_IVEC bin)
{
    if (
        bin.y<0 || bin.y >= d_gridDim.y
        )
    {
        return true;
    }
    return false;
}
__device__ __forceinline__ bool invalidBinX(DIMENSIONS_IVEC bin)
{
    if (
        bin.x<0 || bin.x >= d_gridDim.x 
        )
    {
        return true;
    }
    return false;
}
__device__ bool LocationMessages::nextBin(LocationMessage *sm_message)
{
//Max number of bins, fixed loop might get CUDA to unroll
#if defined(STRIPS) //Strips has less passes
#pragma unroll
#if defined(_2D)
    for(unsigned int i = 0;i<3;++i)
#elif defined(_3D)
    for (unsigned int i = 0; i<9; ++i)
#endif
#elif !(defined(MODULAR)||defined(MODULAR_STRIPS)) //Modular only checks if we have bin, not if bin is valid
#pragma unroll
#if defined(_2D)
for(unsigned int i = 0;i<9;++i)
#elif defined(_3D)
for (unsigned int i = 0; i<27; ++i)
#endif
#endif
{
//Get next bin
#ifdef BITFIELDS_V2
#if defined(MODULAR)
    {
        if (sm_message->state.relativeX() >= 1)
        {
            sm_message->state.relativeX(-1);

            if (sm_message->state.relativeY() >= 1)
            {

#ifdef _3D
                sm_message->state.relativeY(-1);

                if (sm_message->state.relativeZ() >= 1)
                {
                    sm_message->state.cont(false);
                }
                else
                {
                    sm_message->state.relativeZpp();
                }
#else
                sm_message->state.cont(false);
#endif
            }
            else
            {
                sm_message->state.relativeYpp();
            }
        }
        else
        {
            sm_message->state.relativeXpp();
        }
    }
#ifndef NO_SYNC
    //Wait for all threads to finish previous bin
    __syncthreads();
#endif
    if(!sm_message->state.cont())
    {
#if defined(_GL) || defined(_DEBUG)
        //No more neighbours, finalise count by dividing by the number of messages.
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        d_locationMessagesA->count[id] /= d_locationMessageCount;
        d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
        return false;
    }
#elif defined(STRIPS)
#ifdef _3D
    if (sm_message->state.relativeX() >= 1)
    {
        sm_message->state.relativeX(-1);

        if (sm_message->state.relativeY() >= 1)
        {
#if defined(_GL) || defined(_DEBUG)
            //No more neighbours, finalise count by dividing by the number of messages.
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            d_locationMessagesA->count[id] /= d_locationMessageCount;
            d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
            return false;
        }
        else
        {
            sm_message->state.relativeYpp();
        }
    }
    else
    {
        sm_message->state.relativeXpp();
    }
#else
    if (sm_message->state.relativeX() >= 1)
    {
#if defined(_GL) || defined(_DEBUG)
        //No more neighbours, finalise count by dividing by the number of messages.
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        d_locationMessagesA->count[id] /= d_locationMessageCount;
        d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
        return false;
    }
    else
    {
        sm_message->state.relativeXpp();
    }
#endif
#elif defined(MODULAR_STRIPS)
    {
#if defined(_3D)
        if (sm_message->state.relativeX() >= 1)
        {
            sm_message->state.relativeX(-1);

            if (sm_message->state.relativeY() >= 1)
            {
                sm_message->state.cont(false);
            }
            else
            {
                sm_message->state.relativeYpp();
            }
#elif defined(_2D)
        if (sm_message->state.relativeX() >= 1)
        {
            sm_message->state.cont(false);
#else
#error "Unexpected dims"
#endif
        }
        else
        {
            sm_message->state.relativeXpp();
        }
        }
#ifndef NO_SYNC
//Wait for all threads to finish previous bin
__syncthreads();
#endif
if (!sm_message->state.cont())
{
#if defined(_GL) || defined(_DEBUG)
    //No more neighbours, finalise count by dividing by the number of messages.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_locationMessagesA->count[id] /= d_locationMessageCount;
    d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
    return false;
}
#else
    if (sm_message->state.relativeX() >= 1)
    {
        sm_message->state.relativeX(-1);
        if (sm_message->state.relativeY() >= 1)
        {

#ifdef _3D
            sm_message->state.relativeY(-1);

            if (sm_message->state.relativeZ() >= 1)
            {
#if defined(_GL) || defined(_DEBUG)
                //No more neighbours, finalise count by dividing by the number of messages.
                int id = blockIdx.x * blockDim.x + threadIdx.x;
                d_locationMessagesA->count[id] /= d_locationMessageCount;
                d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
                return false;
            }
            else
            {
                sm_message->state.relativeZpp();
            }
#else
#if defined(_GL) || defined(_DEBUG)
            //No more neighbours, finalise count by dividing by the number of messages.
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            d_locationMessagesA->count[id] /= d_locationMessageCount;
            d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
            return false;
#endif
        }
        else
        {
            sm_message->state.relativeYpp();
        }
    }
    else
    {
        sm_message->state.relativeXpp();
    }
#endif
#else
#if defined(MODULAR)
    {
        if (sm_message->state.blockRelativeX >= 1)
        {
            sm_message->state.blockRelativeX = -1;

            if (sm_message->state.blockRelativeY >= 1)
            {

#ifdef _3D
                sm_message->state.blockRelativeY = -1;

                if (sm_message->state.blockRelativeZ >= 1)
                {
                    sm_message->state.blockContinue = false;
                }
                else
                {
                    sm_message->state.blockRelativeZ++;
                }
#else
                sm_message->state.blockContinue = false;
#endif
            }
            else
            {
                sm_message->state.blockRelativeY++;
            }
        }
        else
        {
            sm_message->state.blockRelativeX++;
        }
    }
#ifndef NO_SYNC
    //Wait for all threads to finish previous bin
    __syncthreads();
#endif
    if(!sm_message->state.blockContinue)
    {
#if defined(_GL) || defined(_DEBUG)
        //No more neighbours, finalise count by dividing by the number of messages.
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        d_locationMessagesA->count[id] /= d_locationMessageCount;
        d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
        return false;
    }
#elif defined(STRIPS)
#ifdef _3D
    if (sm_message->state.relativeX >= 1)
    {
        sm_message->state.relativeX = -1;

        if (sm_message->state.relativeY >= 1)
        {
#if defined(_GL) || defined(_DEBUG)
            //No more neighbours, finalise count by dividing by the number of messages.
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            d_locationMessagesA->count[id] /= d_locationMessageCount;
            d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
            return false;
        }
        else
        {
            sm_message->state.relativeY++;
        }
    }
    else
    {
        sm_message->state.relativeX++;
    }
#else
    if (sm_message->state.relativeX >= 1)
    {
#if defined(_GL) || defined(_DEBUG)
        //No more neighbours, finalise count by dividing by the number of messages.
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        d_locationMessagesA->count[id] /= d_locationMessageCount;
        d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
        return false;
    }
    else
    {
        sm_message->state.relativeX++;
    }
#endif
#elif defined(MODULAR_STRIPS)
    {
#if defined(_3D)
        if (sm_message->state.blockRelativeX >= 1)
        {
            sm_message->state.blockRelativeX = -1;

            if (sm_message->state.blockRelativeY >= 1)
            {
                sm_message->state.blockContinue = false;
            }
            else
            {
                sm_message->state.blockRelativeY++;
            }
#elif defined(_2D)
        if (sm_message->state.blockRelativeX >= 1)
        {
            sm_message->state.blockContinue = false;
#else
#error "Unexpected dims"
#endif
        }
        else
        {
            sm_message->state.blockRelativeX++;
        }
    }
#ifndef NO_SYNC
    //Wait for all threads to finish previous bin
    __syncthreads();
#endif
    if (!sm_message->state.blockContinue)
    {
#if defined(_GL) || defined(_DEBUG)
        //No more neighbours, finalise count by dividing by the number of messages.
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        d_locationMessagesA->count[id] /= d_locationMessageCount;
        d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
        return false;
    }
#else
    if (sm_message->state.relativeX >= 1)
    {
        sm_message->state.relativeX = -1;

        if (sm_message->state.relativeY >= 1)
        {

#ifdef _3D
            sm_message->state.relativeY = -1;

            if (sm_message->state.relativeZ >= 1)
            {
#if defined(_GL) || defined(_DEBUG)
                //No more neighbours, finalise count by dividing by the number of messages.
                int id = blockIdx.x * blockDim.x + threadIdx.x;
                d_locationMessagesA->count[id] /= d_locationMessageCount;
                d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
                return false;
            }
            else
            {
                sm_message->state.relativeZ++;
            }
#else
#if defined(_GL) || defined(_DEBUG)
            //No more neighbours, finalise count by dividing by the number of messages.
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            d_locationMessagesA->count[id] /= d_locationMessageCount;
            d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
            return false;
#endif
        }
        else
        {
            sm_message->state.relativeY++;
            }
        }
    else
    {
        sm_message->state.relativeX++;
    }
#endif
#endif //BITFIELDS_V2
    //Process the strip
#if defined(STRIPS)
    //Iterate bins in strips
    //calculate the next strip of contiguous bins
#ifdef BITFIELDS_V2
#ifdef _3D
    glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(-1, sm_message->state.relativeX(), sm_message->state.relativeY());
#else
    glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(-1, (int)sm_message->state.relativeX());
#endif
#else
#ifdef _3D
    glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(-1, sm_message->state.relativeX, sm_message->state.relativeY);
#else
    glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(-1, (int)sm_message->state.relativeX);
#endif
#endif

    DIMENSIONS_IVEC next_bin_last = next_bin_first;
    next_bin_last.x += 2;
    bool firstInvalid = invalidBinX(next_bin_first);
    bool lastInvalid = invalidBinX(next_bin_last);
    if (invalidBinYZ(next_bin_first))
    {//Whole strip invalid, skip
        continue;
    }
    if (firstInvalid)
    {
        next_bin_first.x = 0;
    }
    if (lastInvalid)
    {//If strip ends out of bounds only
        next_bin_last.x = d_gridDim.x - 1;//Max x coord
    }

    int next_bin_first_hash = getHash(next_bin_first);
    int next_bin_last_hash = next_bin_first_hash + (next_bin_last.x - next_bin_first.x);//Strips are at most length 3
    //use the hash to calculate the start index (pbm stores location of 1st item)
#if defined(GLOBAL_PBM)
    sm_message->state.binIndex = d_pbm_index[next_bin_first_hash];
    sm_message->state.binIndexMax = d_pbm_count[next_bin_last_hash];
#elif defined(LDG_PBM)
    sm_message->state.binIndex = __ldg(&d_pbm_index[next_bin_first_hash]);
    sm_message->state.binIndexMax = __ldg(&d_pbm_count[next_bin_last_hash]);
#else
    sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM_index, next_bin_first_hash);
    sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM_count, next_bin_last_hash);
#endif
#ifdef ATOMIC_PBM
    sm_message->state.binIndexMax += sm_message->state.binIndex;
#endif
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
    {
#ifdef STRIDED_MESSAGES
        //Adjust bin state for strided access
        sm_message->state.binOffset = sm_message->state.binIndex;
        sm_message->state.binIndexMax -= sm_message->state.binIndex;
        sm_message->state.binIndex = 0;
#endif
        return true;//Bin strip has items!
    }
#elif defined(MODULAR_STRIPS)
//Find the start of the strip
    //Find relative + offset
#ifdef BITFIELDS_V2
#if defined(_3D)
    glm::ivec2 relative = glm::ivec2(sm_message->state.relativeX() + sm_message->state.offsetX(), sm_message->state.relativeY() + sm_message->state.offsetY());
#elif defined(_2D)
    int relative = sm_message->state.relativeX() + sm_message->state.offsetX();
#endif
#else
#if defined(_3D)
    glm::ivec2 relative = glm::ivec2(sm_message->state.blockRelativeX + sm_message->state.offsetX, sm_message->state.blockRelativeY + sm_message->state.offsetY);
#elif defined(_2D)
    int relative = sm_message->state.blockRelativeX + sm_message->state.offsetX;
#endif
#endif
    //For the modular axis, if new relative > 1, set -1
#if defined(_3D)    
    relative.x = relative.x>1 ? relative.x - 3 : relative.x;
    relative.y = relative.y>1 ? relative.y - 3 : relative.y;
#elif defined(_2D)
    relative = relative>1 ? relative - 3 : relative;
#else
#error "Unexpected dims"
#endif
    //Check the new bin is valid
#if defined(_3D)
    DIMENSIONS_IVEC next_bin_first = sm_message->state.location + DIMENSIONS_IVEC(-1, relative.x, relative.y);
#elif defined(_2D)
    DIMENSIONS_IVEC next_bin_first = sm_message->state.location + DIMENSIONS_IVEC(-1, relative);
#else
#error "Unexpected dims"
#endif
    DIMENSIONS_IVEC next_bin_last = next_bin_first;
    next_bin_last.x += 2;
    bool firstInvalid = invalidBinX(next_bin_first);
    bool lastInvalid = invalidBinX(next_bin_last);
#if defined(_3D)
    if (invalidBinYZ(next_bin_first))
#elif defined(_2D)
    if (invalidBinY(next_bin_first))
#else
#error "Unexpected dims"
#endif
    {//Whole strip invalid, skip
        sm_message->state.binIndexMax = 0;
        return true;
    }
    if (firstInvalid)
    {
        next_bin_first.x = 0;
    }
    if (lastInvalid)
    {//If strip ends out of bounds only
        next_bin_last.x = d_gridDim.x - 1;//Max x coord
    }
    //Get PBM bounds
    int next_bin_first_hash = getHash(next_bin_first);
    int next_bin_last_hash = next_bin_first_hash + (next_bin_last.x - next_bin_first.x);//Strips are at most length 3
#ifdef _DEBUG
    assert(next_bin_first_hash >= 0);//Hash must be positive
    assert(next_bin_last_hash >= 0);//Hash must be positive
#endif
    //use the hash to calculate the start index (pbm stores location of 1st item)
#if defined(GLOBAL_PBM)
    sm_message->state.binIndex = d_pbm_index[next_bin_first_hash];
    sm_message->state.binIndexMax = d_pbm_count[next_bin_last_hash];
#elif defined(LDG_PBM)
    sm_message->state.binIndex = __ldg(&d_pbm_index[next_bin_first_hash]);
    sm_message->state.binIndexMax =  __ldg(&d_pbm_count[next_bin_last_hash]);
#else
    sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM_index, next_bin_first_hash);
    sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM_count, next_bin_last_hash);
#endif
#ifdef ATOMIC_PBM
    sm_message->state.binIndexMax += sm_message->state.binIndex;
#endif
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
    {
#ifdef STRIDED_MESSAGES
        //Adjust bin state for strided access
        sm_message->state.binOffset = sm_message->state.binIndex;
        sm_message->state.binIndexMax -= sm_message->state.binIndex;
        sm_message->state.binIndex = 0;
#endif
        return true;//Bin strip has items!
    }
#else
#if defined(MODULAR)
    //Find relative + offset
#ifdef BITFIELDS_V2
#if defined(_3D)
glm::ivec3 relative = glm::ivec3(sm_message->state.relativeX() + sm_message->state.offsetX(), sm_message->state.relativeY() + sm_message->state.offsetY(), sm_message->state.relativeZ() + sm_message->state.offsetZ());
#elif defined(_2D)
glm::ivec2 relative = glm::ivec2(sm_message->state.relativeX() + sm_message->state.offsetX(), sm_message->state.relativeY() + sm_message->state.offsetY());
#endif
#else
#if defined(_3D)
    glm::ivec3 relative = glm::ivec3(sm_message->state.blockRelativeX + sm_message->state.offsetX, sm_message->state.blockRelativeY + sm_message->state.offsetY, sm_message->state.blockRelativeZ + sm_message->state.offsetZ);
#elif defined(_2D)
    glm::ivec2 relative = glm::ivec2(sm_message->state.blockRelativeX + sm_message->state.offsetX, sm_message->state.blockRelativeY + sm_message->state.offsetY);
#endif
#endif
    //For each axis, if new relative > 1, set -1
    relative.x = relative.x>1 ? relative.x - 3 : relative.x;
    relative.y = relative.y>1 ? relative.y - 3 : relative.y;
#ifdef _3D
    relative.z = relative.z>1 ? relative.z - 3 : relative.z;
#endif
    DIMENSIONS_IVEC next_bin_first = sm_message->state.location + relative;
#else
    //Check the new bin is valid

#ifdef BITFIELDS_V2
#ifdef _3D
glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(sm_message->state.relativeX(), sm_message->state.relativeY(), sm_message->state.relativeZ());
#else
glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(sm_message->state.relativeX(), sm_message->state.relativeY());
#endif
#else
#ifdef _3D
glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(sm_message->state.relativeX, sm_message->state.relativeY, sm_message->state.relativeZ);
#else
glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(sm_message->state.relativeX, sm_message->state.relativeY);
#endif
#endif
#endif
    if (invalidBinXYZ(next_bin_first))
    {//Bin invalid, skip to next bin
#if defined(MODULAR)
        sm_message->state.binIndexMax = 0;
        return true;
#else
        continue;
#endif
    }
    //Get PBM bounds
    int next_bin_first_hash = getHash(next_bin_first);
#ifdef _DEBUG
    assert(next_bin_first_hash >= 0);//Hash must be positive
#endif
    //use the hash to calculate the start index (pbm stores location of 1st item)
#if defined(GLOBAL_PBM)
    sm_message->state.binIndex = d_pbm_index[next_bin_first_hash];
    sm_message->state.binIndexMax = d_pbm_count[next_bin_first_hash];
#elif defined(LDG_PBM)
    sm_message->state.binIndex = __ldg(&d_pbm_index[next_bin_first_hash]);
    sm_message->state.binIndexMax = __ldg(&d_pbm_count[next_bin_first_hash]);
#else
    sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM_index, next_bin_first_hash);
    sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM_count, next_bin_first_hash);
#endif
#ifdef ATOMIC_PBM
    sm_message->state.binIndexMax += sm_message->state.binIndex;
#endif
//#if !defined(MODULAR)
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
    {
#ifdef STRIDED_MESSAGES
        //Adjust bin state for strided access
        sm_message->state.binOffset = sm_message->state.binIndex;
        sm_message->state.binIndexMax -= sm_message->state.binIndex;
        sm_message->state.binIndex = 0;
#endif
        return true;//Bin has items!
    }
//#endif
#endif
}
#if defined(MODULAR) || defined(MODULAR_STRIPS)
return true;
#else
return false;
#endif
}
//Load the next desired message into shared memory

#if !defined(SHARED_BINSTATE)
__device__ bool LocationMessages::loadNextMessage(LocationMessage *sm_message)
#else
__device__ LocationMessage *LocationMessages::loadNextMessage(LocationMessage *sm_message)
#endif
{  
    if(sm_message->state.binIndex >= sm_message->state.binIndexMax)//Do we need to change bin?
    {
#if defined(MODULAR)||defined(MODULAR_STRIPS)

#if !defined(SHARED_BINSTATE)
        return false;
#else
        return nullptr;
#endif
#else
		if (!nextBin(sm_message))
        {
#if !defined(SHARED_BINSTATE)
            return false;
#else
            return nullptr;//All bins exhausted
#endif
        }
#endif
    }
#if defined(_GL) || defined(_DEBUG)
	//Found a neighbour, increment count.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_locationMessagesA->count[id] += 1.0;
    d_locationMessagesB->count[id] += 1.0;
#endif
#ifdef STRIDED_MESSAGES
    sm_message->id = sm_message->state.binOffset + ((sm_message->state.binIndex + threadIdx.x)%sm_message->state.binIndexMax);
#else
    sm_message->id = sm_message->state.binIndex;
#endif
#if defined(GLOBAL_MESSAGES)
#ifdef AOS_MESSAGES
    sm_message->location = d_messages->location[sm_message->id];
#else
    sm_message->location.x = d_messages->locationX[sm_message->id];
    sm_message->location.y = d_messages->locationY[sm_message->id];
#ifdef _3D
    sm_message->location.z = d_messages->locationZ[sm_message->id];
#endif
#endif
#elif defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
    //sm_message->location = __ldg(&d_messages->location[sm_message->id]);
    sm_message->location.x = __ldg(&d_messages->location[sm_message->id].x);
    sm_message->location.y = __ldg(&d_messages->location[sm_message->id].y);
#ifdef _3D
    sm_message->location.z = __ldg(&d_messages->location[sm_message->id].z);
#endif
#else
    sm_message->location.x = __ldg(&d_messages->locationX[sm_message->id]);
    sm_message->location.y = __ldg(&d_messages->locationY[sm_message->id]);
#ifdef _3D
    sm_message->location.z = __ldg(&d_messages->locationZ[sm_message->id]);
#endif
#endif
#else//Read message data from tex cache (default)
#ifdef AOS_MESSAGES
#error TODO Read AOS Messages from explicit texture
#else
    sm_message->location.x = tex1Dfetch<float>(d_tex_location[0], sm_message->id);
    sm_message->location.y = tex1Dfetch<float>(d_tex_location[1], sm_message->id);
#ifdef _3D
    sm_message->location.z = tex1Dfetch<float>(d_tex_location[2], sm_message->id);
#endif
#endif
#endif

    sm_message->state.binIndex++;

#if !defined(SHARED_BINSTATE)
    return true;
#else
    return sm_message;
#endif
}

#if !defined(SHARED_BINSTATE)
#if defined(MODULAR) || defined(MODULAR_STRIPS)
__device__ void LocationMessages::firstBin(DIMENSIONS_VEC location, LocationMessage *sm_message)
#else
__device__ void LocationMessages::getFirstNeighbour(DIMENSIONS_VEC location, LocationMessage *sm_message)
#endif
{
#else
#if defined(MODULAR) || defined(MODULAR_STRIPS)
__device__ LocationMessage *LocationMessages::firstBin(DIMENSIONS_VEC location)
#else
__device__ LocationMessage *LocationMessages::getFirstNeighbour(DIMENSIONS_VEC location)
#endif
{
    extern __shared__ LocationMessage sm_messages[];
    LocationMessage *sm_message = &(sm_messages[threadIdx.x]);
#endif

#if defined(MODULAR)
    //Init global relative if block thread X
    //if (threadIdx.x == 0)//&&threadIdx.y==0&&threadIdx.z==0)
    {
#ifdef BITFIELDS_V2
        //Init blockRelative
        sm_message->state.relativeX(-2);
        sm_message->state.relativeY(-1);
#ifdef _3D
        sm_message->state.relativeZ(-1);
#endif
        //Init blockContinue true
        sm_message->state.cont(true);
#else
        //Init blockRelative
        sm_message->state.blockRelativeX = -2;
        sm_message->state.blockRelativeY = -1;
#ifdef _3D
        sm_message->state.blockRelativeZ = -1;
#endif
        //Init blockContinue true
        //((bool*)(void*)&blockRelative[1])[0] = true;
        sm_message->state.blockContinue = true;
#endif
    }
#elif defined(MODULAR_STRIPS)
    //Init global relative if block thread X
    //if (threadIdx.x == 0)//&&threadIdx.y==0&&threadIdx.z==0)
    {
#ifdef BITFIELDS_V2
        //Init blockRelative
        sm_message->state.relativeX(-2);
#if defined(_3D)
        sm_message->state.relativeY(-1);
#endif
        //Init blockContinue true
        sm_message->state.cont(true);
#else
        //Init blockRelative
        sm_message->state.blockRelativeX = -2;
#if defined(_3D)
        sm_message->state.blockRelativeY = -1;
#endif
        //Init blockContinue true
        //((bool*)(void*)&blockRelative[1])[0] = true;
        sm_message->state.blockContinue = true;
#endif
    }
#endif

#ifdef _DEBUG
    //If first thread and PBM isn't built, print warning
    if (!d_PBM_isBuilt && (((blockIdx.x * blockDim.x) + threadIdx.x)) == 0)
        printf("PBM has not been rebuilt after calling swap()!\n");
#endif
#if defined(_GL) || defined(_DEBUG)
    //Store the locations of these in a constant
    //Set both, so we don't have to identify which is current.
    //Set counter to 0
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_locationMessagesA->count[id] = 0;
    d_locationMessagesB->count[id] = 0;
#endif
#if defined(MODULAR)
    {
        sm_message->state.location = getGridPosition(location);
#ifdef BITFIELDS_V2
#if defined(_2D)
        sm_message->state.offsetX((d_offsets[sm_message->state.location.x][sm_message->state.location.y] & (3 << 0)) >> 0);
        sm_message->state.offsetY((d_offsets[sm_message->state.location.x][sm_message->state.location.y] & (3 << 2)) >> 2);
#elif defined(_3D)
        sm_message->state.offsetX((d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 0)) >> 0);
        sm_message->state.offsetY((d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 2)) >> 2);
        sm_message->state.offsetZ((d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 4)) >> 4);
#endif
#else
#if defined(_2D)
        unsigned char t = d_offsets[sm_message->state.location.x][sm_message->state.location.y];
        sm_message->state.offsetX = (t & (3 << 0)) >> 0;
        sm_message->state.offsetY = (t & (3 << 2)) >> 2;
#elif defined(_3D)
        sm_message->state.offsetX = (d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 0)) >> 0;
        sm_message->state.offsetY = (d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 2)) >> 2;
        sm_message->state.offsetZ = (d_offsets[sm_message->state.location.x][sm_message->state.location.y][sm_message->state.location.z] & (3 << 4)) >> 4;
#endif
#endif
//#ifdef BITFIELDS_V2
//        int a = (-sm_message->state.location.x + 1) % 3; 
//        sm_message->state.offsetX(a<0?a+3:a);
//        a = (-sm_message->state.location.y + 1) % 3; 
//        sm_message->state.offsetY(a<0?a+3:a);
//#if defined(_3D)
//        a = (-sm_message->state.location.z + 1) % 3; 
//        sm_message->state.offsetZ(a<0?a+3:a);
//#endif
//#else
//        int a = (-sm_message->state.location.x + 1) % 3; 
//        sm_message->state.offsetX = a<0?a+3:a;
//        a = (-sm_message->state.location.y + 1) % 3; 
//        sm_message->state.offsetY = a<0?a+3:a;
//#if defined(_3D)
//        a = (-sm_message->state.location.z + 1) % 3; 
//        sm_message->state.offsetZ = a<0?a+3:a;
//#endif
//#endif
    }
#elif defined(MODULAR_STRIPS)
    {
        sm_message->state.location = getGridPosition(location);
#ifdef BITFIELDS_V2
        sm_message->state.offsetX((sm_message->state.location.y + 1) % 3);
#if defined(_3D)
        sm_message->state.offsetY((sm_message->state.location.z + 1) % 3);
#endif
#else
        sm_message->state.offsetX = (sm_message->state.location.y + 1) % 3;
#if defined(_3D)
        sm_message->state.offsetY = (sm_message->state.location.z + 1) % 3;
#endif
#endif
    }
#else
    sm_message->state.location = getGridPosition(location);
#endif
    //sm_message->state.binIndex = 0;//Redundant setting this, 0 is min on UINT
    sm_message->state.binIndexMax = 0;//Init binIndex greater than equal to binIndexMax to force bin change
    //Location in moore neighbourhood
    //Start out of range, so we get moved into 1st cell
#ifdef BITFIELDS_V2
#if defined(STRIPS)
    sm_message->state.relativeX(-2);
#ifdef _3D
    sm_message->state.relativeY(-1);
#endif
#elif !(defined(MODULAR)||defined(MODULAR_STRIPS))
    sm_message->state.relativeX(-2);
    sm_message->state.relativeY(-1);
#ifdef _3D
    sm_message->state.relativeZ(-1);
#endif
#endif
#else
#if defined(STRIPS)
    sm_message->state.relativeX = -2;
#ifdef _3D
	sm_message->state.relativeY = -1;
#endif
#elif !(defined(MODULAR)||defined(MODULAR_STRIPS))
    sm_message->state.relativeX = -2;
    sm_message->state.relativeY = -1;
#ifdef _3D
    sm_message->state.relativeZ = -1;
#endif
#endif
#endif
#if defined(MODULAR)||defined(MODULAR_STRIPS)
    nextBin(sm_message);
#if defined(SHARED_BINSTATE)
    return sm_message;
#endif
#else
#if defined(SHARED_BINSTATE)
	return 
#endif
    loadNextMessage(sm_message);
#endif
}
