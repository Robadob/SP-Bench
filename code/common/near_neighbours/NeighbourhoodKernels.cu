#include "NeighbourhoodKernels.cuh"
//getHash already clamps.
//#define SP_NO_CLAMP_GRID //Clamp grid coords to within grid (if it's possible for model to go out of bounds)

__device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos)
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

//Moved to Morton.h
//#if defined(MORTON) && defined(_3D)
//// Expands a 10-bit integer into 30 bits
//// by inserting 2 zeros after each bit.
//__host__ __device__ unsigned int expandBits(unsigned int v)
//{
//	v = (v * 0x00010001u) & 0xFF0000FFu;
//	v = (v * 0x00000101u) & 0x0F00F00Fu;
//	v = (v * 0x00000011u) & 0xC30C30C3u;
//	v = (v * 0x00000005u) & 0x49249249u;
//	return v;
//}
//
//// Calculates a 30-bit Morton code for the
//__host__ __device__ unsigned int morton3D(const DIMENSIONS_IVEC &pos)
//{
//	//Pos should be clamped to 0<=x<1024
//
//#ifdef _DEBUG
//	assert(pos.x >= 0);
//	assert(pos.x < 1024);
//	assert(pos.y >= 0);
//	assert(pos.y < 1024);
//	assert(pos.z >= 0);
//	assert(pos.z < 1024);
//#endif
//	unsigned int xx = expandBits((unsigned int)pos.x);
//	unsigned int yy = expandBits((unsigned int)pos.y);
//	unsigned int zz = expandBits((unsigned int)pos.z);
//	return xx * 4 + yy * 2 + zz;
//}
//#endif

__device__ unsigned int getHash(DIMENSIONS_IVEC gridPos)
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
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}
__global__ void reorderLocationMessages(
    unsigned int* bin_index, 
    unsigned int* bin_sub_index,
    unsigned int *pbm,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
    )
{
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Kill excess threads
    if (index >= d_locationMessageCount) return;

    unsigned int i = bin_index[index];
    unsigned int sorted_index = pbm[i] + bin_sub_index[index];

    //Order messages into swap space
    ordered_messages->locationX[sorted_index] = unordered_messages->locationX[index];
    ordered_messages->locationY[sorted_index] = unordered_messages->locationY[index];
#ifdef _3D
    ordered_messages->locationZ[sorted_index] = unordered_messages->locationZ[index];
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
    unsigned int *pbm,
    LocationMessages *unordered_messages,
    LocationMessages *ordered_messages
    )
{
    extern __shared__ int sm_data[];

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int indexPlus1 = index + 1;

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
    unsigned int next_key;
    //if thread is final thread
    if (index == d_locationMessageCount - 1)
    {
        next_key = d_binCount;
    }
    //If thread is last in block, no next in SM, goto global
    else if (threadIdx.x == blockDim.x - 1)
    {
        next_key = keys[indexPlus1];
    }
    else
    {
        next_key = sm_data[threadIdx.x + 1];
    }
    //Boundary message, set all keys after ours until (inclusive) next_key to our index +1
    if (next_key != key)
    {
        for (int k = next_key; k > key; k--)
            pbm[k] = indexPlus1;
    }
#ifdef _DEBUG
    if (next_key > d_binCount)
    {
		printf("ERROR: PBM generated an out of range next_key (%i > %i).\n", next_key, d_binCount);
		assert(0);
    }
    if (old_pos >= d_locationMessageCount)
    {
		printf("ERROR: PBM generated an out of range old_pos (%i >= %i).\n", old_pos, d_locationMessageCount);
		assert(0);
    }
#endif

    //Order messages into swap space
    ordered_messages->locationX[index] = unordered_messages->locationX[old_pos];
    ordered_messages->locationY[index] = unordered_messages->locationY[old_pos];
#ifdef _3D
    ordered_messages->locationZ[index] = unordered_messages->locationZ[old_pos];
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
    if (index <= d_binCount)
    {
        if (index > 0)
            prev = tex1Dfetch<unsigned int>(d_tex_PBM, index - 1);

        me = tex1Dfetch<unsigned int>(d_tex_PBM, index);
        if (index < d_binCount)
            next = tex1Dfetch<unsigned int>(d_tex_PBM, index + 1);
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

__device__ LocationMessage *LocationMessages::getNextNeighbour(LocationMessage *sm_message)
{
	return loadNextMessage(sm_message);
}
__device__ bool invalidBinXYZ(DIMENSIONS_IVEC bin)
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
__device__ bool invalidBinYZ(DIMENSIONS_IVEC bin)
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
__device__ bool invalidBinX(DIMENSIONS_IVEC bin)
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
	//extern __shared__ LocationMessage sm_messages[];
	//LocationMessage *sm_message = &(sm_messages[threadIdx.x]);
//Max number of bins, fixed loop might get CUDA to unroll
#if defined(STRIPS) //Strips has less passes
#if defined(_2D)
    for(unsigned int i = 0;i<3;++i)
#elif defined(_3D)
    for (unsigned int i = 0; i<9; ++i)
#endif
#elif !defined(MODULAR) //Modular only checks if we have bin, not if bin is valid
#if defined(_2D)
for(unsigned int i = 0;i<9;++i)
#elif defined(_3D)
for (unsigned int i = 0; i<27; ++i)
#endif
#endif
{
//Get next bin
#if defined(MODULAR)
    extern __shared__ LocationMessage sm_messages[];
    DIMENSIONS_IVEC *blockRelative = (DIMENSIONS_IVEC *)(void*)(&(sm_messages[blockDim.x]));
    bool *blockContinue = (bool *)(void*)&blockRelative[1];
    if(threadIdx.x==0)//&&threadIdx.y==0&&threadIdx.z==0)
    {
        if (blockRelative->x >= 1)
        {
            blockRelative->x = -1;

            if (blockRelative->y >= 1)
            {

#ifdef _3D
                blockRelative->y = -1;

                if (blockRelative->z >= 1)
                {
                    *blockContinue = false;
                }
                else
                {
                    blockRelative->z++;
                }
#else
                *blockContinue = false;
#endif
            }
            else
            {
                blockRelative->y++;
            }
        }
        else
        {
            blockRelative->x++;
        }
    }
#ifndef NO_SYNC
    //Wait for all threads to finish previous bin
    __syncthreads();
#endif
    if(!*blockContinue)
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
    if (sm_message->state.relative.x >= 1)
    {
        sm_message->state.relative.x = -1;

        if (sm_message->state.relative.y >= 1)
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
            sm_message->state.relative.y++;
        }
    }
    else
    {
        sm_message->state.relative.x++;
    }
#else
    if (sm_message->state.relative >= 1)
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
        sm_message->state.relative++;
    }
#endif
#else
    if (sm_message->state.relative.x >= 1)
    {
        sm_message->state.relative.x = -1;

        if (sm_message->state.relative.y >= 1)
        {

#ifdef _3D
            sm_message->state.relative.y = -1;

            if (sm_message->state.relative.z >= 1)
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
                sm_message->state.relative.z++;
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
            sm_message->state.relative.y++;
            }
        }
    else
    {
        sm_message->state.relative.x++;
    }
#endif
    //Process the strip
#if defined(STRIPS)
    //Iterate bins in strips
    //calculate the next strip of contiguous bins
#ifdef _3D
    glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(-1, sm_message->state.relative.x, sm_message->state.relative.y);
#else
    glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(-1, sm_message->state.relative);
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
    sm_message->state.binIndex = d_pbm[next_bin_first_hash];
    sm_message->state.binIndexMax = d_pbm[next_bin_last_hash + 1];
#elif defined(LDG_PBM)
    sm_message->state.binIndex = __ldg(&d_pbm[next_bin_first_hash]);
    sm_message->state.binIndexMax = __ldg(&d_pbm[next_bin_last_hash + 1]);
#else
    sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash);
    sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_last_hash + 1);
#endif
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
    {
        return true;//Bin strip has items!
    }
#else
#if defined(MODULAR)
    //Find relative + offset
    sm_message->state.relative = (*blockRelative)+sm_message->state.offset;
    //For each axis, if new relative > 1, set -1
    sm_message->state.relative.x = sm_message->state.relative.x>1 ? sm_message->state.relative.x - 3 : sm_message->state.relative.x;
    sm_message->state.relative.y = sm_message->state.relative.y>1 ? sm_message->state.relative.y - 3 : sm_message->state.relative.y;
#ifdef _3D
    sm_message->state.relative.z = sm_message->state.relative.z>1 ? sm_message->state.relative.z - 3 : sm_message->state.relative.z;
#endif
#endif
    //Check the new bin is valid
     DIMENSIONS_IVEC next_bin_first = sm_message->state.location + sm_message->state.relative;
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
    sm_message->state.binIndex = d_pbm[next_bin_first_hash];
    sm_message->state.binIndexMax = d_pbm[next_bin_first_hash + 1];
#elif defined(LDG_PBM)
    sm_message->state.binIndex = __ldg(&d_pbm[next_bin_first_hash]);
    sm_message->state.binIndexMax = __ldg(&d_pbm[next_bin_first_hash + 1]);
#else
    sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash);
    sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash + 1);
#endif
#if !defined(MODULAR)
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
    {
        return true;//Bin has items!
    }
#endif
#endif
}
#if defined(MODULAR)
return true;
#else
return false;
#endif
}
//Load the next desired message into shared memory
__device__ LocationMessage *LocationMessages::loadNextMessage(LocationMessage *sm_message)
{
    //extern __shared__ LocationMessage sm_messages[];
    //LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

    
    if(sm_message->state.binIndex >= sm_message->state.binIndexMax)//Do we need to change bin?
    {
#if defined(MODULAR)
        return nullptr;
#else
		if (!nextBin(sm_message))
        {
            return nullptr;//All bins exhausted
        }
#endif
    }
#if defined(_GL) || defined(_DEBUG)
	//Found a neighbour, increment count.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_locationMessagesA->count[id] += 1.0;
    d_locationMessagesB->count[id] += 1.0;
#endif
    sm_message->id = sm_message->state.binIndex;//Duplication of data TODO remove stateBinIndex
#if defined(GLOBAL_MESSAGES)
    sm_message->location.x = d_messages->locationX[sm_message->state.binIndex];
    sm_message->location.y = d_messages->locationY[sm_message->state.binIndex];
#ifdef _3D
    sm_message->location. = d_messages->locationZ[sm_message->state.binIndex];
#endif
#elif defined(LDG_MESSAGES)
    sm_message->location.x = __ldg(&d_messages->locationX[sm_message->state.binIndex]);
    sm_message->location.y = __ldg(&d_messages->locationY[sm_message->state.binIndex]);
#ifdef _3D
    sm_message->location. = __ldg(&d_messages->locationZ[sm_message->state.binIndex]);
#endif
#else//Read message data from tex cache (default)
    sm_message->location.x = tex1Dfetch<float>(d_tex_location[0], sm_message->state.binIndex);
    sm_message->location.y = tex1Dfetch<float>(d_tex_location[1], sm_message->state.binIndex);
#ifdef _3D
    sm_message->location.z = tex1Dfetch<float>(d_tex_location[2], sm_message->state.binIndex);
#endif
#endif

    sm_message->state.binIndex++;
    return sm_message;
}

#if defined(MODULAR)
__device__ LocationMessage *LocationMessages::firstBin(DIMENSIONS_VEC location)
#else
__device__ LocationMessage *LocationMessages::getFirstNeighbour(DIMENSIONS_VEC location)
#endif
{
	extern __shared__ LocationMessage sm_messages[];
	LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

#if defined(MODULAR)
    //Init global relative if block thread X
    if (threadIdx.x == 0)//&&threadIdx.y==0&&threadIdx.z==0)
    {
        //Init blockRelative
        DIMENSIONS_IVEC *blockRelative = (DIMENSIONS_IVEC *)(void*)(&(sm_messages[blockDim.x]));
#ifdef _3D
        blockRelative[0] = glm::ivec3(-2, -1, -1);
#else
        blockRelative[0] = glm::ivec2(-2, -1);
#endif
        //Init blockContinue true
        ((bool*)(void*)&blockRelative[1])[0] = true;
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
        DIMENSIONS_IVEC gridPos = getGridPosition(location);
        sm_message->state.offset = (gridPos + DIMENSIONS_IVEC(1)) % 3;
        sm_message->state.location = gridPos;
    }
#else
    sm_message->state.location = getGridPosition(location);
#endif
    //sm_message->state.binIndex = 0;//Redundant setting this, 0 is min on UINT
    sm_message->state.binIndexMax = 0;//Init binIndex greater than equal to binIndexMax to force bin change
    //Location in moore neighbourhood
    //Start out of range, so we get moved into 1st cell
#if defined(STRIPS)
#ifdef _3D
	sm_message->state.relative = glm::ivec2(-2, -1);
#else
    sm_message->state.relative = -2;
#endif
#elif !defined(MODULAR) //No need to initialise this value for modular
#ifdef _3D
	sm_message->state.relative = glm::ivec3(-2, -1, -1);
#else
	sm_message->state.relative = glm::ivec2(-2, -1);
#endif
#endif
#if defined(MODULAR)
    nextBin(sm_message);
    return sm_message;
#else
	return loadNextMessage(sm_message);
#endif
}
