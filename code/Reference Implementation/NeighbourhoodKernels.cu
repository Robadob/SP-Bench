#include "NeighbourhoodKernels.cuh"
//getHash already clamps.
//#define SP_NO_CLAMP_GRID //Clamp grid coords to within grid (if it's possible for model to go out of bounds)

__host__ __device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos)
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

#ifdef MORTON
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
__host__ __device__ unsigned int morton3D(glm::ivec3 pos)
{
	//Pos should be clamped to 0<=x<1024

#ifdef _DEBUG
	assert(pos.x >= 0);
	assert(pos.x < 1024);
	assert(pos.y >= 0);
	assert(pos.y < 1024);
	assert(pos.z >= 0);
	assert(pos.z < 1024);
#endif
	unsigned int xx = expandBits((unsigned int)pos.x);
	unsigned int yy = expandBits((unsigned int)pos.y);
	unsigned int zz = expandBits((unsigned int)pos.z);
	return xx * 4 + yy * 2 + zz;
}
#endif
__device__ unsigned int getHash(DIMENSIONS_IVEC gridPos)
{
    //Bound gridPos to gridDimensions
    //Cheaper to bound without mod
    gridPos = clamp(gridPos, DIMENSIONS_IVEC(0), d_gridDim - DIMENSIONS_IVEC(1));
//    gridPos.x = (gridPos.x<0) ? d_gridDim.x - 1 : gridPos.x;
//    gridPos.x = (gridPos.x >= d_gridDim.x) ? 0 : gridPos.x;
//    gridPos.y = (gridPos.y<0) ? d_gridDim.y - 1 : gridPos.y;
//    gridPos.y = (gridPos.y >= d_gridDim.y) ? 0 : gridPos.y;
//#ifdef _3D
//    gridPos.z = (gridPos.z<0) ? d_gridDim.z - 1 : gridPos.z;
//    gridPos.z = (gridPos.z >= d_gridDim.z) ? 0 : gridPos.z;
//#endif
#ifndef MORTON
    //Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
#ifdef _3D
        (gridPos.z * d_gridDim.y * d_gridDim.x) +   //z
#endif
        (gridPos.y * d_gridDim.x) +					//y
        gridPos.x); 	                            //x
#else
	return morton3D(gridPos);
#endif
}
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
#ifdef _DEBUG
__global__ void assertPBMIntegrity()
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int prev = 0, me = 0, next = d_locationMessageCount;
    // (tex->local)x3, is faster than (tex->local)x1 (local->shared)x1 (shared->local)x2 right?
    if (index <= d_binCount)
    {
        if (index > 0)
            prev = tex1Dfetch<unsigned int>(d_tex_PBM, index-1);

        me = tex1Dfetch<unsigned int>(d_tex_PBM, index);
        if (index < d_binCount)
            next = tex1Dfetch<unsigned int>(d_tex_PBM, index+1);
    }

    //Assert Order
    if (prev>me||me>next)
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
    if (ordered_messages->locationX[index] == NAN ||
        ordered_messages->locationY[index] == NAN ||
        ordered_messages->locationZ[index] == NAN
        )
    {
        printf("ERROR: Location containing NaN detected.\n");
    }
#endif
}


__device__ LocationMessage *LocationMessages::getNextNeighbour(LocationMessage *sm_message)
{
	return loadNextMessage(sm_message);
}
__device__ bool invalidBinXYZ(glm::ivec3 bin)
{
	if (
		bin.x<0 || bin.x >= d_gridDim.x ||
		bin.y<0 || bin.y >= d_gridDim.y ||
		bin.z<0 || bin.z >= d_gridDim.z
		)
	{
		return true;
	}
	return false;
}
__device__ bool invalidBinYZ(glm::ivec3 bin)
{
    if (
        bin.y<0 || bin.y >= d_gridDim.y ||
        bin.z<0 || bin.z >= d_gridDim.z
        )
    {
        return true;
    }
    return false;
}
__device__ bool invalidBinX(glm::ivec3 bin)
{
    if (
        bin.x<0 || bin.x >= d_gridDim.x 
        )
    {
        return true;
    }
    return false;
}
//If we want to get next bin as strip
#if defined(STRIPS) && !defined(MORTON)
__device__ bool LocationMessages::nextBin(LocationMessage *sm_message)
{
    //extern __shared__ LocationMessage sm_messages[];
    //LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

#ifdef _3D
    if (sm_message->state.relative.x >= 1)
    {
        sm_message->state.relative.x = -1;

        if (sm_message->state.relative.y >= 1)
        {
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
    return true;
#else
    if (sm_message->state.relative >= 1)
    {
        return false;
    }
    else
    {
        sm_message->state.relative++;
    }
    return true;
#endif
}
#else
//Next bin individually
__device__ bool LocationMessages::nextBin(LocationMessage *sm_message)
{
	//extern __shared__ LocationMessage sm_messages[];
	//LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

	if (sm_message->state.relative.x >= 1)
	{
		sm_message->state.relative.x = -1;

		if (sm_message->state.relative.y >= 1)
		{

#ifdef _3D
			sm_message->state.relative.y = -1;

			if (sm_message->state.relative.z >= 1)
			{
				return false;
			}
			else
			{
				sm_message->state.relative.z++;
			}
#else
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
	return true;
}
#endif
//Load the next desired message into shared memory
__device__ LocationMessage *LocationMessages::loadNextMessage(LocationMessage *sm_message)
{
    //extern __shared__ LocationMessage sm_messages[];
    //LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

    sm_message->state.binIndex++;
    
	bool changeBin = (sm_message->state.binIndex >= sm_message->state.binIndexMax);

    while (changeBin)
    {
		if (nextBin(sm_message))
        {
#if defined(STRIPS) && !defined(MORTON)
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
                next_bin_last.x = d_gridDim.x-1;//Max x coord
            }

            int next_bin_first_hash = getHash(next_bin_first);
            int next_bin_last_hash = next_bin_first_hash + (next_bin_last.x-next_bin_first.x);//Strips are at most length 3

            //use the hash to calculate the start index (pbm stores location of 1st item)
            sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash);
            sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_last_hash+1);
            
            if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
            {
                break;//Bin strip has items!
            }
#else
//Iterate bins individually
#ifdef _3D
			glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(sm_message->state.relative.x, sm_message->state.relative.y, sm_message->state.relative.z);
#else
			glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(sm_message->state.relative.x, sm_message->state.relative.y);
#endif
			if (invalidBinXYZ(next_bin_first))
			{//Bin invalid, skip
				continue;
			}
			//Get PBM bounds
			int next_bin_first_hash = getHash(next_bin_first);
#ifdef _DEBUG
			assert(next_bin_first_hash < 100000);//arbitrary max
			assert(next_bin_first_hash >= 0);
#endif
			//use the hash to calculate the start index (pbm stores location of 1st item)
			sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash);
			sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash + 1);

			if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
			{
				break;//Bin has items!
			}
#endif
        }
        else
        {
#if defined(_GL) || defined(_DEBUG)
			//No more neighbours, finalise count by dividing by the number of messages.
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            d_locationMessagesA->count[id] /= d_locationMessageCount;
            d_locationMessagesB->count[id] /= d_locationMessageCount;
#endif
            return 0;//All bins exhausted
        }
    }
#if defined(_GL) || defined(_DEBUG)
	//Found a neighbour, increment count.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    d_locationMessagesA->count[id] += 1.0;
    d_locationMessagesB->count[id] += 1.0;
#endif
    sm_message->id = sm_message->state.binIndex;//Duplication of data TODO remove stateBinIndex
    sm_message->location.x = tex1Dfetch<float>(d_tex_location[0], sm_message->state.binIndex);
    sm_message->location.y = tex1Dfetch<float>(d_tex_location[1], sm_message->state.binIndex);
#ifdef _3D
    sm_message->location.z = tex1Dfetch<float>(d_tex_location[2], sm_message->state.binIndex);
#endif

    return sm_message;
}


__device__ LocationMessage *LocationMessages::getFirstNeighbour(DIMENSIONS_VEC location)
{
	extern __shared__ LocationMessage sm_messages[];
	LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

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
    sm_message->state.location = getGridPosition(location);
    sm_message->state.binIndex = 0;//Init binIndex greater than equal to binIndexMax to force bin change
    sm_message->state.binIndexMax = 0;
    //Location in moore neighbourhood
    //Start out of range, so we get moved into 1st cell
#if defined(STRIPS) && !defined(MORTON)
#ifdef _3D
	sm_message->state.relative = glm::ivec2(-2, -1);
#else
    sm_message->state.relative = -2;
#endif
#else
#ifdef _3D
	sm_message->state.relative = glm::ivec3(-2, -1, -1);
#else
	sm_message->state.relative = glm::ivec2(-2, -1);
#endif
#endif
	return loadNextMessage(sm_message);
}
