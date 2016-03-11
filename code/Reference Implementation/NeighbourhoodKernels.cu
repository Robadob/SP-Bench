#include "NeighbourhoodKernels.cuh"
//getHash already clamps.
#define SP_NO_CLAMP_GRID //Clamp grid coords to within grid (if it's possible for model to go out of bounds)

__device__ DIMENSIONS_IVEC getGridPosition(DIMENSIONS_VEC worldPos)
{
#ifndef SP_NO_CLAMP_GRID
    //Clamp each grid coord to 0<=x<dim
    return clamp(floor(((worldPos - d_environmentMin) / (d_environmentMax - d_environmentMin))*d_gridDim_float), glm::vec3(0), d_gridDim_float-glm::vec3(1));
#else
    return floor(((worldPos - d_environmentMin) / (d_environmentMax - d_environmentMin))*d_gridDim_float);
#endif
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

__device__ int getHash(DIMENSIONS_IVEC gridPos)
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

    //Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return
#ifdef _3D
        (gridPos.z * d_gridDim.y * d_gridDim.x) +   //z
#endif
        (gridPos.y * d_gridDim.x) +					//y
        gridPos.x; 	                                //x
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

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

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
    unsigned int prev_key = key;//0
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
        for (int k = prev_key; k < key;k++)//Loop here stops empty bins being left at 0
            pbm[k] = index;
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



__device__ LocationMessage *LocationMessages::getNextNeighbour(LocationMessage *message)
{
    extern __shared__ LocationMessage sm_messages[];
    //LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

    return loadNextMessage();
}
__device__ bool LocationMessages::nextBin()
{
    extern __shared__ LocationMessage sm_messages[];
    LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

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
//Load the next desired message into shared memory
__device__ LocationMessage *LocationMessages::loadNextMessage()
{
    extern __shared__ LocationMessage sm_messages[];
    LocationMessage *sm_message = &(sm_messages[threadIdx.x]);

    bool changeBin = true;
    sm_message->state.binIndex++;
    if (sm_message->state.binIndex < sm_message->state.binIndexMax)
        changeBin = false;

    while (changeBin)
    {
        if (nextBin())
        {
            //calculate the next strip of contiguous bins
#ifdef _3D
            glm::ivec3 next_bin_first = sm_message->state.location + glm::ivec3(-1, sm_message->state.relative.x, sm_message->state.relative.y);
#else
            glm::ivec2 next_bin_first = sm_message->state.location + glm::ivec2(-1, sm_message->state.relative);
#endif
            int next_bin_first_hash = getHash(next_bin_first);
            int next_bin_last_hash = next_bin_first_hash + 2;//Strips are length 3
            //use the hash to calculate the start index (pbm stores location of
            if (next_bin_last_hash >= d_binCount)
                next_bin_last_hash = d_binCount - 1;
            sm_message->state.binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_first_hash - 1);
            sm_message->state.binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, next_bin_last_hash);
            
            if (sm_message->state.binIndex < sm_message->state.binIndexMax)//(bin_index_min != 0xffffffff)
            {
                break;
            }
            continue;//Strip is empty, continue
        }
        else
        {
            return 0;//All bins exhausted
        }
    }
    sm_message->id = sm_message->state.binIndex;//Duplication of data TODO remove stateBinIndex
    //From texture
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
    sm_message->state.location = getGridPosition(location);
    sm_message->state.binIndex = 0;//Init binIndex greater than equal to binIndexMax to force bin change
    sm_message->state.binIndexMax = 0;
    //Location in moore neighbourhood
    //Start out of range, so we get moved into 1st cell
#ifdef _3D
    sm_message->state.relative = glm::ivec2(-2, -1);
#else
    sm_message->state.relative = -2;
#endif

    return loadNextMessage();
}
