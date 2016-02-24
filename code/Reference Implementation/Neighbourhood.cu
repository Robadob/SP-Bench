#include "Neighbourhood.cuh"
#include "NeighbourhoodKernels.cuh"
#include "NeighbourhoodConstants.cuh"
#ifndef THRUST
#include <cub\cub.cuh>
#else
#include <thrust\sort.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

#ifdef _3D
SpatialPartition::SpatialPartition(glm::vec3  environmentMin, glm::vec3 environmentMax, unsigned int maxAgents, float neighbourRad)
#else
SpatialPartition::SpatialPartition(glm::vec2  environmentMin, glm::vec2 environmentMax, unsigned int maxAgents, float neighbourRad)
#endif
    : environmentMin(environmentMin)
    , environmentMax(environmentMax)
    , maxAgents(maxAgents)
    , neighbourRad(neighbourRad)
    , locationMessageCount(0)
    , gridDim((environmentMax - environmentMin) / neighbourRad)
{
    //Allocate bins in GPU memory
    deviceAllocateLocationMessages(&d_locationMessages);
    //Allocate bins swap in GPU memory
    deviceAllocateLocationMessages(&d_locationMessages_swap);
    //Allocate PBM
    deviceAllocatePBM(&d_PBM);
    //Allocate primitive structures
    deviceAllocatePrimitives(&d_keys, &d_vals);
#ifndef THRUST
    deviceAllocatePrimitives(&d_keys_swap, &d_vals_swap);
#endif
    //Set device constants
#ifdef _3D
    cudaMemcpyToSymbol(&d_gridDim, &gridDim, sizeof(glm::ivec3));
    cudaMemcpyToSymbol(&d_environmentMin, &environmentMin, sizeof(glm::vec3));
    cudaMemcpyToSymbol(&d_environmentMax, &environmentMax, sizeof(glm::vec3));
#else
    cudaMemcpyToSymbol(&d_gridDim, &gridDim, sizeof(glm::ivec2));
    cudaMemcpyToSymbol(&d_environmentMin, &environmentMin, sizeof(glm::vec2));
    cudaMemcpyToSymbol(&d_environmentMax, &environmentMax, sizeof(glm::vec2));
#endif
    setLocationCount(locationMessageCount);
}
SpatialPartition::~SpatialPartition()
{
    //Dellocate bins in GPU memory
    deviceDeallocateLocationMessages(d_locationMessages);
    //Dellocate bins swap in GPU memory
    deviceDeallocateLocationMessages(d_locationMessages_swap);
    //Dellocate PBM
    deviceDeallocatePBM(d_PBM);
    //Deallocated primitive structures
    deviceDeallocatePrimitives(d_keys, d_vals);
#ifndef THRUST
    deviceDeallocatePrimitives(d_keys_swap, d_vals_swap);
#endif
}
void SpatialPartition::deviceAllocateLocationMessages(LocationMessages **d_locMessage)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_locMessage, sizeof(LocationMessages)));
    float *d_loc_temp;
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*binCount));
    CUDA_CALL(cudaMemcpy((*d_locMessage)->locationX, d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*binCount));
    CUDA_CALL(cudaMemcpy((*d_locMessage)->locationY, d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
#ifdef _3D
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*binCount));
    CUDA_CALL(cudaMemcpy((*d_locMessage)->locationZ, d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
#endif
}
void SpatialPartition::deviceAllocatePBM(unsigned int **d_PBM_t)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_PBM_t, sizeof(unsigned int)*binCount));
}
void SpatialPartition::deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_keys, sizeof(unsigned int)*binCount));
    CUDA_CALL(cudaMalloc(d_vals, sizeof(unsigned int)*binCount));
}
void SpatialPartition::deviceAllocateTextures()
{
    float *d_bufferPtr;
    //Potentially refactor so we store/swap these pointers on host in syncrhonisation
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, d_locationMessages->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
    deviceAllocateTexture_float(&tex_locationX, d_bufferPtr, locationMessageCount);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, d_locationMessages->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
    deviceAllocateTexture_float(&tex_locationY, d_bufferPtr, locationMessageCount);
#ifdef _3D
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, d_locationMessages->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
    deviceAllocateTexture_float(&tex_locationZ, d_bufferPtr, locationMessageCount);
#endif
    //PBM
    deviceAllocateTexture_int(&tex_PBM, d_PBM, getBinCount());
}

void SpatialPartition::deviceAllocateTexture_float(cudaTextureObject_t *tex, float* d_data, const int size)
{
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = size*sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
}
void SpatialPartition::deviceAllocateTexture_int(cudaTextureObject_t *tex, unsigned int* d_data, const int size)
{
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = size*sizeof(unsigned int);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
}
void SpatialPartition::deviceDeallocateLocationMessages(LocationMessages *d_locMessage)
{
    float *d_loc_temp;
    CUDA_CALL(cudaMemcpy(&d_loc_temp, d_locMessage->locationX, sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaFree(d_loc_temp));
    CUDA_CALL(cudaMemcpy(&d_loc_temp, d_locMessage->locationY, sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaFree(d_loc_temp));
#ifdef _3D
    CUDA_CALL(cudaMemcpy(d_loc_temp, d_locMessage->locationZ, sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaFree(d_loc_temp));
#endif
    CUDA_CALL(cudaFree(d_locMessage));
}
void SpatialPartition::deviceDeallocatePBM(unsigned int *d_PBM_t)
{
    CUDA_CALL(cudaFree(d_PBM_t));
}
void SpatialPartition::deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals)
{
    CUDA_CALL(cudaFree(d_keys));
    CUDA_CALL(cudaFree(d_vals));
}
void SpatialPartition::deviceDeallocateTextures()
{
    cudaDestroyTextureObject(tex_locationX);
    cudaDestroyTextureObject(tex_locationY);
#ifdef _3D
    cudaDestroyTextureObject(tex_locationZ);
#endif
    cudaDestroyTextureObject(tex_PBM);
}

unsigned int SpatialPartition::getBinCount()
{
    return (unsigned int)glm::compMul((environmentMax - environmentMin) / neighbourRad);
}
void SpatialPartition::setLocationCount(unsigned int t_locationMessageCount)
{
    //Set local copy
    locationMessageCount = t_locationMessageCount;
    //Set device constants
    cudaMemcpyToSymbol(&d_locationMessageCount, &locationMessageCount, sizeof(unsigned int));
}

void SpatialPartition::launchHashLocationMessages()
{
    int blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, hashLocationMessages, 0, 0);
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    hashLocationMessages <<<gridSize, blockSize>>>(d_keys, d_vals, d_locationMessages);
    cudaDeviceSynchronize();
    CUDA_CALL(cudaGetLastError());
}
int requiredSM_reorderLocationMessages(int blockSize)
{
    return sizeof(unsigned int)*blockSize;
}
void SpatialPartition::launchReorderLocationMessages()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorderLocationMessages, requiredSM_reorderLocationMessages, 0);
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    //Copy messages from d_messages to d_messages_swap, in hash order
    reorderLocationMessages <<<gridSize, blockSize, requiredSM_reorderLocationMessages(blockSize) >>>(d_keys, d_vals, d_PBM, d_locationMessages, d_locationMessages_swap);
    //Switch d_locationMessages and d_locationMessages_swap
    LocationMessages* d_locationmessages_temp = d_locationMessages;
    d_locationMessages = d_locationMessages_swap;
    d_locationMessages_swap = d_locationmessages_temp;
    //Wait for return
    cudaDeviceSynchronize();
    CUDA_CALL(cudaGetLastError());
}
void SpatialPartition::buildPBM()
{
    //Clear previous textures
    deviceDeallocateTextures();

    //If no messages, or instances, don't bother
    if (locationMessageCount<1) return;
    //Fill primitive key/val arrays for sort
    launchHashLocationMessages();
    //Sort key val arrays using thrust/CUB
#ifndef THRUST
    //CUB version
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, locationMessageCount);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, locationMessageCount);
    //Swap arrays
    unsigned int *temp;
    temp = d_keys;
    d_keys = d_keys_swap;
    d_keys_swap = temp;
    temp = d_vals;
    d_vals = d_vals_swap;
    d_vals_swap = temp;
    //Free temporary memory
    cudaFree(d_temp_storage);

    //Clone data to textures ready for neighbourhood search
    deviceAllocateTextures();
#else
    //Thrust version
    //cudaStream_t s1;
    //cudaStreamCreate(&s1);
    //thrust::sort_by_key(thrust::cuda::par(s1), d_keys, d_keys + locationMessageCount, d_vals);
    thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + locationMessageCount, d_vals);
    //cudaStreamSynchronize(s1);
    //cudaStreamDestroy(s1);
#endif
    CUDA_CALL(cudaGetLastError());
    //Reorder map in order of message_hash	
    //Fill pbm start coords with known value 0xffffffff
    //CUDA_CALL(cudaMemset(d_PBM, 0xffffffff, PARTITION_GRID_BIN_COUNT * sizeof(int)));
    //Fill pbm end coords with known value 0x00000000 (this should mean if the mysterious bug does occur, the cell is just dropped, not large loop created)
    unsigned int binCount = getBinCount(); 
    CUDA_CALL(cudaMemset(d_PBM, 0x00000000, binCount * sizeof(unsigned int)));
    launchReorderLocationMessages();
}










//
//
//
//
////#include "CudppMgr.h"//Replacing cudpp with CUB
//#include <limits.h>
//#include <float.h>
////#include "Environment.h"
/////Device constants 
//namespace Neighbourhood
//{
//    __constant__ int COUNT__;
//    __constant__ int MESSAGE_COUNT__;
//    __constant__ int3 GRID_DIMENSIONS__;
//    __constant__ float SEARCH_RADIUS__;
//    //Global Textures & Texture Constants
//    __constant__ int d_tex_message_location_x_offset;
//    __constant__ int d_tex_message_location_y_offset;
//    __constant__ int d_tex_message_location_z_offset;
//    __constant__ int d_tex_message_radius_offset;
//    __constant__ int d_tex_message_ent_id_offset;
//    __constant__ int d_tex_message_team_offset;
//    __constant__ int d_tex_pbm_start_offset;
//    __constant__ int d_tex_pbm_end_offset;
//    //Texture binding vars (used for reading location messages more efficiently that from global memory)
//    //These aren't valid c++ prior to being parsed by nvcc, so must live in a .cu file
//    //1 per LocationMessage part
//    texture<float, 1, cudaReadModeElementType> tex_message_location_x;
//    texture<float, 1, cudaReadModeElementType> tex_message_location_y;
//#ifdef 3D
//    texture<float, 1, cudaReadModeElementType> tex_message_location_z;
//#endif
//    ////Pbm ones
//    texture<int, 1, cudaReadModeElementType> tex_pbm_start;
//
//    __device__ bool loadNextLocationMessage(int3 relative_bin, uint bin_index_max, int3 central_bin, int bin_index)
//    {
//        extern __shared__ int sm_data[];
//        char* message_share = (char*)&sm_data[0];
//
//        int change_bin = true;
//        bin_index++;
//
//        //Check if there are messages left to check in current bin
//        if (bin_index < bin_index_max)
//            change_bin = false;
//
//        while (change_bin)
//        {
//            //get the next relative grid position 
//            if (getNextBin(&relative_bin))
//            {
//                //calculate the next cells grid position and hash
//                int3 next_bin_position = central_bin + relative_bin;
//                int next_bin_hash = getHash(next_bin_position);
//                //use the hash to calculate the start index
//                uint bin_index_min = tex1Dfetch(tex_pbm_start, next_bin_hash + d_tex_pbm_start_offset);
//
//                //check for messages in the cell (empty cells with have a start index of 0xffffffff)
//                if (bin_index_min != 0xffffffff)
//                {
//                    //if there are messages in the cell then update the cell index max value
//                    bin_index_max = tex1Dfetch(tex_pbm_end, next_bin_hash + d_tex_pbm_end_offset);
//                    //start from the cell index min
//                    bin_index = bin_index_min;
//                    //exit the loop as we have found a valid cell with message data
//                    change_bin = false;
//                }
//            }
//            else
//            {
//                //We have exhausted all the neightbouring cells so there are no more messages
//                return false;
//            }
//        }
//
//        LocationMessage temp_message;
//
//        //get message data using texture fetch
//        temp_message._relative_bin = relative_bin;
//        temp_message._bin_index_max = bin_index_max;
//        temp_message._bin_index = bin_index;
//        temp_message._central_bin = central_bin;
//
//        //Using texture cache
//        temp_message.position.x = tex1Dfetch(tex_message_location_x, bin_index + d_tex_message_location_x_offset);
//        temp_message.position.y = tex1Dfetch(tex_message_location_y, bin_index + d_tex_message_location_y_offset);
//        temp_message.position.z = tex1Dfetch(tex_message_location_z, bin_index + d_tex_message_location_z_offset);
//        temp_message.bounding_radius = tex1Dfetch(tex_message_radius, bin_index + d_tex_message_radius_offset);
//        temp_message.ent_id = tex1Dfetch(tex_message_ent_id, bin_index + d_tex_message_ent_id_offset);
//        temp_message.team = tex1Dfetch(tex_message_team, bin_index + d_tex_message_team_offset);
//
//        //load it into shared memory (no sync as no sharing between threads)
//        int message_index = threadIdx.x * sizeof(LocationMessage);
//        LocationMessage* sm_message = ((LocationMessage*)&message_share[message_index]);
//        sm_message[0] = temp_message;
//        return true;
//    }
//    void alloc()
//    {//Only run first time, to malloc statics
//        h_messageCount = 12;
//        CUDA_CALL(cudaMemcpyToSymbol(MESSAGE_COUNT__, &h_messageCount, sizeof(int)));
//        //Malloc
//        CUDA_CALL(cudaMalloc(&d_messages, sizeof(LocationMessageList)));
//        CUDA_CALL(cudaMalloc(&d_messages_swap, sizeof(LocationMessageList)));
//        CUDA_CALL(cudaMalloc(&d_PBM, sizeof(PartitionBoundaryMatrix)));
//        //ccTex = new CCTextures;
//        //CUDA Constants
//        float t_searchRad = SEARCH_RADIUS;
//        CUDA_CALL(cudaMemcpyToSymbol(SEARCH_RADIUS__, &t_searchRad, sizeof(float)));
//        int3 t_paritionDims = make_int3(
//            (int)ceil((WORLD_X_MAX - WORLD_X_MIN) / (float)SEARCH_RADIUS),
//            (int)ceil((WORLD_Y_MAX - WORLD_Y_MIN) / (float)SEARCH_RADIUS),
//            (int)ceil((WORLD_Z_MAX - WORLD_Z_MIN) / (float)SEARCH_RADIUS)
//            );
//        CUDA_CALL(cudaMemcpyToSymbol(GRID_DIMENSIONS__, &t_paritionDims, sizeof(int3)));
//    }
//    void free()
//    {
//        cudaFree(&d_messages);
//        cudaFree(&d_messages_swap);
//        cudaFree(&d_PBM);
//    }
//    void clearBuffer()
//    {
//        //If no messages, or instances, don't bother
//        if (h_messageCount<1) return;
//        h_messageCount = 0;
//        CUDA_CALL(cudaMemcpyToSymbol(MESSAGE_COUNT__, &h_messageCount, sizeof(int)));
//    }
//    void buildPBM()
//    {
//
//    }
//    PartitionBoundaryMatrix *allocCollisionBuffers()
//    {
//        ///Bind location message buffer to textures
//        //LocationMessage->position.x
//        CUDA_CALL(cudaGetLastError());
//        size_t tex_message_x_byte_offset;
//        cudaChannelFormatDesc tex_desc_x = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&tex_message_x_byte_offset, &tex_message_location_x, d_messages->position_x, &tex_desc_x, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_location_x_offset = (int)tex_message_x_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_x_offset, &h_tex_message_location_x_offset, sizeof(int)));
//        //LocationMessage->position.y
//        size_t tex_message_y_byte_offset;
//        cudaChannelFormatDesc tex_desc_y = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&tex_message_y_byte_offset, &tex_message_location_y, d_messages->position_y, &tex_desc_y, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_location_y_offset = (int)tex_message_y_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_y_offset, &h_tex_message_location_y_offset, sizeof(int)));
//        //LocationMessage->position.z
//        size_t tex_message_z_byte_offset;
//        cudaChannelFormatDesc tex_desc_z = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&tex_message_z_byte_offset, &tex_message_location_z, d_messages->position_z, &tex_desc_z, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_location_z_offset = (int)tex_message_z_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_z_offset, &h_tex_message_location_z_offset, sizeof(int)));
//        //LocationMessage->bounding_radius
//        size_t d_tex_message_radius_byte_offset;
//        cudaChannelFormatDesc tex_desc_radius = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&d_tex_message_radius_byte_offset, &tex_message_radius, d_messages->bounding_radius, &tex_desc_radius, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_radius_offset = (int)d_tex_message_radius_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_y_offset, &h_tex_message_radius_offset, sizeof(int)));
//        size_t d_tex_message_ent_id_byte_offset;
//        cudaChannelFormatDesc tex_desc_ent_id = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&d_tex_message_ent_id_byte_offset, &tex_message_ent_id, d_messages->ent_id, &tex_desc_ent_id, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_ent_id_offset = (int)d_tex_message_ent_id_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_y_offset, &h_tex_message_ent_id_offset, sizeof(int)));
//        //LocationMessage->team
//        size_t d_tex_message_team_byte_offset;
//        cudaChannelFormatDesc tex_desc_team = cudaCreateChannelDesc<float>();
//        CUDA_CALL(cudaBindTexture(&d_tex_message_team_byte_offset, &tex_message_team, d_messages->team, &tex_desc_team, sizeof(float)*LOCATION_MESSAGE_MAX));//sizeof(int)
//        int h_tex_message_team_offset = (int)d_tex_message_team_byte_offset / sizeof(float);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_message_location_y_offset, &h_tex_message_team_offset, sizeof(int)));
//
//        ///Bind PBM start and end indices to textures
//        //Start
//        size_t d_tex_pbm_start_byte_offset;
//        cudaChannelFormatDesc tex_desc_pbm_start = cudaCreateChannelDesc<int>();
//        CUDA_CALL(cudaBindTexture(&d_tex_pbm_start_byte_offset, &tex_pbm_start, d_PBM->start, &tex_desc_pbm_start, sizeof(int)*PARTITION_GRID_BIN_COUNT));
//        int h_tex_pbm_start_offset = (int)d_tex_pbm_start_byte_offset / sizeof(int);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_pbm_start_offset, &h_tex_pbm_start_offset, sizeof(int)));
//        //End
//        size_t d_tex_pbm_end_byte_offset;
//        cudaChannelFormatDesc tex_desc_pbm_end = cudaCreateChannelDesc<int>();
//        CUDA_CALL(cudaBindTexture(&d_tex_pbm_end_byte_offset, &tex_pbm_end, d_PBM->end, &tex_desc_pbm_end, sizeof(int)*PARTITION_GRID_BIN_COUNT));
//        int h_tex_pbm_end_offset = (int)d_tex_pbm_end_byte_offset / sizeof(int);
//        CUDA_CALL(cudaMemcpyToSymbol(d_tex_pbm_end_offset, &h_tex_pbm_end_offset, sizeof(int)));
//
//        return d_PBM;
//    }
//    void freeCollisionBuffers()
//    {
//        CUDA_CALL(cudaUnbindTexture(tex_message_location_x));
//        CUDA_CALL(cudaUnbindTexture(tex_message_location_y));
//        CUDA_CALL(cudaUnbindTexture(tex_message_location_z));
//        CUDA_CALL(cudaUnbindTexture(tex_message_radius));
//        CUDA_CALL(cudaUnbindTexture(tex_message_ent_id));
//        CUDA_CALL(cudaUnbindTexture(tex_message_team));
//        CUDA_CALL(cudaUnbindTexture(tex_pbm_start));
//        CUDA_CALL(cudaUnbindTexture(tex_pbm_end));
//    }
//    __device__ LocationMessage *getFirstLocationMessage(float x, float y, float z)
//    {
//        extern __shared__ int sm_data[];
//        char* message_share = (char*)&sm_data[0];
//
//        int3 relative_bin = make_int3(-2, -1, -1);//Start out of range, so we get moved into 1st cell
//        int bin_index_max = 0;
//        int bin_index = 0;
//        float3 position = make_float3(x, y, z);
//        int3 agent_grid_bin = getGridPosition(position);
//
//        if (loadNextLocationMessage(relative_bin, bin_index_max, agent_grid_bin, bin_index))
//        {
//            int message_index = __mul24(threadIdx.x, sizeof(LocationMessage));
//            return ((LocationMessage*)&message_share[message_index]);
//        }
//        else
//        {
//            return 0;
//        }
//    }
//    __device__ LocationMessage *getFirstLocationMessage(float3 position)
//    {
//        extern __shared__ int sm_data[];
//        char* message_share = (char*)&sm_data[0];
//
//        int3 relative_bin = make_int3(-2, -1, -1);//Start out of range, so we get moved into 1st cell
//        int bin_index_max = 0;
//        int bin_index = 0;
//        int3 agent_grid_cell = getGridPosition(position);
//
//        if (loadNextLocationMessage(relative_bin, bin_index_max, agent_grid_cell, bin_index))
//        {
//            int message_index = __mul24(threadIdx.x, sizeof(LocationMessage));
//            return ((LocationMessage*)&message_share[message_index]);
//        }
//        else
//        {
//            return 0;
//        }
//    }
//    __device__ LocationMessage *getNextLocationMessage(LocationMessage *message)
//    {
//        extern __shared__ int sm_data[];
//        char* message_share = (char*)&sm_data[0];
//
//        if (loadNextLocationMessage(message->_relative_bin, message->_bin_index_max, message->_central_bin, message->_bin_index))
//        {
//            int message_index = __mul24(threadIdx.x, sizeof(LocationMessage));
//            return ((LocationMessage*)&message_share[message_index]);
//        }
//        else
//        {
//            return 0;
//        }
//    }
//
//    __constant__ int ID_OFF__;
//    int *getMessageCount(){ return &h_messageCount; }
//    template <class T_EntityList>
//    __global__ void addEntitySetToBuffer(T_EntityList *entity_buffer, CollisionCore::LocationMessageList *message_buffer)
//    {
//        int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//        //Agent out of bounds
//        if (index >= COUNT__) return;
//        int message_index = index + MESSAGE_COUNT__;
//        message_buffer->position_x[message_index] = entity_buffer->position_x[index];
//        message_buffer->position_y[message_index] = entity_buffer->position_y[index];
//        message_buffer->position_z[message_index] = entity_buffer->position_z[index];
//        message_buffer->bounding_radius[message_index] = entity_buffer->bounding_radius[index];
//        message_buffer->team[message_index] = GET_TEAM_MASK(entity_buffer->states[index]);
//        message_buffer->ent_id[message_index] = entity_buffer->_id[index] + ID_OFF__;
//    }
//
//    template <class T_EntityList>
//    void postLocationMessages(T_EntityList *d_entities, int count, int id_offset)
//    {
//        if (!count)
//            return;
//        if (h_messageCount + count > LOCATION_MESSAGE_MAX){
//            printf("Error: Location message buffer would be exceeded, skipping addition of entities.\n");
//            return;
//        }
//        CUDA_CALL(cudaMemcpyToSymbol(COUNT__, &count, sizeof(int)));
//        CUDA_CALL(cudaMemcpyToSymbol(ID_OFF__, &id_offset, sizeof(int)));
//        KERNEL_PARAMS(count)
//            addEntitySetToBuffer << <grid, threads >> >(d_entities, d_messages);
//        h_messageCount += count;
//        cudaDeviceSynchronize();//Don't update device msgCount const until kernal returns
//        CUDA_CALL(cudaGetLastError());
//        CollisionCore::updateDMessageCount();
//    }
//    namespace
//    {
//        __device__ int getNextBin(int3* relative_bin)
//        {
//            int3 oldbin = make_int3(relative_bin->x, relative_bin->y, relative_bin->z);
//            if (relative_bin->x < 1)
//            {
//                relative_bin->x++;
//                return true;
//            }
//            relative_bin->x = -1;
//
//            if (relative_bin->y < 1)
//            {
//                relative_bin->y++;
//                return true;
//            }
//            relative_bin->y = -1;
//
//            if (relative_bin->z < 1)
//            {
//                relative_bin->z++;
//                return true;
//            }
//            relative_bin->z = -1;
//            return false;
//        }
//    }
//}

#define __Neighbourhood_cuh__
//#include "Asteroid.h"//Using C++ templates properly should remove the requirement of this
#undef __Neighbourhood_cuh__