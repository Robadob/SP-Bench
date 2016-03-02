#include "Neighbourhood.cuh"
#include "NeighbourhoodConstants.cuh"
#include "NeighbourhoodKernels.cuh"
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
    deviceAllocateTexture_float(&tex_locationX, d_bufferPtr, locationMessageCount, &d_tex_locationX);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, d_locationMessages->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
    deviceAllocateTexture_float(&tex_locationY, d_bufferPtr, locationMessageCount, &d_tex_locationY);
#ifdef _3D
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, d_locationMessages->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
    deviceAllocateTexture_float(&tex_locationZ, d_bufferPtr, locationMessageCount, &d_tex_locationZ);
#endif
    //PBM
    deviceAllocateTexture_int(&tex_PBM, d_PBM, getBinCount(), &d_tex_PBM);
}

void SpatialPartition::deviceAllocateTexture_float(cudaTextureObject_t *tex, float* d_data, const int size, cudaTextureObject_t *d_const)
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
    cudaMemcpyToSymbol(d_const, tex, sizeof(cudaTextureObject_t));
}
void SpatialPartition::deviceAllocateTexture_int(cudaTextureObject_t *tex, unsigned int* d_data, const int size, cudaTextureObject_t *d_const)
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
    cudaMemcpyToSymbol(d_const, tex, sizeof(cudaTextureObject_t));
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
