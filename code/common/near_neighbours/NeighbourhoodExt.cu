#include "NeighbourhoodExt.cuh"

template<class T>
SpatialPartitionExt<T>::SpatialPartitionExt<T>(unsigned int binCount, unsigned int agentCount)
    : SpatialPartition(binCount, agentCount)
{
    deviceAllocateExt(&d_ext);
    deviceAllocateExt(&d_ext_swap);
}
template<class T>
SpatialPartitionExt<T>::SpatialPartitionExt<T>(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad)
    : SpatialPartition(environmentMin, environmentMax, maxAgents, interactionRad)
{
    deviceAllocateExt(&d_ext);
    deviceAllocateExt(&d_ext_swap);
}
template<class T>
SpatialPartitionExt<T>::~SpatialPartitionExt<T>()
{
    deviceDeallocateExt(d_ext);
    deviceDeallocateExt(d_ext_swap); 
}
template<class T>
void SpatialPartitionExt<T>::deviceAllocateExt(T **d_data)
{
    CUDA_CALL(cudaMalloc(d_data, sizeof(T)));
}
template<class T>
void SpatialPartitionExt<T>::deviceDeallocateExt(T *d_data)
{
    CUDA_CALL(cudaFree(d_data));
}
#ifdef ATOMIC_PBM
template<class T>
void SpatialPartitionExt<T>::launchReorderLocationMessages()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessagesExt<T>, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    //Copy messages from d_messages to d_messages_swap, in hash order
    reorderLocationMessagesExt<T> <<<gridSize, blockSize>>>(d_keys, d_vals, d_PBM_index, d_locationMessages, d_locationMessages_swap, d_ext, d_ext_swap);
    CUDA_CHECK();
    swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}
#else
template<class T>
void SpatialPartitionExt<T>::launchReorderLocationMessages()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorderLocationMessagesExt<T>, requiredSM_reorderLocationMessages, 0);
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    //Copy messages from d_messages to d_messages_swap, in hash order
    reorderLocationMessagesExt<T> <<<gridSize, blockSize, requiredSM_reorderLocationMessages(blockSize)>>>(d_keys, d_vals, d_PBM_index, d_PBM_count, d_locationMessages, d_locationMessages_swap, d_ext, d_ext_swap);
    CUDA_CHECK();
    swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}
#endif

template<class T>
void SpatialPartitionExt<T>::swap()
{
    SpatialPartition::swap();
    //Switch d_ext and d_locationMessages_swap
    T* d_ext_temp = d_ext;
    d_ext = d_ext_swap;
    d_ext_swap = d_ext_temp;
}