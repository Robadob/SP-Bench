#include "NeighbourhoodConstants.cuh"

#if defined(MORTON)
#include "Morton.h"
#elif defined(HILBERT)
#include "Hilbert.h"
#elif defined(PEANO)
#include "Peano.h"
#elif defined(MORTON_COMPUTE)
#include "MortonCompute.h"
#endif

#include "Neighbourhood.cuh"
#include "NeighbourhoodKernels.cuh"
#include <cuda_runtime_api.h>
#include <glm/detail/func_common.hpp>

#ifndef THRUST
#include <cub/cub.cuh>
#else
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#endif
#ifdef _GL
#include <cuda_gl_interop.h>
#endif
#ifdef _DEBUG
#include <glm/gtc/epsilon.hpp>
#endif
SpatialPartition::SpatialPartition(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad)
    : maxAgents(maxAgents)
    , interactionRad(interactionRad)
    , locationMessageCount(0)
    , environmentMin(environmentMin)
    , environmentMax(environmentMax)
    , gridDim((environmentMax - environmentMin) / interactionRad)
#if defined(MORTON) || defined(HILBERT) || defined(PEANO) || defined(MORTON_COMPUTE)
    , gridExponent(0)
#endif
#ifdef _DEBUG
    , PBM_isBuilt(0)
#endif
{
    assert(interactionRad > 0);
#ifdef _DEBUG
//#if defined(_2D)
//    printf("Spatial Partition: Interaction Rad(%.3f), Grid Dims(%d,%d)\n", interactionRad, gridDim.x, gridDim.y);
//#elif defined(_3D)
//    printf("Spatial Partition: Interaction Rad(%.3f), Grid Dims(%d,%d,%d)\n", interactionRad, gridDim.x, gridDim.y, gridDim.z);
//#endif
#endif
    setBinCount();
    //Allocate bins in GPU memory
    deviceAllocateLocationMessages(&d_locationMessages, &hd_locationMessages);
    //Allocate bins swap in GPU memory
    deviceAllocateLocationMessages(&d_locationMessages_swap, &hd_locationMessages_swap);
    //Allocate PBM
    deviceAllocatePBM(&d_PBM_index, &d_PBM_count);
    //Allocate primitive structures
    deviceAllocatePrimitives(&d_keys, &d_vals);
#ifndef THRUST
#ifndef ATOMIC_PBM
    deviceAllocatePrimitives(&d_keys_swap, &d_vals_swap);
#endif
    deviceAllocateCUBTemp(&d_CUB_temp_storage, d_CUB_temp_storage_bytes);
#endif
    //Allocate tex
    deviceAllocateTextures();
    //Set device constants
    CUDA_CALL(cudaMemcpyToSymbol(d_interactionRad, &interactionRad, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &gridDim, sizeof(DIMENSIONS_IVEC)));
    DIMENSIONS_VEC t_gridDim = (DIMENSIONS_VEC)gridDim;
    CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &t_gridDim, sizeof(DIMENSIONS_VEC)));

    CUDA_CALL(cudaMemcpyToSymbol(d_environmentMin, &environmentMin, sizeof(DIMENSIONS_VEC)));
    CUDA_CALL(cudaMemcpyToSymbol(d_environmentMax, &environmentMax, sizeof(DIMENSIONS_VEC)));
#if defined(GLOBAL_PBM) || defined(LDG_PBM)
    CUDA_CALL(cudaMemcpyToSymbol(d_pbm_index, &d_PBM_index, sizeof(unsigned int *)));
    CUDA_CALL(cudaMemcpyToSymbol(d_pbm_count, &d_PBM_count, sizeof(unsigned int *)));
#endif
#ifdef _DEBUG
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif
    setLocationCount(locationMessageCount);
    unsigned int t_binCount = this->binCountMax;
    CUDA_CALL(cudaMemcpyToSymbol(d_binCount, &t_binCount, sizeof(unsigned int)));

#if defined(_GL) || defined(_DEBUG)
    CUDA_CALL(cudaMemcpyToSymbol(d_locationMessagesA, &d_locationMessages, sizeof(LocationMessages *)));
    CUDA_CALL(cudaMemcpyToSymbol(d_locationMessagesB, &d_locationMessages_swap, sizeof(LocationMessages *)));
#endif
    //Init lookup table
#if defined(MORTON)
    initMorton(gridDim);
#elif defined(HILBERT)
    initHilbert(gridDim);
#elif defined(PEANO)
    initPeano(gridDim);
#endif
#ifdef MODULAR
    assert(glm::compMax(gridDim)<MODULAR_OFFSETS_MAX);
#if defined(_2D)
    unsigned char h_offsets[MODULAR_OFFSETS_MAX][MODULAR_OFFSETS_MAX];
    for (unsigned int x = 0; x<MODULAR_OFFSETS_MAX; ++x)
    {
        for (unsigned int y = 0; y<MODULAR_OFFSETS_MAX; ++y)
        {
            int a = ((((-x+1)%3)+3)%3);
            int b = ((((-y+1)%3)+3)%3);
            h_offsets[x][y] = (a & 3) + ((b & 3) << 2);
        }
    }
    CUDA_CALL(cudaMemcpyToSymbol(d_offsets, &h_offsets, sizeof(unsigned char[MODULAR_OFFSETS_MAX][MODULAR_OFFSETS_MAX])));
#elif defined(_3D)
    unsigned char h_offsets[MODULAR_OFFSETS_MAX][MODULAR_OFFSETS_MAX][MODULAR_OFFSETS_MAX];
    for (unsigned int x = 0; x<MODULAR_OFFSETS_MAX; ++x)
    {
        for (unsigned int y = 0; y<MODULAR_OFFSETS_MAX; ++y)
        {
            for (unsigned int z = 0; z<MODULAR_OFFSETS_MAX; ++z)
            {
                int a = ((((-x+1)%3)+3)%3);
                int b = ((((-y+1)%3)+3)%3);
                int c = ((((-z+1)%3)+3)%3);
                h_offsets[x][y][z] = (a & 3) + ((b & 3) << 2) + ((c&3) << 4);
            }
        }
    }
    CUDA_CALL(cudaMemcpyToSymbol(d_offsets, &h_offsets, sizeof(unsigned char[MODULAR_OFFSETS_MAX][MODULAR_OFFSETS_MAX])));
#endif
#endif
}
SpatialPartition::~SpatialPartition()
{
    CUDA_CHECK();
    //Dellocate bins in GPU memory
    deviceDeallocateLocationMessages(d_locationMessages, hd_locationMessages);
    //Dellocate bins swap in GPU memory
    deviceDeallocateLocationMessages(d_locationMessages_swap, hd_locationMessages_swap);
    //Dellocate PBM
    deviceDeallocatePBM(d_PBM_index, d_PBM_count);
    //Deallocated primitive structures
    deviceDeallocatePrimitives(d_keys, d_vals);
#ifndef THRUST
#ifndef ATOMIC_PBM
    deviceDeallocatePrimitives(d_keys_swap, d_vals_swap);
#endif
    deviceDeallocateCUBTemp(d_CUB_temp_storage);
#endif
    //Deallocate tex
    deviceDeallocateTextures();
    //Free lookup table
#if defined(MORTON)
    freeMorton();
#elif defined(HILBERT)
    freeHilbert();
#elif defined(PEANO)
    freePeano();
#endif
}
unsigned int SpatialPartition::getHash(DIMENSIONS_IVEC gridPos)
{//Host version using host copy of gridDim
    gridPos = glm::clamp(gridPos, DIMENSIONS_IVEC(0), gridDim - DIMENSIONS_IVEC(1));
#if defined(MORTON)
    return h_mortonEncode(gridPos);
#elif defined(HILBERT)
    return h_hilbertEncode(gridPos);
#elif defined(PEANO)
    return h_peanoEncode(gridPos);
#elif defined(MORTON_COMPUTE)
    return mortonComputeEncode(gridPos);
#else
    return
#ifdef _3D
        (gridPos.z * gridDim.y * gridDim.x) +   //z
#endif
        (gridPos.y * gridDim.x) +					//y
        gridPos.x;
#endif
}
DIMENSIONS_IVEC SpatialPartition::getPos(unsigned int hash)
{
#if defined(MORTON)
    return mortonDecode(hash);
#elif defined(HILBERT)
    return hilbertDecode(hash, this->gridExponent);
#elif defined(PEANO)
    return peanoDecode(hash, this->gridExponent);
#elif defined(MORTON_COMPUTE)
    return mortonComputeDecode(hash);
#else
    if (hash >= this->binCountMax)
        return DIMENSIONS_IVEC(-1);
    else
    {
#ifdef _3D

        int z = (hash / (gridDim.y * gridDim.x));
        int y = (hash % (gridDim.y * gridDim.x)) / gridDim.x;
        int x = (hash % (gridDim.y * gridDim.x)) % gridDim.x;
        return DIMENSIONS_IVEC(x, y, z);
#else
        int y = hash / gridDim.x;
        int x = hash % gridDim.x;
        return DIMENSIONS_IVEC(x, y);
#endif
    }
#endif
}
bool SpatialPartition::isValid(DIMENSIONS_IVEC bin) const
{
    if (
#ifdef _3D
        bin.z<0 || bin.z >= gridDim.z ||
#endif
        bin.y<0 || bin.y >= gridDim.y ||
        bin.x<0 || bin.x >= gridDim.x
        )
    {
        return false;
    }
    return true;
}
#ifdef _DEBUG
void SpatialPartition::assertSearch()
{
//    //return;//
//    unsigned int outCount = this->binCountMax;
//    //unsigned int tableSize = ((outCount / 10) + 1) * 10;
//
//    //Copy raw PBM from device to host
//    unsigned int *PBM_raw = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*outCount));
//    memset(PBM_raw, 0, outCount * sizeof(unsigned int));
//    CUDA_CALL(cudaMemcpy(PBM_raw, d_PBM_index, sizeof(unsigned int)*outCount, cudaMemcpyDeviceToHost));
//    //Calculate the size of every bin
//    unsigned int agtCount = 0;
//    unsigned int *PBM_binSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*this->binCountMax));
//    for (unsigned int i = 0; i < this->binCountMax; i++)
//    {
//            PBM_binSize[i] = PBM_raw[i + 1] - PBM_raw[i];
//            agtCount += PBM_binSize[i];
//    }
//    if (agtCount != maxAgents&&agtCount != 0)
//    {
//        printf("%i PBM records exist for %i agents.\n", agtCount, maxAgents);
//    }
//    //Calculate the size of each bin's neighbourhood
//    unsigned int *PBM_neighbourhoodSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*this->binCountMax));
//    for (unsigned int i = 0; i < this->binCountMax; i++)
//    {
//        PBM_neighbourhoodSize[i] = 0;
//        DIMENSIONS_IVEC curCell = getPos(i);
//        for (int x = -1; x <= 1; x++)
//            for (int y = -1; y <= 1; y++)
//            {
//#if defined(_2D)
//                DIMENSIONS_IVEC neighbourCell = curCell + DIMENSIONS_IVEC(x, y);
//#elif defined(_3D)
//                for (int z = -1; z <= 1; z++)
//                {
//                    DIMENSIONS_IVEC neighbourCell = curCell + DIMENSIONS_IVEC(x, y, z);
//#endif
//                    if (isValid(neighbourCell))
//                    {
//                        unsigned int hash = getHash(neighbourCell);
//                        assert(hash < this->binCountMax);
//                        PBM_neighbourhoodSize[i] += PBM_binSize[hash];
//                    }
//#ifdef _3D
//                }
//#endif
//            }
//
//    }
//    //Copy every location and neighbour count from device to host
//    float *d_bufferPtr;
//    LocationMessages lm;
//    lm.locationX = (float*)malloc(sizeof(float)*locationMessageCount);
//    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(lm.locationX, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
//    lm.locationY = (float*)malloc(sizeof(float)*locationMessageCount);
//    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(lm.locationY, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
//#ifdef _3D
//    lm.locationZ = (float*)malloc(sizeof(float)*locationMessageCount);
//    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(lm.locationZ, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
//#endif
//    lm.count = (float*)malloc(sizeof(float)*locationMessageCount);
//    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->count, sizeof(float*), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(lm.count, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
//    //ASSERT: Every agent searched the right amount of neighbours
//    unsigned int matchFails = 0;
//    for (unsigned int i = 0; i < locationMessageCount; i++)
//    {
//        //For rendering purposes the count is stored as count/totalMessages, invert this math for assertion
//#if defined(_2D)
//        unsigned int hash = getHash(getGridPosition(DIMENSIONS_VEC(lm.locationX[i], lm.locationY[i])));
//#elif defined(_3D)
//        unsigned int hash = getHash(getGridPosition(DIMENSIONS_VEC(lm.locationX[i], lm.locationY[i], lm.locationZ[i])));
//#endif
//
//        assert(hash < this->binCountMax);
//        if (glm::epsilonNotEqual(lm.count[i], PBM_neighbourhoodSize[hash] / (float)locationMessageCount, 0.5f))
//        {
//            //printf("%u=%u-%f=%f,", (unsigned int)(lm.count[i] * locationMessageCount), PBM_neighbourhoodSize[hash], lm.count[i], PBM_neighbourhoodSize[hash] / (float)locationMessageCount);
//            matchFails++;
//        }
//    }
//    //Free location/count data
//    free(lm.locationX);
//    free(lm.locationY);
//#ifdef _3D
//    free(lm.locationZ);
//#endif
//    free(lm.count);
//    if (matchFails>0)
//    {
//        printf("ERROR: Neighbour search totals do not match (%u/%u)\n", matchFails, locationMessageCount);
//    }
//    else
//    {
//        free(PBM_raw);
//        free(PBM_binSize);
//        free(PBM_neighbourhoodSize);
//        return;
//    }
//    //Output the 3 PBM_ data structures to file in a readable format
//    FILE *file = fopen("../logs/PBM.txt", "w");
//    fprintf(file, "ERROR: Neighbour search totals do not match (%u/%u)\n", matchFails, locationMessageCount);
//    fprintf(file, "Raw PBM\n");
//    fprintf(file, "|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
//    fprintf(file, "|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
//    for (unsigned int i = 0; i < (outCount / 10) - 1; i++)
//    {
//        fprintf(file, "|%4u0|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i,
//            PBM_raw[(10 * i) + 0],
//            PBM_raw[(10 * i) + 1],
//            PBM_raw[(10 * i) + 2],
//            PBM_raw[(10 * i) + 3],
//            PBM_raw[(10 * i) + 4],
//            PBM_raw[(10 * i) + 5],
//            PBM_raw[(10 * i) + 6],
//            PBM_raw[(10 * i) + 7],
//            PBM_raw[(10 * i) + 8],
//            PBM_raw[(10 * i) + 9]
//            );
//    }
//    fprintf(file, "Bin Size\n");
//    fprintf(file, "|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
//    fprintf(file, "|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
//    for (unsigned int i = 0; i < ((this->binCount + 1) / 10) - 1; i++)
//    {
//        fprintf(file, "|%4u0|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i,
//            PBM_binSize[(10 * i) + 0],
//            PBM_binSize[(10 * i) + 1],
//            PBM_binSize[(10 * i) + 2],
//            PBM_binSize[(10 * i) + 3],
//            PBM_binSize[(10 * i) + 4],
//            PBM_binSize[(10 * i) + 5],
//            PBM_binSize[(10 * i) + 6],
//            PBM_binSize[(10 * i) + 7],
//            PBM_binSize[(10 * i) + 8],
//            PBM_binSize[(10 * i) + 9]
//            );
//    }
//    fprintf(file, "Neighbour Count\n");
//    fprintf(file, "|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
//    fprintf(file, "|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
//    for (unsigned int i = 0; i < (outCount / 10) - 1; i++)
//    {
//        fprintf(file, "|%4u0|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i,
//            PBM_neighbourhoodSize[(10 * i) + 0],
//            PBM_neighbourhoodSize[(10 * i) + 1],
//            PBM_neighbourhoodSize[(10 * i) + 2],
//            PBM_neighbourhoodSize[(10 * i) + 3],
//            PBM_neighbourhoodSize[(10 * i) + 4],
//            PBM_neighbourhoodSize[(10 * i) + 5],
//            PBM_neighbourhoodSize[(10 * i) + 6],
//            PBM_neighbourhoodSize[(10 * i) + 7],
//            PBM_neighbourhoodSize[(10 * i) + 8],
//            PBM_neighbourhoodSize[(10 * i) + 9]
//            );
//    }
//    //Cleanup resources
//    fclose(file);
//    free(PBM_raw);
//    free(PBM_binSize);
//    free(PBM_neighbourhoodSize);
}
#endif
NeighbourhoodStats SpatialPartition::getNeighbourhoodStats()
{//Based on assertSearch()
    NeighbourhoodStats rtn;
//    //Copy raw PBM from device to host
//    unsigned int *PBM_raw = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*(this->binCountMax + 1)));
//    memset(PBM_raw, 0, (this->binCountMax + 1) * sizeof(unsigned int));
//    CUDA_CALL(cudaMemcpy(PBM_raw, d_PBM, sizeof(unsigned int)*(this->binCountMax + 1), cudaMemcpyDeviceToHost));
//
//    //Calculate the size of every bin
//    unsigned int agtCount = 0;
//    unsigned int *PBM_binSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*this->binCountMax));
//    for (unsigned int i = 0; i < this->binCountMax; i++)
//    {
//        PBM_binSize[i] = PBM_raw[i + 1] - PBM_raw[i];
//        agtCount += PBM_binSize[i];
//    }
//    free(PBM_raw);
//    if (agtCount != maxAgents&&agtCount != 0)
//    {
//        printf("%i PBM records exist for %i agents.\n", agtCount, maxAgents);
//    }
//    //Calculate the size of each bin's neighbourhood
//    unsigned int *PBM_neighbourhoodSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*this->binCountMax));
//    for (unsigned int i = 0; i < this->binCountMax; i++)
//    {
//        PBM_neighbourhoodSize[i] = 0;
//        DIMENSIONS_IVEC curCell = getPos(i);
//        for (int x = -1; x <= 1; x++)
//            for (int y = -1; y <= 1; y++)
//            {
//#if defined(_2D)
//            DIMENSIONS_IVEC neighbourCell = curCell + DIMENSIONS_IVEC(x, y);
//#elif defined(_3D)
//            for (int z = -1; z <= 1; z++)
//            {
//                DIMENSIONS_IVEC neighbourCell = curCell + DIMENSIONS_IVEC(x, y, z);
//#endif
//                if (isValid(neighbourCell))
//                {
//                    unsigned int hash = getHash(neighbourCell);
//                    assert(hash < this->binCountMax);
//                    PBM_neighbourhoodSize[i] += PBM_binSize[hash];
//                }
//#ifdef _3D
//            }
//#endif
//            }
//        if (PBM_binSize[i])
//        {
//            rtn.min = min(PBM_neighbourhoodSize[i], rtn.min);
//            rtn.max = max(PBM_neighbourhoodSize[i], rtn.max);
//        }
//        rtn.average += ((PBM_neighbourhoodSize[i] * PBM_binSize[i]) / (float)maxAgents);
//    }
//    //Calculate standard deviation
//    double valMinAvqSqAvg = 0.0;
//    for (unsigned int i = 0; i < this->binCountMax; i++)
//    {
//        valMinAvqSqAvg += pow(PBM_neighbourhoodSize[i] - rtn.average, 2.0)* PBM_binSize[i] / maxAgents;
//    }
//    rtn.standardDeviation = (float)sqrt(valMinAvqSqAvg);
//    free(PBM_binSize);
//    free(PBM_neighbourhoodSize);
    return rtn;
}
void SpatialPartition::deviceAllocateLocationMessages(LocationMessages **d_locMessage, LocationMessages *hd_locMessage)
{
    CUDA_CALL(cudaMalloc(d_locMessage, sizeof(LocationMessages)));
#ifdef AOS_MESSAGES
    CUDA_CALL(cudaMalloc(&hd_locMessage->location, sizeof(DIMENSIONS_VEC)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->location), &(hd_locMessage->location), sizeof(DIMENSIONS_VEC*), cudaMemcpyHostToDevice));
#else
    CUDA_CALL(cudaMalloc(&hd_locMessage->locationX, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationX), &(hd_locMessage->locationX), sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&hd_locMessage->locationY, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationY), &(hd_locMessage->locationY), sizeof(float*), cudaMemcpyHostToDevice));
#ifdef _3D
    CUDA_CALL(cudaMalloc(&hd_locMessage->locationZ, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationZ), &(hd_locMessage->locationZ), sizeof(float*), cudaMemcpyHostToDevice));
#endif
#endif
#if defined(_GL) || defined(_DEBUG)
    CUDA_CALL(cudaMalloc(&hd_locMessage->count, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->count), &(hd_locMessage->count), sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(hd_locMessage->count, 0, sizeof(float)*maxAgents));//Must be 0'd to protect assertions on k20
#endif
}
void SpatialPartition::deviceAllocatePBM(unsigned int **d_PBM_index_t, unsigned int **d_PBM_count_t)
{
    CUDA_CALL(cudaMalloc(d_PBM_index_t, sizeof(unsigned int)*this->binCountMax));
    CUDA_CALL(cudaMemset(*d_PBM_index_t, 0, sizeof(unsigned int)*this->binCountMax));//Must be 0'd to protect assertions on k20
    CUDA_CALL(cudaMalloc(d_PBM_count_t, sizeof(unsigned int)*this->binCountMax));
    CUDA_CALL(cudaMemset(*d_PBM_count_t, 0, sizeof(unsigned int)*this->binCountMax));//Must be 0'd to protect assertions on k20
}
void SpatialPartition::deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals)
{
    CUDA_CALL(cudaMalloc(d_keys, sizeof(unsigned int)*maxAgents));
    CUDA_CALL(cudaMalloc(d_vals, sizeof(unsigned int)*maxAgents));
}
#ifndef THRUST
void SpatialPartition::deviceAllocateCUBTemp(void **d_CUB_temp, size_t &d_cub_temp_bytes)
{
    //CUB version
    // Determine temporary device storage requirements
    d_cub_temp_bytes = 0;
    *d_CUB_temp = NULL;
#ifdef ATOMIC_PBM
    cub::DeviceScan::ExclusiveSum(*d_CUB_temp, d_cub_temp_bytes, d_PBM_count, d_PBM_index, binCountMax);
#else
    cub::DeviceRadixSort::SortPairs(*d_CUB_temp, d_cub_temp_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, maxAgents);
#endif
    // Allocate temporary storage
    CUDA_CALL(cudaMalloc(d_CUB_temp, d_cub_temp_bytes));
}
#endif
void SpatialPartition::deviceAllocateTextures()
{
    //Locations
#ifdef _GL
#ifdef AOS_MESSAGES
#error TODO Allocate AOS message texture
#else
#pragma unroll 3
    for (unsigned int i = 0; i < DIMENSIONS; i++)
        deviceAllocateGLTexture_float(i);
#endif
    deviceAllocateGLTexture_float2();//Allocate a texture to store counting info in (Used to colour the visualisation
#else
#if !defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
#error TODO Allocate AOS message texture
#else
    for (unsigned int i = 0; i < DIMENSIONS; i++)
        deviceAllocateTexture_float(i);
#endif
#endif
#endif
    //PBM
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
    deviceAllocateTexture_int();
#endif
}
void SpatialPartition::fillTextures()
{
#if defined(_GL) ||(!defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES))
#ifdef AOS_MESSAGES
#error TODO Fill AOS message texture
#else
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[0], hd_locationMessages.locationX, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[1], hd_locationMessages.locationY, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#ifdef _3D
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[2], hd_locationMessages.locationZ, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#endif
#endif
#endif
#ifdef _GL
    CUDA_CALL(cudaMemcpy(tex_location_ptr_count, hd_locationMessages.count, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#endif
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
    CUDA_CALL(cudaMemcpy(tex_PBM_index_ptr, d_PBM_index, this->binCountMax*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(tex_PBM_count_ptr, d_PBM_count, this->binCountMax*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
#endif
}

#if !defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES)
void SpatialPartition::deviceAllocateTexture_float(unsigned int i)
{
    if (i >= DIMENSIONS)
        return;
    //Allocate cuda array
    CUDA_CALL(cudaMalloc(&tex_loc_ptr[i], maxAgents*sizeof(float)));
    //Define cuda resource from array
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = tex_loc_ptr[i];
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = maxAgents*sizeof(float);
    //Define a cuda texture format
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;
    //Create texture obj
    CUDA_CALL(cudaCreateTextureObject(&tex_location[i], &resDesc, &texDesc, NULL));
    //Copy obj to const memory
    CUDA_CALL(cudaMemcpyToSymbol(d_tex_location, &tex_location[i], sizeof(cudaTextureObject_t), i*sizeof(cudaTextureObject_t)));
}
#endif

#ifdef _GL
void SpatialPartition::deviceAllocateGLTexture_float(unsigned int i)//GLuint *glTex, GLuint *glTbo, cudaGraphicsResource_t *cuGres, cudaArray_t *cuArr, cudaTextureObject_t *tex, cudaTextureObject_t *d_const, const unsigned int size)
{
    if (i >= DIMENSIONS)
        return;
    float *data = new float[maxAgents];
    //Gen tex
    GL_CALL(glGenTextures(1, &gl_tex[i]));
    //Gen buffer
    GL_CALL(glGenBuffers(1, &gl_tbo[i]));
    //Size buffer and tie to tex
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, gl_tbo[i]));
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, gl_tbo[i]));
    GL_CALL(glBufferData(GL_TEXTURE_BUFFER, maxAgents*sizeof(float), 0, GL_STATIC_DRAW));

    GL_CALL(glBindTexture(GL_TEXTURE_BUFFER, gl_tex[i]));
    //glBindTexture(GL_TEXTURE_2D, 0);
    GL_CALL(glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, gl_tbo[i]));
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, 0));
    GL_CALL(glBindTexture(GL_TEXTURE_BUFFER, 0));

    //Get CUDA handle to texture
    memset(&gl_gRes[i], 0, sizeof(cudaGraphicsResource_t));
    CUDA_CALL(cudaGraphicsGLRegisterBuffer(&gl_gRes[i], gl_tbo[i], cudaGraphicsMapFlagsNone));//GL_TEXTURE_BUFFER IS UNDOCUMENTED
    //Map/convert this to something CUarray
    CUDA_CALL(cudaGraphicsMapResources(1, &gl_gRes[i]));
    CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void**)&tex_loc_ptr[i], 0, gl_gRes[i]));
    CUDA_CALL(cudaGraphicsUnmapResources(1, &gl_gRes[i], 0));
    //Create a texture object from the CUarray
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = tex_loc_ptr[i];
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = maxAgents*sizeof(float);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;
    CUDA_CALL(cudaCreateTextureObject(&tex_location[i], &resDesc, &texDesc, nullptr));
    //Copy texture object to device constant
    CUDA_CALL(cudaMemcpyToSymbol(d_tex_location, &tex_location[i], sizeof(cudaTextureObject_t), i*sizeof(cudaTextureObject_t)));
    delete data;
}
#endif
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
/*
Allocates the PBM texture, which is only accessed via CUDA & memcpy
*/
void SpatialPartition::deviceAllocateTexture_int()
{
    {//PBM index
        //Define cuda array format
        //Allocate cuda array
        unsigned int size = this->binCountMax;
        CUDA_CALL(cudaMalloc(&tex_PBM_index_ptr, size*sizeof(unsigned int)));//Number of elements, not bytes
        //Define cuda resource from array
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = tex_PBM_index_ptr;
        resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = size*sizeof(unsigned int);

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.readMode = cudaReadModeElementType;

        CUDA_CALL(cudaCreateTextureObject(&tex_PBM_index, &resDesc, &texDesc, NULL));
        CUDA_CALL(cudaMemcpyToSymbol(d_tex_PBM_index, &tex_PBM_index, sizeof(cudaTextureObject_t)));
    }
    {//PBM count
        //Define cuda array format
        //Allocate cuda array
        unsigned int size = this->binCountMax;
        CUDA_CALL(cudaMalloc(&tex_PBM_count_ptr, size*sizeof(unsigned int)));//Number of elements, not bytes
        //Define cuda resource from array
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = tex_PBM_count_ptr;
        resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = size*sizeof(unsigned int);

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.readMode = cudaReadModeElementType;

        CUDA_CALL(cudaCreateTextureObject(&tex_PBM_count, &resDesc, &texDesc, NULL));
        CUDA_CALL(cudaMemcpyToSymbol(d_tex_PBM_count, &tex_PBM_count, sizeof(cudaTextureObject_t)));
    }
}
#endif
#ifdef _GL
/*
Allocates the count texture, which is only accessed via memcpy & GL
*/
void SpatialPartition::deviceAllocateGLTexture_float2()
{
    int *data = new int[maxAgents];
    //Gen tex
    GL_CALL(glGenTextures(1, &gl_tex_count));
    //Gen buffer
    GL_CALL(glGenBuffers(1, &gl_tbo_count));
    //Size buffer and tie to tex
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, gl_tbo_count));
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, gl_tbo_count));
    GL_CALL(glBufferData(GL_TEXTURE_BUFFER, maxAgents*sizeof(float), 0, GL_STATIC_DRAW));

    GL_CALL(glBindTexture(GL_TEXTURE_BUFFER, gl_tex_count));
    //glBindTexture(GL_TEXTURE_2D, 0);
    GL_CALL(glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, gl_tbo_count));
    GL_CALL(glBindBuffer(GL_TEXTURE_BUFFER, 0));
    GL_CALL(glBindTexture(GL_TEXTURE_BUFFER, 0));

    //Get CUDA handle to texture
    memset(&gl_gRes_count, 0, sizeof(cudaGraphicsResource_t));
    CUDA_CALL(cudaGraphicsGLRegisterBuffer(&gl_gRes_count, gl_tbo_count, cudaGraphicsMapFlagsNone));
    //Map/convert this to something CUarray
    CUDA_CALL(cudaGraphicsMapResources(1, &gl_gRes_count));
    CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void**)&tex_location_ptr_count, 0, gl_gRes_count));
    CUDA_CALL(cudaGraphicsUnmapResources(1, &gl_gRes_count, 0));
    delete data;
}
#endif
void SpatialPartition::deviceDeallocateLocationMessages(LocationMessages *d_locMessage, LocationMessages hd_locMessage)
{
#ifdef AOS_MESSAGES
    CUDA_CALL(cudaFree(hd_locMessage.location));
#else
    CUDA_CALL(cudaFree(hd_locMessage.locationX));
    CUDA_CALL(cudaFree(hd_locMessage.locationY));
#ifdef _3D
    CUDA_CALL(cudaFree(hd_locMessage.locationZ));
#endif
#endif
#ifdef _GL
    CUDA_CALL(cudaFree(hd_locMessage.count));
#endif
    CUDA_CALL(cudaFree(d_locMessage));
}
void SpatialPartition::deviceDeallocatePBM(unsigned int *d_PBM_index_t, unsigned int *d_PBM_count_t)
{
    CUDA_CALL(cudaFree(d_PBM_index_t));
    CUDA_CALL(cudaFree(d_PBM_count_t));
}
void SpatialPartition::deviceDeallocatePrimitives(unsigned int *d_keys, unsigned int *d_vals)
{
    CUDA_CALL(cudaFree(d_keys));
    CUDA_CALL(cudaFree(d_vals));
}
#ifndef THRUST
void SpatialPartition::deviceDeallocateCUBTemp(void *d_CUB_temp)
{
    CUDA_CALL(cudaFree(d_CUB_temp));
}
#endif
void SpatialPartition::deviceDeallocateTextures()
{

#if defined(_GL) ||(!defined(GLOBAL_MESSAGES) && !defined(LDG_MESSAGES))
#ifdef AOS_MESSAGES
#error TODO Deallocate AOS message texture
#else
    for (unsigned int i = 0; i < DIMENSIONS; i++)
    {
        cudaDestroyTextureObject(tex_location[i]);
#ifdef _GL
        cudaGraphicsUnregisterResource(gl_gRes[i]);
        GL_CALL(glDeleteBuffers(1, &gl_tbo[i]));
        GL_CALL(glDeleteTextures(1, &gl_tex[i]));
#else
        cudaFree(tex_loc_ptr[i]);
#endif
    }
#endif
#endif
#if !(defined(GLOBAL_PBM) || defined(LDG_PBM))
    cudaDestroyTextureObject(tex_PBM_index);
    cudaDestroyTextureObject(tex_PBM_count);
    cudaFree(tex_PBM_index_ptr);
    cudaFree(tex_PBM_count_ptr);
#endif
#ifdef _GL
    cudaGraphicsUnregisterResource(gl_gRes_count);
    GL_CALL(glDeleteBuffers(1, &gl_tbo_count));
    GL_CALL(glDeleteTextures(1, &gl_tex_count));
#endif
}

unsigned int SpatialPartition::getBinCount() const
{
    return binCountMax;
}

void SpatialPartition::setBinCount()
{
    //Get max grid dimension
    this->binCount = glm::compMax(gridDim);
    //Find the next biggest power of two
#if defined(MORTON) || defined(HILBERT) || defined(MORTON_COMPUTE)
    this->gridExponent = (unsigned int)ceil(log2f((float)this->binCount));
    int l2 = (int)pow(2, this->gridExponent);
    this->binCountMax = (unsigned int)pow(l2, DIMENSIONS);
#elif defined(PEANO)
    this->gridExponent =ceil(log(this->binCount) / log(3));
    int l3 = pow(3, this->gridExponent);
    this->binCountMax = (unsigned int)pow(l3, DIMENSIONS);
#else
    this->binCountMax = (unsigned int)pow(this->binCount, DIMENSIONS);
#endif
    this->binCount = (unsigned int)pow(this->binCount, DIMENSIONS);
    this->binCountBits = (unsigned long)ceil(log(this->binCountMax) / log(2));
//#if defined(_DEBUG) &&(defined(MORTON) || defined(HILBERT) ||defined(PEANO))
//    printf("Space-filling grid exponent set to: %u\n", this->gridExponent);
//    printf("Bin Count Max %u\n", this->binCountMax);
//#endif
}
void SpatialPartition::setLocationCount(unsigned int t_locationMessageCount)
{
    //Set local copy
    locationMessageCount = t_locationMessageCount;
    //Set device constants
    CUDA_CALL(cudaMemcpyToSymbol(d_locationMessageCount, &locationMessageCount, sizeof(unsigned int)));
}

#ifdef ATOMIC_PBM
void SpatialPartition::launchAtomicHistogram()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    //Keys = bin_index
    //Vals = bin_sub_index
    //Histogram into
    atomicHistogram<<<gridSize, blockSize>>>(d_keys, d_vals, d_PBM_count, d_locationMessages);
    CUDA_CALL(cudaDeviceSynchronize());
}
void SpatialPartition::launchReorderLocationMessages()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    //Copy messages from d_messages to d_messages_swap, in hash order
    reorderLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM_index, d_locationMessages, d_locationMessages_swap);
    CUDA_CHECK();
    swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}
#else
void SpatialPartition::launchHashLocationMessages()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, hashLocationMessages, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    hashLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_locationMessages);
    CUDA_CALL(cudaDeviceSynchronize());
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
    reorderLocationMessages << <gridSize, blockSize, requiredSM_reorderLocationMessages(blockSize) >> >(d_keys, d_vals, d_PBM_index, d_PBM_count, d_locationMessages, d_locationMessages_swap);
    CUDA_CHECK();
    swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}
#endif
#ifdef _DEBUG
void SpatialPartition::launchAssertPBMIntegerity()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, assertPBMIntegrity, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (this->binCountMax + blockSize - 1) / blockSize;
    //Copy messages from d_messages to d_messages_swap, in hash order
    assertPBMIntegrity << <gridSize, blockSize >> >();
    //No sync, called directly after textures have been updated
}
#endif
void SpatialPartition::swap()
{
    //Switch d_locationMessages and d_locationMessages_swap
    LocationMessages* d_locationmessages_temp = d_locationMessages;
    d_locationMessages = d_locationMessages_swap;
    d_locationMessages_swap = d_locationmessages_temp;
    //Switch hd_locationMessages and hd_locationMessages_swap
    LocationMessages hd_locationmessages_temp = hd_locationMessages;
    hd_locationMessages = hd_locationMessages_swap;
    hd_locationMessages_swap = hd_locationmessages_temp;

//Update active message list
#if defined(GLOBAL_MESSAGES) || defined(LDG_MESSAGES)
    CUDA_CALL(cudaMemcpyToSymbol(d_messages, &d_locationMessages, sizeof(LocationMessages*)));
#endif

#ifdef _DEBUG
    PBM_isBuilt = 0;
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif
}
PBM_Time SpatialPartition::buildPBM()
{
    cudaEvent_t start_overall, start_reorder, start_texcopy, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_reorder);
    cudaEventCreate(&start_texcopy);
    cudaEventCreate(&stop_overall);
    PBM_Time p = {0,0};
    //If no messages, or instances, don't bother
    if (locationMessageCount<1) return p;
#ifdef _DEBUG
    static bool __first = true;
    if (!__first)
        assertSearch();
    if (__first)
        __first = false;
#endif

    //Start overall timer
    cudaEventRecord(start_overall);
#ifdef ATOMIC_PBM
    //Reset PBM
    CUDA_CALL(cudaMemset(d_PBM_count, 0x00000000, this->binCountMax * sizeof(unsigned int)));
    //Build histogram using atomics
    launchAtomicHistogram();
    //prefix sum PBM to convert counts to indices
#ifndef THRUST
    // Run exclusive prefix min-scan
    cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_PBM_count, d_PBM_index, this->binCountMax);
#else
#error Thrust prefix sum not implemented/tested
    //If we do implement this, make sure we exclusive scan from d_PBM_count into d_PBM_index, else use of d_PBM_count will be incorrect
    thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_location_partition_matrix->end_or_count);
    thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_location_partition_matrix->start);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_location_grid_size, ptr_index); // scan
#endif
    //Start Reorder timer
    cudaEventRecord(start_reorder);
    //Reorder messages
    launchReorderLocationMessages();
#else
    //Fill primitive key/val arrays for sort
    launchHashLocationMessages();
    //Sort key val arrays using thrust/CUB
#ifndef THRUST
    ////CUB version
    //// Determine temporary device storage requirements
    //void *d_temp_storage = NULL;
    //size_t   temp_storage_bytes = 0;
    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, locationMessageCount);
    //// Allocate temporary storage
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_keys, d_keys_swap, d_vals, d_vals_swap, locationMessageCount, 0, this->binCountBits);
    //Swap arrays
    unsigned int *temp;
    temp = d_keys;
    d_keys = d_keys_swap;
    d_keys_swap = temp;
    temp = d_vals;
    d_vals = d_vals_swap;
    d_vals_swap = temp;
    ////Free temporary memory
    //cudaFree(d_temp_storage);
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
    unsigned int binCount = this->binCountMax;
    CUDA_CALL(cudaMemset(d_PBM_index, 0xffffffff, binCount * sizeof(unsigned int)));
    //Start Reorder timer
    cudaEventRecord(start_reorder);    
    //Reorder messages and create PBM index
    launchReorderLocationMessages();
#endif
    //Start Tex Copy
    cudaEventRecord(start_texcopy);
    //Clone data to textures ready for neighbourhood search
    fillTextures();
    //End overall timer
    cudaEventRecord(stop_overall);
#ifdef _DEBUG
    launchAssertPBMIntegerity();
    PBM_isBuilt = 1;
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif

    //Calculate return struct
    cudaEventSynchronize(stop_overall);
    cudaEventElapsedTime(&p.sort, start_overall, start_reorder);
    cudaEventElapsedTime(&p.reorder, start_reorder, start_texcopy);
    cudaEventElapsedTime(&p.texcopy, start_texcopy, stop_overall);
    //Cleanup
    cudaEventDestroy(start_overall);
    cudaEventDestroy(start_reorder);
    cudaEventDestroy(start_texcopy);
    cudaEventDestroy(stop_overall);
    return p;
}

int SpatialPartition::requiredSM(int blockSize)
{
#if !defined(SHARED_BINSTATE)
    return 0
#else
    return blockSize*sizeof(LocationMessage)
#endif
//#if defined(MODULAR) //BlockRelative + BlockContinue
//        + sizeof(DIMENSIONS_IVEC) + sizeof(bool)
//#elif defined(MODULAR_STRIPS) //BlockRelative + BlockContinue
//        +sizeof(DIMENSIONS_IVEC_MINUS1) + sizeof(bool)
//#endif
    ;
}
DIMENSIONS_IVEC SpatialPartition::getGridPosition(DIMENSIONS_VEC worldPos)
{
#ifndef SP_NO_CLAMP_GRID
    //Clamp each grid coord to 0<=x<dim
    return clamp(floor(((worldPos - environmentMin) / (environmentMax - environmentMin))*DIMENSIONS_VEC(gridDim)), DIMENSIONS_VEC(0), DIMENSIONS_VEC(gridDim) - DIMENSIONS_VEC(1));
#else
    return floor(((worldPos - environmentMin) / (environmentMax - environmentMin))*gridDim);
    //#ifdef _3D
    //    glm::ivec3 gridPos;
    //#else
    //    glm::ivec2 gridPos;
    //#endif
    //    gridPos.x = floor(gridDim.x * (worldPos.x - environmentMin.x) / (environmentMax.x - environmentMin.x));
    //    gridPos.y = floor(gridDim.y * (worldPos.y - environmentMin.y) / (environmentMax.y - environmentMin.y));
    //#ifdef _3D
    //    gridPos.z = floor(gridDim.z * (worldPos.z - environmentMin.z) / (environmentMax.z - environmentMin.z));
    //#endif
    //
    //    return gridPos;
#endif
}