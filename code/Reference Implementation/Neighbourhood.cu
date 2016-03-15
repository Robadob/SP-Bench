#include "Neighbourhood.cuh"
#include "NeighbourhoodConstants.cuh"
#include "NeighbourhoodKernels.cuh"
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
    : environmentMin(environmentMin)
    , environmentMax(environmentMax)
    , maxAgents(maxAgents)
    , interactionRad(interactionRad)
    , locationMessageCount(0)
    , gridDim((environmentMax - environmentMin) / interactionRad)
#ifdef _DEBUG
    , PBM_isBuilt(0)
#endif
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
    //Allocate tex
    deviceAllocateTextures();
    //Set device constants
    CUDA_CALL(cudaMemcpyToSymbol(d_interactionRad, &interactionRad, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &gridDim, sizeof(DIMENSIONS_IVEC)));
    DIMENSIONS_VEC t_gridDim = (DIMENSIONS_VEC)gridDim;
    CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &t_gridDim, sizeof(DIMENSIONS_VEC)));

    CUDA_CALL(cudaMemcpyToSymbol(d_environmentMin, &environmentMin, sizeof(DIMENSIONS_VEC)));
    CUDA_CALL(cudaMemcpyToSymbol(d_environmentMax, &environmentMax, sizeof(DIMENSIONS_VEC)));

#ifdef _DEBUG
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif
    setLocationCount(locationMessageCount);
    unsigned int t_binCount = getBinCount();
    CUDA_CALL(cudaMemcpyToSymbol(d_binCount, &t_binCount, sizeof(unsigned int)));
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
    //Deallocate tex
    deviceDeallocateTextures();
}
#ifdef _DEBUG

DIMENSIONS_IVEC SpatialPartition::getGridPosition(DIMENSIONS_VEC worldPos)
{
#ifndef SP_NO_CLAMP_GRID
    //Clamp each grid coord to 0<=x<dim
    return clamp(floor(((worldPos - environmentMin) / (environmentMax - environmentMin))*glm::vec3(gridDim)), glm::vec3(0), glm::vec3(gridDim)-glm::vec3(1));
#else
    return floor(((worldPos - environmentMin) / (environmentMax - environmentMin))*glm::vec3(gridDim));
#endif
}

int SpatialPartition::getHash(DIMENSIONS_IVEC gridPos)
{
    gridPos = clamp(gridPos, DIMENSIONS_IVEC(0), gridDim - DIMENSIONS_IVEC(1));
    return
#ifdef _3D
        (gridPos.z * gridDim.y * gridDim.x) +   //z
#endif
        (gridPos.y * gridDim.x) +					//y
        gridPos.x; 	                                //x
}
DIMENSIONS_IVEC SpatialPartition::getPos(unsigned int hash)
{
    if (hash >= getBinCount())
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
void SpatialPartition::assertSearch()
{
    unsigned int outCount = getBinCount() + 1;
    unsigned int tableSize = ((outCount / 10) + 1) * 10;

    //Copy raw PBM from device to host
    unsigned int *PBM_raw = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*tableSize));
    memset(PBM_raw, 0, tableSize * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(PBM_raw, d_PBM, sizeof(unsigned int)*outCount, cudaMemcpyDeviceToHost));

    //Calculate the size of every bin
    unsigned int *PBM_binSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*tableSize));
    for (unsigned int i = 0; i < tableSize; i++)
    {
        if (i < outCount - 1)
            PBM_binSize[i] = PBM_raw[i + 1] - PBM_raw[i];
        else
        {
            PBM_binSize[i] = 11111;
        }

    }

    //Calculate the size of each bin's neighbourhood
    unsigned int *PBM_neighbourhoodSize = static_cast<unsigned int *>(malloc(sizeof(unsigned int)*tableSize));
    for (unsigned int i = 0; i < ((outCount / 10) + 1) * 10; i++)
    {
        PBM_neighbourhoodSize[i] = 0;
        if (i < outCount - 1)
        {
            DIMENSIONS_IVEC curCell = getPos(i);
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    for (int z = -1; z <= 1; z++)
                    {
                DIMENSIONS_IVEC neighbourCell = curCell + DIMENSIONS_IVEC(x, y, z);
                if (isValid(neighbourCell))
                {
                    PBM_neighbourhoodSize[i] += PBM_binSize[getHash(neighbourCell)];
                }
                    }
        }

    }

    //Copy every location and neighbour count from device to host
    float *d_bufferPtr;
    LocationMessages lm;
    lm.locationX = (float*)malloc(sizeof(float)*locationMessageCount);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(lm.locationX, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
    lm.locationY = (float*)malloc(sizeof(float)*locationMessageCount);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(lm.locationY, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
#ifdef _3D
    lm.locationZ = (float*)malloc(sizeof(float)*locationMessageCount);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(lm.locationZ, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
#endif
    lm.count = (float*)malloc(sizeof(float)*locationMessageCount);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages_swap->count, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(lm.count, d_bufferPtr, sizeof(float)*locationMessageCount, cudaMemcpyDeviceToHost));
    //ASSERT: Every agent searched the right amount of neighbours
    unsigned int matchFails = 0;
    for (unsigned int i = 0; i < locationMessageCount; i++)
    {
        //For rendering purposes the count is stored as count/totalMessages, invert this math for assertion
        unsigned int hash = getHash(getGridPosition(glm::vec3(lm.locationX[i], lm.locationY[i], lm.locationZ[i])));
        if (glm::epsilonNotEqual(lm.count[i], PBM_neighbourhoodSize[hash] / (float)locationMessageCount, 0.5f))
        {
            //printf("%u=%u-%f=%f,", (unsigned int)(lm.count[i] * locationMessageCount), PBM_neighbourhoodSize[hash], lm.count[i], PBM_neighbourhoodSize[hash] / (float)locationMessageCount);
            matchFails++;
        }
    }
    //Free location/count data
    free(lm.locationX);
    free(lm.locationY);
    free(lm.locationZ);
    free(lm.count);
    if (matchFails>0)
    {
        printf("ERROR: Neighbour search totals do not match (%u/%u)\n", matchFails, locationMessageCount);
    }
    else
    {
        free(PBM_raw);
        free(PBM_binSize);
        free(PBM_neighbourhoodSize);
        return;
    }
    //Output the 3 PBM_ data structures to file in a readable format
    FILE *file = fopen("../logs/PBM.txt", "w");
    fprintf(file, "ERROR: Neighbour search totals do not match (%u/%u)\n", matchFails, locationMessageCount);
    fprintf(file, "Raw PBM\n");
    fprintf(file, "|%5s¦%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    fprintf(file, "|-----¦-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
    for (unsigned int i = 0; i < (outCount / 10)-1; i++)
    {
        fprintf(file, "|%4u0¦%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i, 
            PBM_raw[((outCount / 10)*i) + 0],
            PBM_raw[((outCount / 10)*i) + 1],
            PBM_raw[((outCount / 10)*i) + 2],
            PBM_raw[((outCount / 10)*i) + 3],
            PBM_raw[((outCount / 10)*i) + 4],
            PBM_raw[((outCount / 10)*i) + 5],
            PBM_raw[((outCount / 10)*i) + 6],
            PBM_raw[((outCount / 10)*i) + 7],
            PBM_raw[((outCount / 10)*i) + 8],
            PBM_raw[((outCount / 10)*i) + 9]
            );
    }
    fprintf(file, "Bin Size\n");
    fprintf(file, "|%5s¦%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    fprintf(file, "|-----¦-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
    for (unsigned int i = 0; i < (outCount / 10) - 1; i++)
    {
        fprintf(file, "|%4u0¦%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i,
            PBM_binSize[((outCount / 10)*i) + 0],
            PBM_binSize[((outCount / 10)*i) + 1],
            PBM_binSize[((outCount / 10)*i) + 2],
            PBM_binSize[((outCount / 10)*i) + 3],
            PBM_binSize[((outCount / 10)*i) + 4],
            PBM_binSize[((outCount / 10)*i) + 5],
            PBM_binSize[((outCount / 10)*i) + 6],
            PBM_binSize[((outCount / 10)*i) + 7],
            PBM_binSize[((outCount / 10)*i) + 8],
            PBM_binSize[((outCount / 10)*i) + 9]
            );
    }
    fprintf(file, "Neighbour Count\n");
    fprintf(file, "|%5s¦%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|%5s|\n", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    fprintf(file, "|-----¦-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|\n");
    for (unsigned int i = 0; i < (outCount / 10) - 1; i++)
    {
        fprintf(file, "|%4u0¦%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|%5u|\n", i,
            PBM_neighbourhoodSize[((outCount / 10)*i) + 0],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 1],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 2],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 3],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 4],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 5],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 6],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 7],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 8],
            PBM_neighbourhoodSize[((outCount / 10)*i) + 9]
            );
    }
    //Cleanup resources
    fclose(file);
    free(PBM_raw);
    free(PBM_binSize);
    free(PBM_neighbourhoodSize);
}
#endif
void SpatialPartition::deviceAllocateLocationMessages(LocationMessages **d_locMessage)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_locMessage, sizeof(LocationMessages)));
    float *d_loc_temp;
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationX), &d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationY), &d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
#ifdef _3D
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->locationZ), &d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
#endif
#if defined(_GL) || defined(_DEBUG)
    CUDA_CALL(cudaMalloc(&d_loc_temp, sizeof(float)*maxAgents));
    CUDA_CALL(cudaMemcpy(&((*d_locMessage)->count), &d_loc_temp, sizeof(float*), cudaMemcpyHostToDevice));
#endif
}
void SpatialPartition::deviceAllocatePBM(unsigned int **d_PBM_t)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_PBM_t, sizeof(unsigned int)*(binCount+1)));
}
void SpatialPartition::deviceAllocatePrimitives(unsigned int **d_keys, unsigned int **d_vals)
{
    unsigned int binCount = getBinCount();
    CUDA_CALL(cudaMalloc(d_keys, sizeof(unsigned int)*maxAgents));
    CUDA_CALL(cudaMalloc(d_vals, sizeof(unsigned int)*maxAgents));
}
void SpatialPartition::deviceAllocateTextures()
{
    //Locations
#ifdef _GL
#pragma unroll 3
    for (unsigned int i = 0; i < DIMENSIONS;i++)
        deviceAllocateGLTexture_float(i);
    deviceAllocateGLTexture_float2();//Allocate a texture to store counting info in (Used to colour the visualisation
#else
#pragma unroll 3
    for (unsigned int i = 0; i < DIMENSIONS;i++)
        deviceAllocateTexture_float(i);
#endif
    //PBM
    deviceAllocateTexture_int();
}
void SpatialPartition::fillTextures()
{
    float *d_bufferPtr;
    //Potentially refactor so we store/swap these pointers on host in syncrhonisation
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[0], d_bufferPtr, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[1], d_bufferPtr, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#ifdef _3D
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tex_loc_ptr[2], d_bufferPtr, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#endif
#ifdef _GL
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_locationMessages->count, sizeof(float*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tex_location_ptr_count, d_bufferPtr, locationMessageCount*sizeof(float), cudaMemcpyDeviceToDevice));
#endif

    CUDA_CALL(cudaMemcpy(tex_PBM_ptr, d_PBM, (getBinCount()+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

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
/*
Allocates the PBM texture, which is only accessed via CUDA & memcpy
*/
void SpatialPartition::deviceAllocateTexture_int()
{
    //Define cuda array format
    //Allocate cuda array
    unsigned int size = getBinCount() + 1;
    CUDA_CALL(cudaMalloc(&tex_PBM_ptr, size*sizeof(unsigned int)));//Number of elements, not bytes
    //Define cuda resource from array
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = tex_PBM_ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = size*sizeof(unsigned int);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;

    CUDA_CALL(cudaCreateTextureObject(&tex_PBM, &resDesc, &texDesc, NULL));
    CUDA_CALL(cudaMemcpyToSymbol(d_tex_PBM, &tex_PBM, sizeof(cudaTextureObject_t)));
}
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

#pragma unroll
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
    cudaDestroyTextureObject(tex_PBM);
    cudaFree(tex_PBM_ptr);
#ifdef _GL
    cudaGraphicsUnregisterResource(gl_gRes_count);
    GL_CALL(glDeleteBuffers(1, &gl_tbo_count));
    GL_CALL(glDeleteTextures(1, &gl_tex_count));
#endif
}

unsigned int SpatialPartition::getBinCount()
{
    return (unsigned int)glm::compMul(gridDim);
}
void SpatialPartition::setLocationCount(unsigned int t_locationMessageCount)
{
    //Set local copy
    locationMessageCount = t_locationMessageCount;
    //Set device constants
    CUDA_CALL(cudaMemcpyToSymbol(d_locationMessageCount, &locationMessageCount, sizeof(unsigned int)));
}

void SpatialPartition::launchHashLocationMessages()
{
    int blockSize;   // The launch configurator returned block size 
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, hashLocationMessages, 32, 0));//Randomly 32
    // Round up according to array size
    int gridSize = (locationMessageCount + blockSize - 1) / blockSize;
    hashLocationMessages <<<gridSize, blockSize>>>(d_keys, d_vals, d_locationMessages);
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
    reorderLocationMessages <<<gridSize, blockSize, requiredSM_reorderLocationMessages(blockSize) >>>(d_keys, d_vals, d_PBM, d_locationMessages, d_locationMessages_swap);
    CUDA_CALL(cudaDeviceSynchronize());//unncecssary sync
    swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}
void SpatialPartition::swap()
{
    //Switch d_locationMessages and d_locationMessages_swap
    LocationMessages* d_locationmessages_temp = d_locationMessages;
    d_locationMessages = d_locationMessages_swap;
    d_locationMessages_swap = d_locationmessages_temp;

#ifdef _DEBUG
    PBM_isBuilt = 0;
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif
}
void SpatialPartition::buildPBM()
{
    //If no messages, or instances, don't bother
    if (locationMessageCount<1) return;
#if _DEBUG
    assertSearch();
#endif
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
    CUDA_CALL(cudaMemset(d_PBM, 0x00000000, (binCount + 1) * sizeof(unsigned int)));
    launchReorderLocationMessages();

    //Clone data to textures ready for neighbourhood search
    fillTextures();
#ifdef _DEBUG
    PBM_isBuilt = 1;
    CUDA_CALL(cudaMemcpyToSymbol(d_PBM_isBuilt, &PBM_isBuilt, sizeof(unsigned int)));
#endif
}
