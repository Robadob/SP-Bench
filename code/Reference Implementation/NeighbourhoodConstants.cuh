#ifndef __NeighbourhoodConstants_cuh__
#define __NeighbourhoodConstants_cuh__
#include "Neighbourhood.cuh"
//Constants included here are externed everywhere else, also useful for storing textures given their use of constants

__device__ __constant__ unsigned int d_locationMessageCount;
__device__ __constant__ unsigned int d_binCount;
__device__ __constant__ float d_interactionRad; //Not required by data structure, however provided so models remain consistent with data-structure.

__device__ __constant__ DIMENSIONS_IVEC d_gridDim;
__device__ __constant__ DIMENSIONS_VEC  d_gridDim_float;//Float clone, to save cast at runtime inside getGridPosition()
__device__ __constant__ DIMENSIONS_VEC  d_environmentMin;
__device__ __constant__ DIMENSIONS_VEC  d_environmentMax;

#ifdef _DEBUG
__device__ __constant__ unsigned int d_PBM_isBuilt;
#endif
#if defined(_GL) || defined(_DEBUG)
__device__ __constant__ LocationMessages *d_locationMessagesA;
__device__ __constant__ LocationMessages *d_locationMessagesB;
#endif
__device__ __constant__ cudaTextureObject_t d_tex_location[DIMENSIONS];
__device__ __constant__ cudaTextureObject_t d_tex_PBM;
//    __constant__ int d_tex_pbm_offset;


//    texture<int, 1, cudaReadModeElementType> tex_pbm;
#endif
