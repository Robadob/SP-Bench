#ifndef __NeighbourhoodConstants_cuh__
#define __NeighbourhoodConstants_cuh__
#include "Neighbourhood.cuh"
//Constants included here are externed everywhere else, also useful for storing textures given their use of constants

__device__ __constant__ unsigned int d_locationMessageCount;
__device__ __constant__ float d_interactionRad; //Not required by data structure, however provided so models remain consistent with data-structure.
#ifdef _3D
__device__ __constant__ glm::ivec3 d_gridDim;
__device__ __constant__ glm::vec3  d_environmentMin;
__device__ __constant__ glm::vec3  d_environmentMax;
#else
__device__ __constant__ glm::ivec2 d_gridDim;
__device__ __constant__ glm::vec2  d_environmentMin;
__device__ __constant__ glm::vec2  d_environmentMax;
#endif

#ifdef _DEBUG
__device__ __constant__ unsigned int d_PBM_isBuilt;
#endif

__device__ __constant__ cudaTextureObject_t d_tex_locationX;
__device__ __constant__ cudaTextureObject_t d_tex_locationY;
#ifdef _3D
__device__ __constant__ cudaTextureObject_t d_tex_locationZ;
#endif
__device__ __constant__ cudaTextureObject_t d_tex_PBM;
//    __constant__ int d_tex_pbm_offset;


//    texture<int, 1, cudaReadModeElementType> tex_pbm;
#endif