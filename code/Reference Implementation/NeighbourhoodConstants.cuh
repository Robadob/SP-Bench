#include "Neighbourhood.cuh"
//Constants included here are externed everywhere else, also useful for storing textures given their use of constants

__device__ __constant__ unsigned int d_locationMessageCount;
#ifdef _3D
__device__ __constant__ glm::ivec3 d_gridDim;
__device__ __constant__ glm::vec3  d_environmentMin;
__device__ __constant__ glm::vec3  d_environmentMax;
#else
__device__ __constant__ glm::ivec2 d_gridDim;
__device__ __constant__ glm::vec2  d_environmentMin;
__device__ __constant__ glm::vec2  d_environmentMax;
#endif

//    __constant__ int d_tex_pbm_offset;


//    texture<int, 1, cudaReadModeElementType> tex_pbm;