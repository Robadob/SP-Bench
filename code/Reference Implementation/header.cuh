//This file contains various static items that need to be available to multiple .cu files.
//Including them in shared headers causes multiple definition errors
//This would be much better if Lewin computers had Compute Capability 2.0
//Include me in every.cu file
#ifndef __header_cuh_
#define __header_cuh_
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#include "glm/glm.hpp"

//Cuda call
static void HandleCUDAError(const char *file,
    int line,
    cudaError_t status = cudaGetLastError()) {
    if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS)
    {
        if (status == cudaErrorUnknown)
        {
            printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
            printf("Perhaps performing the same operation under the CUDA debugger with Memory Checker enabled could help!\n");
            printf("If this error only occurs outside of NSight debugging sessions, or causes the system to lock up. It may be caused by not passing the required amount of shared memory to a kernal launch that uses runtime sized shared memory.\n", file, line);
            printf("Also possible you have forgotten to allocate texture memory you are trying to read\n");
            printf("Passing a buffer to 'cudaGraphicsSubResourceGetMappedArray' or a texture to 'cudaGraphicsResourceGetMappedPointer'.\n");
            getchar();
            exit(1);
        }
        printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
        getchar();
        exit(1);
    }
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))

//Set bits in a state from the mask
__device__ __host__ static int performSetBitwise(int state, int mask, int op) {
    //op: 0 - UnSet, 1 - Set
    int ret = -1;
    switch (op) {
    case 0:
        ret = ~((~state) | mask);
        break;
    case 1:
        ret = state | mask;
        break;
    }
    return ret;
}
#define PERFORM_SET_BITWISE(state, mask, op) (performSetBitwise(state,mask, op))
#define SELECT_MASK 1

#define DEVICE_RND \
__device__ static float rnd( curandState* globalState, int index ){ \
	curandState localState = globalState[index]; \
	float RANDOM = curand_uniform( &localState ); \
	globalState[index] = localState; \
	return RANDOM; \
}

#define THREADS_PER_TILE 128

#define KERNEL_PARAMS(count) \
const int threads_per_tile = THREADS_PER_TILE; \
int tile_size; \
dim3 grid; \
dim3 threads; \
tile_size = (int) ceil((float) count /threads_per_tile); \
grid = dim3(tile_size, 1, 1); \
threads = dim3(threads_per_tile, 1, 1);

#define RE_KERNEL_PARAMS(count) \
tile_size = (int) ceil((float) count /threads_per_tile); \
grid = dim3(tile_size, 1, 1); \
threads = dim3(threads_per_tile, 1, 1);

#define COLLISION_SHARE threads_per_tile*sizeof(LocationMessage)

//h(ost) because cuda headers already def float3 & int3 as structs
//typedef float h_float3[3];
//typedef float h_float4[4];
//typedef int h_int3[3];


#endif
