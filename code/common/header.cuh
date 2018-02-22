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
#ifdef _DEBUG
	cudaDeviceSynchronize();
#endif
    if (status != cudaError::cudaSuccess || (status = cudaGetLastError()) != cudaError::cudaSuccess)
    {
        if (status == cudaErrorUnknown)
        {
            printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
            printf("Perhaps performing the same operation under the CUDA debugger with Memory Checker enabled could help!\n");
            printf("If this error only occurs outside of NSight debugging sessions, or causes the system to lock up. It may be caused by not passing the required amount of shared memory to a kernal launch that uses runtime sized shared memory.\n");
            printf("Also possible you have forgotten to allocate texture memory you are trying to read\n");
            printf("Passing a buffer to 'cudaGraphicsSubResourceGetMappedArray' or a texture to 'cudaGraphicsResourceGetMappedPointer'.\n");
            getchar();
            exit(1);
        }
        printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
#ifdef _DEBUG
        getchar();
#endif
        exit(1);
    }
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))


#endif
