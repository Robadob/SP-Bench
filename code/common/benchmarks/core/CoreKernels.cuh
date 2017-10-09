#ifndef __CoreKernels_cuh__
#define __CoreKernels_cuh__

#include "near_neighbours/Neighbourhood.cuh"

//#pragma warning(disable:4244)
//#pragma warning(disable:4305)
#include <curand_kernel.h>
//#pragma warning (default : 4244)
//#pragma warning (default : 4305)

__global__ void init_curand(curandState *state, unsigned long long seed = 12);
__global__ void init_particles(curandState *state, LocationMessages *locationMessages);
__global__ void init_particles_uniform(LocationMessages *locationMessages);

#endif //__CoreKernels_cuh__