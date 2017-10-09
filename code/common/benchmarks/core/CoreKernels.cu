#include "CoreKernels.cuh"

__global__ void init_curand(curandState *state, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < d_locationMessageCount)
        curand_init(seed, id, 0, &state[id]);
}
__global__ void init_particles(curandState *state, LocationMessages *locationMessages) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
    //negate and  + 1.0, to make  0<=x<1.0
    locationMessages->locationX[id] = (-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.x;
    locationMessages->locationY[id] = (-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.y;
#ifdef _3D
    locationMessages->locationZ[id] = (-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.z;
#endif
}
__global__ void init_particles_uniform(LocationMessages *locationMessages) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_locationMessageCount)
		return;
	int hash = id % (glm::compMul(d_gridDim));
	int div = id / (glm::compMul(d_gridDim));
	int max = d_locationMessageCount / (glm::compMul(d_gridDim));
	int z = (hash / (d_gridDim.y * d_gridDim.x));
	int y = (hash % (d_gridDim.y * d_gridDim.x)) / d_gridDim.x;
	int x = (hash % (d_gridDim.y * d_gridDim.x)) % d_gridDim.x;
	//In a regular manner, scatter particles evenly between bins
	locationMessages->locationX[id] = (x * (d_environmentMax.x / (float)d_gridDim.x)) + (d_environmentMax.x / (float)d_gridDim.x)*0.5;
	locationMessages->locationY[id] = (y * (d_environmentMax.y / (float)d_gridDim.y)) + (d_environmentMax.y / (float)d_gridDim.y)*0.5;
#ifdef _3D
	locationMessages->locationZ[id] = (z * (d_environmentMax.z / (float)d_gridDim.z)) + (d_environmentMax.z / (float)d_gridDim.z)*0.5;
#endif
}