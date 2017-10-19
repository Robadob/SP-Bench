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
    locationMessages->locationX[id] = d_environmentMin.x + ((-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.x);
    locationMessages->locationY[id] = d_environmentMin.y + ((-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.y);
#ifdef _3D
    locationMessages->locationZ[id] = d_environmentMin.z + ((-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.z);
#endif
}
__global__ void init_particles_uniform(LocationMessages *locationMessages, int particlesPerDim, DIMENSIONS_VEC offset) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_locationMessageCount)
        return;
    //Get index along in each dimension

#ifdef _3D
    int z = (id / (particlesPerDim * particlesPerDim));
    int y = (id % (particlesPerDim * particlesPerDim)) / particlesPerDim;
    int x = (id % (particlesPerDim * particlesPerDim)) % particlesPerDim;
#endif
#ifdef _2D
    int y = id / particlesPerDim;
    int x = id % particlesPerDim;
#endif
    //Distribute particles across that spread
    locationMessages->locationX[id] = d_environmentMin.x + ((x * offset.x) + offset.x*0.5f);
    locationMessages->locationY[id] = d_environmentMin.y + ((y * offset.y) +offset.y*0.5f);
#ifdef _3D
    locationMessages->locationZ[id] = d_environmentMin.z + ((z * offset.z) + offset.z*0.5f);
#endif
//Distributes particles to center of each bin
//    int hash = id % (glm::compMul(d_gridDim));
//	int div = id / (glm::compMul(d_gridDim));
//	int max = d_locationMessageCount / (glm::compMul(d_gridDim));
//	int z = (hash / (d_gridDim.y * d_gridDim.x));
//	int y = (hash % (d_gridDim.y * d_gridDim.x)) / d_gridDim.x;
//	int x = (hash % (d_gridDim.y * d_gridDim.x)) % d_gridDim.x;
//	//In a regular manner, scatter particles evenly between bins
//	locationMessages->locationX[id] = (x * (d_environmentMax.x / (float)d_gridDim.x)) + (d_environmentMax.x / (float)d_gridDim.x)*0.5;
//	locationMessages->locationY[id] = (y * (d_environmentMax.y / (float)d_gridDim.y)) + (d_environmentMax.y / (float)d_gridDim.y)*0.5;
//#ifdef _3D
//	locationMessages->locationZ[id] = (z * (d_environmentMax.z / (float)d_gridDim.z)) + (d_environmentMax.z / (float)d_gridDim.z)*0.5;
//#endif
}

__global__ void init_particles_clusters(curandState *state, LocationMessages *locationMessages, unsigned int startIndex, unsigned int clusterSize, DIMENSIONS_VEC clusterCenter, float clusterWidth)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= clusterSize)
        return;
    id += startIndex;
    if (id >= d_locationMessageCount)
        return;
    DIMENSIONS_VEC newPos;
    do
    {
        newPos.x = clusterCenter.x + ((curand_uniform(&state[id]) - 0.5f) * clusterWidth);
        newPos.y = clusterCenter.y + ((curand_uniform(&state[id]) - 0.5f) * clusterWidth);
#ifdef _3D
        newPos.z = clusterCenter.z + ((curand_uniform(&state[id]) - 0.5f) * clusterWidth);
#endif
    } while (2*distance(clusterCenter, newPos)>clusterWidth);
    locationMessages->locationX[id] = newPos.x;
    locationMessages->locationY[id] = newPos.y;
#ifdef _3D
    locationMessages->locationZ[id] = newPos.z;
#endif
}