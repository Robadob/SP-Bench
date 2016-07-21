#include "CirclesKernels.cuh"

__device__ __constant__ float d_attract;
__device__ __constant__ float d_repulse;

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
__global__ void step_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;

    //Get my local location
#ifdef _3D
    glm::vec3 myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]), locDiff, newLoc;
#else
	glm::vec2 myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]), locDiff, newLoc;
#endif
	newLoc = myLoc;
	//Get first message
    float dist, separation, k;
    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
    //Always atleast 1 location message, our own location!
    do
    {
        if ((lm->id != id))
        {
			locDiff = myLoc - lm->location;//Difference
            if (locDiff!=DIMENSIONS_VEC(0))//Ignore distance 0
			{
				dist = length(locDiff);//Distance (via pythagoras)
				separation = dist - d_interactionRad;
				if (separation < d_interactionRad)
				{

					k = (separation > 0.0f) ? d_attract : -d_repulse;
					newLoc += (k * separation * locDiff / d_interactionRad);
				}
            }
        }
        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
    } while (lm);
    //Export newLoc
	newLoc = glm::clamp(newLoc, d_environmentMin, d_environmentMax);
	locationMessagesOut->locationX[id] = newLoc.x;
	locationMessagesOut->locationY[id] = newLoc.y;
#ifdef _3D
	locationMessagesOut->locationZ[id] = newLoc.z;
#endif
}