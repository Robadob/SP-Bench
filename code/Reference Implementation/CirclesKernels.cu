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
	DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]), toLoc, newLoc;
#else
	DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]), toLoc, newLoc;
#endif
	newLoc =  DIMENSIONS_VEC(0);//myLoc;//
	//Get first message
    float separation, k;
#ifdef _local
	LocationMessage lm2;
	LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc, &lm2);
#else
	LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
#endif
    //Always atleast 1 location message, our own location!
	const float rHalf = d_interactionRad/2.0f;
    do
	{
		assert(lm != 0);
        if ((lm->id != id))//CHANGED: Don't sort particles
        {
			toLoc = lm->location - myLoc;//Difference
			if (toLoc != DIMENSIONS_VEC(0))//Ignore distance 0
			{
				separation = length(toLoc);
				if (separation < d_interactionRad)
				{
					k = (separation < rHalf) ? d_repulse : d_attract;
					toLoc = (separation < rHalf) ? -toLoc : toLoc;
					toLoc /= separation;//Normalize (without recalculating seperation)
					separation = (separation < rHalf) ? separation : (d_interactionRad - separation);
					newLoc += k * separation * toLoc;
				}
            }
        }
		lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
	} while (lm);
    //Export newLoc
	newLoc += myLoc;
#ifdef _DEBUG
	assert(!isnan(newLoc.x));
	assert(!isnan(newLoc.y));
	assert(!isnan(newLoc.z));
	assert(!isnan(myLoc.x));
	assert(!isnan(myLoc.y));
	assert(!isnan(myLoc.z));
#endif
	newLoc = glm::clamp(newLoc, d_environmentMin, d_environmentMax);
	locationMessagesOut->locationX[id] = newLoc.x;
	locationMessagesOut->locationY[id] = newLoc.y;
#ifdef _3D
	locationMessagesOut->locationZ[id] = newLoc.z;
#endif
}