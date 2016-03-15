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

__global__ void step_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;

    //Get my local location
#ifdef _3D
    glm::vec3 myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]), theirLoc, locDiff;
#else
    glm::vec2 myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]), theirLoc, locDiff;
#endif
    //Get first message
    float dist, separation, k;
    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
    //Always atleast 1 location message, our own location!
    int counter = 0;
    do
    {
        counter++;
        if ((lm->id != id))
        {
            locDiff = myLoc - lm->location;//Difference
            if (locDiff==DIMENSIONS_VEC(0))//Ignore distance 0
            {
                lm = locationMessagesIn->getNextNeighbour(lm);
                continue;
            }
            theirLoc = locDiff*locDiff;//Squared
            dist = sqrt(glm::compAdd(theirLoc));//Distance (via pythagoras)
            separation = dist - d_interactionRad - d_interactionRad;
            if (separation < d_interactionRad)
            {
                if (separation > 0.0f)
                    k = d_attract;
                else
                    k = d_repulse;
                myLoc += (k*separation*(locDiff / dist));
            }
        }
        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
    } while (lm);
    //Export myloc?
    locationMessagesOut->locationX[id] = myLoc.x;
    locationMessagesOut->locationY[id] = myLoc.y;
#ifdef _3D
    locationMessagesOut->locationZ[id] = myLoc.z;
#endif
#if defined(_GL) || defined(_DEBUG)
    locationMessagesOut->count[id] = counter/(float)d_locationMessageCount;
  //  printf("%.3f\n", locationMessagesOut->count[id]);
#endif
}
