#include "NetworkKernels.cuh"

__global__ void
__launch_bounds__(64)
step_network_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //Get my local location
#ifdef AOS_MESSAGES
    const DIMENSIONS_VEC myLoc = locationMessagesIn->location[id];
#else
#ifdef _3D
    const DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]);
#else
    const DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]);
#endif
#endif
    DIMENSIONS_VEC toLoc, newLoc = DIMENSIONS_VEC(0);//myLoc;//
    //Get first message
    float separation, k;
//#ifdef _local
//    LocationMessage lm2;
//    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc, &lm2);
//#else
//    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
//#endif
    //Always atleast 1 location message, our own location!
    //const float rHalf = d_interactionRad / 2.0f;
    int ct = 0;
#if !defined(SHARED_BINSTATE)
    LocationMessage _lm;
    LocationMessage *lm = &_lm;
#if defined(MODULAR) || defined(MODULAR_STRIPS)
    locationMessagesIn->firstBin(myLoc, &_lm);
    do
    {
        while (locationMessagesIn->getNextNeighbour(&_lm))//Returns true if next neighbour was found
#else
    //Always atleast 1 location message, our own location!
    locationMessagesIn->getFirstNeighbour(myLoc, &_lm);
    do
#endif
#else
#if defined(MODULAR) || defined(MODULAR_STRIPS)
    LocationMessage *lm = locationMessagesIn->firstBin(myLoc);
do
{
    while (locationMessagesIn->getNextNeighbour(lm))//Returns a pointer to shared memory or 0
#else
    //Always atleast 1 location message, our own location!
    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
    do
#endif
#endif
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
                    k = sinf((separation / d_interactionRad)*3.141*-2)*d_repulse;
                    //k = (separation < rHalf) ? d_repulse : d_attract;
                    //toLoc = (separation < rHalf) ? -toLoc : toLoc;
                    toLoc /= separation;//Normalize (without recalculating seperation)
                    //separation = (separation < rHalf) ? separation : (d_interactionRad - separation);
                    newLoc += k * toLoc;
                    ct++;
                }
            }
        }
#if (!(defined(MODULAR) || defined(MODULAR_STRIPS))&&defined(SHARED_BINSTATE))
        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
#endif
    }
#if defined(MODULAR) || defined(MODULAR_STRIPS)
} while (locationMessagesIn->nextBin(lm));
#else
#if !defined(SHARED_BINSTATE)
    while (locationMessagesIn->getNextNeighbour(lm));
#else
    while (lm);
#endif
#endif
    //Export newLoc
    newLoc /= ct>0?ct:1;
    newLoc += myLoc;
#ifdef _DEBUG
    assert(!isnan(newLoc.x));
    assert(!isnan(newLoc.y));
#ifdef _3D
    assert(!isnan(newLoc.z));
#endif
    assert(!isnan(myLoc.x));
    assert(!isnan(myLoc.y));
#ifdef _3D
    assert(!isnan(myLoc.z));
#endif
#endif

#ifdef AOS_MESSAGES
    locationMessagesOut->location[id] = glm::clamp(newLoc, d_environmentMin, d_environmentMax);
#else
    newLoc = glm::clamp(newLoc, d_environmentMin, d_environmentMax);
    locationMessagesOut->locationX[id] = newLoc.x;
    locationMessagesOut->locationY[id] = newLoc.y;
#ifdef _3D
    locationMessagesOut->locationZ[id] = newLoc.z;
#endif
#endif
}


__global__ void init_network(curandState *state, LocationMessages *locationMessages, VertexData *v)
{
    //This is copied frm init_particles, but i dont think we actually plan to use the locations.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
    //negate and  + 1.0, to make  0<=x<1.0
#ifdef AOS_MESSAGES
    locationMessages->location[id].x = floor(d_environmentMin.x + ((-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.x));
    locationMessages->location[id].y = 0;
#ifdef _3D
    locationMessages->location[id].z = 0;
#endif
#else
    locationMessages->locationX[id] = floor(d_environmentMin.x + ((-curand_uniform(&state[id]) + 1.0f)*d_environmentMax.x));
    locationMessages->locationY[id] = 0;
#ifdef _3D
    locationMessages->locationZ[id] = 0;
#endif
#endif
    //Network specific stuff

}