#include "NullKernels.cuh"

__global__ void step_null_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut, DIMENSIONS_VEC *result)
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
    DIMENSIONS_VEC averageLoc = DIMENSIONS_VEC(0);
    int ct = 0;
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
    {
        assert(lm != 0);
        if ((lm->id != id))//CHANGED: Don't sort particles
        {
            //Sum neighbours
            averageLoc += lm->location;
            //averageLoc += glm::vec2(lm->location.x, lm->location.y);
            //averageLoc += glm::vec2(sqrt(pow(sqrt(pow(lm->location.x, 2.0f)), 2.0f)), sqrt(pow(sqrt(pow(lm->location.y, 2.0f)), 2.0f)));
            ct++;
        }
#if !(defined(MODULAR) || defined(MODULAR_STRIPS))
        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
#endif
    }
#if defined(MODULAR) || defined(MODULAR_STRIPS)
} while (locationMessagesIn->nextBin(lm));
#else
    while (lm);
#endif

    //Export newLoc
    if (ct)
    {
        averageLoc /= ct;
    }
#ifdef _DEBUG
    //if (isnan(averageLoc.x) || isnan(averageLoc.y))
    //{
    //    lm = locationMessagesIn->getFirstNeighbour(myLoc);do
    //    {
    //        assert(lm != 0);
    //        if ((lm->id != id))//CHANGED: Don't sort particles
    //        {
    //            //Sum neighbours
    //            averageLoc += lm->location;
    //            ct++;
    //        }
    //        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
    //} while (lm);
    //}
    assert(!isnan(averageLoc.x));
    assert(!isnan(averageLoc.y));
#ifdef _3D
    assert(!isnan(averageLoc.z));
#endif
    assert(!isnan(myLoc.x));
    assert(!isnan(myLoc.y));
#ifdef _3D
    assert(!isnan(myLoc.z));
#endif
#endif
    result[id] = averageLoc;
}