#include "NullKernels.cuh"

__global__ void step_null_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut, DIMENSIONS_VEC *result)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;

    //Get my local location
#ifdef _3D
    DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]), averageLoc;
#else
    DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]),  averageLoc;
#endif
    averageLoc = DIMENSIONS_VEC(0);
    LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
    //Always atleast 1 location message, our own location!
    int ct = 0;
    do
    {
        assert(lm != 0);
        if ((lm->id != id))//CHANGED: Don't sort particles
        {
            //Sum neighbours
            averageLoc += lm->location;
            ct++;
        }
        lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
    } while (lm);

    //Export newLoc
    averageLoc /= ct;
#ifdef _DEBUG
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