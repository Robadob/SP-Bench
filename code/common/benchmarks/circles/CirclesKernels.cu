#include "CirclesKernels.cuh"

__device__ __constant__ float d_attract;
__device__ __constant__ float d_repulse;

//Padabs model
/*
__global__ void step_circles_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
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
*/
//Improved Sin Model
//__launch_bounds__(64, 32)
__global__ void
#if defined(MODULAR)||defined(MODULAR_STRIPS)
__launch_bounds__(64)
#endif
step_circles_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
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