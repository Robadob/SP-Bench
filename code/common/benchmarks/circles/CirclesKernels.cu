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
__global__ void step_circles_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
#if defined(THREADBLOCK_BINS)
{
    __shared__ *void shared_data;

    const int id = threadIdx.x;
    DIMENSIONS_IVEC binPos =
#ifdef _3D
        DIMENSIONS_IVEC(blockIdx.x, blockIdx.y, blockIdx.z);
#else
        DIMENSIONS_IVEC(blockIdx.x, blockIdx.y);
#endif
    unsigned int binId = locationMessagesIn->getHash(binPos);
    unsigned int binSize = d_pbm[binId]-d_pbm[binId];

    //If bin has messages
    if(binSize)
    {
        unsigned int *msgCount = (unsigned int*)shared_data;
#ifdef _3D
        unsigned int *msgIndex = msgCount + 27;
        LocationMessage *messageData = msgIndex + 28;
#else
        unsigned int *msgIndex = msgCount + 9;
        LocationMessage *messageData = msgIndex + 10;
#endif
        //Calculate how many messages we have
        if(threadIdx.x<pow(3,DIMENSIONS))
        {//First 9(27) threads copy relevant moore neighbourhood size into shared
            msgCount[threadIdx.x]=0;
            //Calculate relative offset based of 3x3(x3)
            DIMENSIONS_IVEC relative;
#ifdef _3D
            relative.z = (hash / 9) - 1;
            relative.y = ((hash % 9) / 3) - 1;
            relative.x = ((hash % 9) % 3) - 1;
#else
            relative.y = (hash / 3) - 1;
            relative.x = (hash % 3) - 1;
#endif
            //Check this is in bounds
            relative+=binPos
            if (!(
#ifdef _3D
                relative.z<0 || relative.z >= d_gridDim.z ||
#endif
                relative.y<0 || relative.y >= d_gridDim.y ||
                relative.x<0 || relative.x >= d_gridDim.x
                ))
            {
                unsigned int relativeId = locationMessagesIn->getHash(binPos);
                msgCount[threadIdx.x] = d_pbm[relativeId]-d_pbm[relativeId];
            }
        }
        _syncthreads();
        //Scan to calculate where threads store their data and total count
        if (threadIdx.x==0)
        {
            unsigned int sum = 0;
            //Prefix sum
            unsigned int i= 0;
#ifdef _3D
            for (;i < 27;++i) 
#else
            for (;i < 9; ++i)
#endif
            {
                msgIndex[i] = sum;
                sum += msgCount[i];
            }
            msgIndex[i]=sum;
        }
        _syncthreads();
        //while has messages
        unsigned int msgCtr = 0;
#ifdef _3D
        while (msgCtr<msgIndex[27])
#else
        while (msgCtr<msgIndex[9])
#endif
        {
            //Load messages into shared
#ifdef _3D
            if(threadIdx.x<27)
#else
            if(threadIdx.x<9)
#endif
            {
                //Calc bin again
                DIMENSIONS_IVEC relative;
#ifdef _3D
                relative.z = (hash / 9) - 1;
                relative.y = ((hash % 9) / 3) - 1;
                relative.x = ((hash % 9) % 3) - 1;
#else
                relative.y = (hash / 3) - 1;
                relative.x = (hash % 3) - 1;
#endif
                //Check this is in bounds
                relative+=binPos
                if (!(
#ifdef _3D
                    relative.z<0 || relative.z >= d_gridDim.z ||
#endif
                    relative.y<0 || relative.y >= d_gridDim.y ||
                    relative.x<0 || relative.x >= d_gridDim.x
                    ))
                {
                    unsigned int relativeId = locationMessagesIn->getHash(binPos);
                    unsigned int pbmStart = d_pbm[relativeId];
                    unsigned int sharedStart = msgIndex[threadIdx.x];
                    //How many messages can we load?
                    ///Currently we assume all messages fit
                    for (unsigned int i = 0; i<msgCount[threadIdx.x]; ++i)
                    {
#ifdef AOS_MESSAGES
                        messageData[sharedStart + i] = d_messages->location[pbmStart + i];
#else
                        messageData[sharedStart + i].location.x = d_messages->locationX[pbmStart + i];
                        messageData[sharedStart + i].location.y = d_messages->locationY[pbmStart + i];
#ifdef _3D
                        messageData[sharedStart + i].location.z = d_messages->locationZ[pbmStart + i];
#endif
                    }
                }
                //How far into total messages are we?
            }
            _syncthreads();
            //Iterate messages
            for (; msgCtr < ?; ++msgCtr)
            {

                //Do Model!
            }
        }
    }
}
#else
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
#endif