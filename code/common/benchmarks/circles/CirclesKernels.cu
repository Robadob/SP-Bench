#include "CirclesKernels.cuh"
/* Not required by RDC=false
__device__ __constant__ float d_attract;
__device__ __constant__ float d_repulse;
*/
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

__device__ __forceinline__ DIMENSIONS_VEC reduceCircles(DIMENSIONS_VEC me, DIMENSIONS_VEC them, unsigned int &ct)
{
    DIMENSIONS_VEC toLoc = them - me;//Difference
    if (toLoc != DIMENSIONS_VEC(0))//Ignore distance 0
    {
        float separation = glm::length(toLoc);
        if (separation < d_interactionRad)
        {
            float k = sinf((separation / d_interactionRad)*3.141*-2)*d_repulse;
            //k = (separation < rHalf) ? d_repulse : d_attract;
            //toLoc = (separation < rHalf) ? -toLoc : toLoc;
            toLoc /= separation;//Normalize (without recalculating seperation)
            //separation = (separation < rHalf) ? separation : (d_interactionRad - separation);
            ct++;
            return k * toLoc;
        }
    }
    return DIMENSIONS_VEC(0);
}

__global__ void
#if defined(MODULAR)||defined(MODULAR_STRIPS)
__launch_bounds__(64)
#endif
step_circles_model2(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;

#ifdef _3D
    const DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]);
#else
    const DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id]);
#endif
    DIMENSIONS_IVEC myGrid = getGridPosition(myLoc);
    DIMENSIONS_VEC newLoc = DIMENSIONS_VEC(0);
    unsigned int ct = 0;
#if defined(MODULAR)
    DIMENSIONS_IVEC offset(
        (myGrid.x + 1) % 3
        ,(myGrid.y + 1) % 3
#if defined(_3D)
        ,(myGrid.z + 1) % 3
#endif
        );
#elif defined(MODULAR_STRIPS)
    DIMENSIONS_IVEC_MINUS1 offset(
        (myGrid.y + 1) % 3
#if defined(_3D)
        ,(myGrid.z + 1) % 3
#endif
        );
#endif
#pragma unroll
    for (int x = -1; x <= 1; ++x)
#if !((defined(STRIPS)||defined(MODULAR_STRIPS))&&defined(_2D))
#pragma unroll
    for (int y = -1; y <= 1; ++y)
#if !(defined(STRIPS)||defined(MODULAR_STRIPS)||defined(_2D))
#pragma unroll
    for (int z = -1; z <= 1; ++z)    
#endif
#endif
#if defined(STRIPS)
    {
#if defined(_3D)
        DIMENSIONS_IVEC binFirst = myGrid + DIMENSIONS_IVEC(-1, x, y);
#elif defined(_2D)
        DIMENSIONS_IVEC binFirst = myGrid + DIMENSIONS_IVEC(-1, x);
#else
#error "Unexpected dims"
#endif
        DIMENSIONS_IVEC binLast = binFirst;
        binLast.x += 2;
        bool firstInvalid = invalidBinX(binFirst);
        bool lastInvalid = invalidBinX(binLast);
#if defined(_3D)
        if (!invalidBinYZ(binFirst))
#elif defined(_2D)
        if (!invalidBinY(binFirst))
#else
#error "Unexpected dims"
#endif
        {
            if (firstInvalid)
            {
                binFirst.x = 0;
            }
            if (lastInvalid)
            {//If strip ends out of bounds only
                binLast.x = d_gridDim.x - 1;//Max x coord
            }
            unsigned int binFirstHash = getHash(binFirst);
            unsigned int binLastHash = binFirstHash + (binLast.x - binFirst.x);//Strips at most len 3
#if defined(GLOBAL_PBM)
            unsigned int binIndex = d_pbm[binFirstHash];
            unsigned int binIndexMax = d_pbm[binLastHash + 1];
#elif defined(LDG_PBM)
            unsigned int binIndex = __ldg(&d_pbm[binFirstHash]);
            unsigned int binIndexMax = __ldg(&d_pbm[binLastHash + 1]);
#else
            unsigned int binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, binFirstHash);
            unsigned int binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, binLastHash + 1);
#endif
            /**
            * Start Same Block
            */
            if (binIndex < binIndexMax)
            {
                for (unsigned int i = binIndex; i < binIndexMax; ++i)
                {
                    if (i != id)
                    {
                        DIMENSIONS_VEC neighbourPos;
#if defined(GLOBAL_MESSAGES)
#ifdef AOS_MESSAGES
                        neighbourPos = d_messages->location[i];
#else
                        neighbourPos.x = d_messages->locationX[i];
                        neighbourPos.y = d_messages->locationY[i];
#ifdef _3D
                        neighbourPos.z = d_messages->locationZ[i];
#endif
#endif
#elif defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
                        //sm_message->location = __ldg(&d_messages->location[i]);
                        neighbourPos.x = __ldg(&d_messages->location[i].x);
                        neighbourPos.y = __ldg(&d_messages->location[i].y);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->location[i].z);
#endif
#else
                        neighbourPos.x = __ldg(&d_messages->locationX[i]);
                        neighbourPos.y = __ldg(&d_messages->locationY[i]);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->locationZ[i]);
#endif
#endif
#else//Read message data from tex cache (default)
#ifdef AOS_MESSAGES
#error TODO Read AOS Messages from explicit texture
#else
                        neighbourPos.x = tex1Dfetch<float>(d_tex_location[0], i);
                        neighbourPos.y = tex1Dfetch<float>(d_tex_location[1], i);
#ifdef _3D
                        neighbourPos.z = tex1Dfetch<float>(d_tex_location[2], i);
#endif
#endif
#endif
                        newLoc += reduceCircles(myLoc, neighbourPos, ct);
                    }
                }
            }
            /**
            * End Same block
            */
        }
    }
#elif defined(MODULAR)
    {
#if defined(_3D)
        DIMENSIONS_IVEC binPos = glm::ivec3(x + offset.x, y + offset.y, y + offset.z);
#elif defined(_2D)
        DIMENSIONS_IVEC binPos = glm::ivec2(x + offset.x, y + offset.y);
#endif
        //For each axis, if new relative > 1, set -1
        binPos.x = binPos.x>1 ? binPos.x - 3 : binPos.x;
        binPos.y = binPos.y>1 ? binPos.y - 3 : binPos.y;
#ifdef _3D
        binPos.z = binPos.z>1 ? binPos.z - 3 : binPos.z;
#endif
        binPos += myGrid;
        //Get bin  start/end index
        unsigned int binHash = getHash(binPos);
#if defined(GLOBAL_PBM)
        unsigned int binIndex = d_pbm[binHash];
        unsigned int binIndexMax = d_pbm[binHash + 1];
#elif defined(LDG_PBM)
        unsigned int binIndex = __ldg(&d_pbm[binHash]);
        unsigned int binIndexMax = __ldg(&d_pbm[binHash + 1]);
#else
        unsigned int binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, binHash);
        unsigned int binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, binHash + 1);
#endif
            /**
            * Start Same Block
            */
            if (binIndex < binIndexMax)
            {
                for (unsigned int i = binIndex; i < binIndexMax; ++i)
                {
                    if (i != id)
                    {
                        DIMENSIONS_VEC neighbourPos;
#if defined(GLOBAL_MESSAGES)
#ifdef AOS_MESSAGES
                        neighbourPos = d_messages->location[i];
#else
                        neighbourPos.x = d_messages->locationX[i];
                        neighbourPos.y = d_messages->locationY[i];
#ifdef _3D
                        neighbourPos.z = d_messages->locationZ[i];
#endif
#endif
#elif defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
                        //sm_message->location = __ldg(&d_messages->location[i]);
                        neighbourPos.x = __ldg(&d_messages->location[i].x);
                        neighbourPos.y = __ldg(&d_messages->location[i].y);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->location[i].z);
#endif
#else
                        neighbourPos.x = __ldg(&d_messages->locationX[i]);
                        neighbourPos.y = __ldg(&d_messages->locationY[i]);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->locationZ[i]);
#endif
#endif
#else//Read message data from tex cache (default)
#ifdef AOS_MESSAGES
#error TODO Read AOS Messages from explicit texture
#else
                        neighbourPos.x = tex1Dfetch<float>(d_tex_location[0], i);
                        neighbourPos.y = tex1Dfetch<float>(d_tex_location[1], i);
#ifdef _3D
                        neighbourPos.z = tex1Dfetch<float>(d_tex_location[2], i);
#endif
#endif
#endif
                        newLoc += reduceCircles(myLoc, neighbourPos, ct);
                    }
                }
            }
            /**
            * End Same block
            */
    }
#elif defined(MODULAR_STRIPS)
    {
#if defined(_3D)
        DIMENSIONS_IVEC binFirst = DIMENSIONS_IVEC(-1, x + offset.x, y + offset.y);
#elif defined(_2D)
        DIMENSIONS_IVEC binFirst = DIMENSIONS_IVEC(-1, x + offset);
#else
#error "Unexpected dims"
#endif
        //For each axis, if new relative > 1, set -1
        binFirst.y = binFirst.y>1 ? binFirst.y - 3 : binFirst.y;
#ifdef _3D
        binFirst.z = binFirst.z>1 ? binFirst.z - 3 : binFirst.z;
#endif
        binFirst+=myGrid;
        DIMENSIONS_IVEC binLast = binFirst;
        binLast.x += 2;
        bool firstInvalid = invalidBinX(binFirst);
        bool lastInvalid = invalidBinX(binLast);
#if defined(_3D)
        if (!invalidBinYZ(binFirst))
#elif defined(_2D)
        if (!invalidBinY(binFirst))
#else
#error "Unexpected dims"
#endif
        {
            if (firstInvalid)
            {
                binFirst.x = 0;
            }
            if (lastInvalid)
            {//If strip ends out of bounds only
                binLast.x = d_gridDim.x - 1;//Max x coord
            }
            unsigned int binFirstHash = getHash(binFirst);
            unsigned int binLastHash = binFirstHash + (binLast.x - binFirst.x);//Strips at most len 3
#if defined(GLOBAL_PBM)
            unsigned int binIndex = d_pbm[binFirstHash];
            unsigned int binIndexMax = d_pbm[binLastHash + 1];
#elif defined(LDG_PBM)
            unsigned int binIndex = __ldg(&d_pbm[binFirstHash]);
            unsigned int binIndexMax = __ldg(&d_pbm[binLastHash + 1]);
#else
            unsigned int binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, binFirstHash);
            unsigned int binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, binLastHash + 1);
#endif
            /**
            * Start Same Block
            */
            if (binIndex < binIndexMax)
            {
                for (unsigned int i = binIndex; i < binIndexMax; ++i)
                {
                    if (i != id)
                    {
                        DIMENSIONS_VEC neighbourPos;
#if defined(GLOBAL_MESSAGES)
#ifdef AOS_MESSAGES
                        neighbourPos = d_messages->location[i];
#else
                        neighbourPos.x = d_messages->locationX[i];
                        neighbourPos.y = d_messages->locationY[i];
#ifdef _3D
                        neighbourPos.z = d_messages->locationZ[i];
#endif
#endif
#elif defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
                        //sm_message->location = __ldg(&d_messages->location[i]);
                        neighbourPos.x = __ldg(&d_messages->location[i].x);
                        neighbourPos.y = __ldg(&d_messages->location[i].y);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->location[i].z);
#endif
#else
                        neighbourPos.x = __ldg(&d_messages->locationX[i]);
                        neighbourPos.y = __ldg(&d_messages->locationY[i]);
#ifdef _3D
                        neighbourPos.z = __ldg(&d_messages->locationZ[i]);
#endif
#endif
#else//Read message data from tex cache (default)
#ifdef AOS_MESSAGES
#error TODO Read AOS Messages from explicit texture
#else
                        neighbourPos.x = tex1Dfetch<float>(d_tex_location[0], i);
                        neighbourPos.y = tex1Dfetch<float>(d_tex_location[1], i);
#ifdef _3D
                        neighbourPos.z = tex1Dfetch<float>(d_tex_location[2], i);
#endif
#endif
#endif
                        newLoc += reduceCircles(myLoc, neighbourPos, ct);
                    }
                }
            }
            /**
            * End Same block
            */
        }
    }
#else
    {
        //Calc current bin
#ifdef _3D
        DIMENSIONS_IVEC binPos = myGrid + DIMENSIONS_IVEC(x, y, z);
#else
        DIMENSIONS_IVEC binPos = myGrid + DIMENSIONS_IVEC(x, y);
#endif
        //Get bin  start/end index
        unsigned int binHash = getHash(binPos);
#if defined(GLOBAL_PBM)
        unsigned int binIndex = d_pbm[binHash];
        unsigned int binIndexMax = d_pbm[binHash + 1];
#elif defined(LDG_PBM)
        unsigned int binIndex = __ldg(&d_pbm[binHash]);
        unsigned int binIndexMax = __ldg(&d_pbm[binHash + 1]);
#else
        unsigned int binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, binHash);
        unsigned int binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, binHash + 1);
#endif
        /**
         * Start Same Block
         */
        if (binIndex < binIndexMax)
        {
            for (unsigned int i = binIndex; i < binIndexMax; ++i)
            {
                if (i != id)
                {
                    DIMENSIONS_VEC neighbourPos;
#if defined(GLOBAL_MESSAGES)
#ifdef AOS_MESSAGES
                    neighbourPos = d_messages->location[i];
#else
                    neighbourPos.x = d_messages->locationX[i];
                    neighbourPos.y = d_messages->locationY[i];
#ifdef _3D
                    neighbourPos.z = d_messages->locationZ[i];
#endif
#endif
#elif defined(LDG_MESSAGES)
#ifdef AOS_MESSAGES
                    //sm_message->location = __ldg(&d_messages->location[i]);
                    neighbourPos.x = __ldg(&d_messages->location[i].x);
                    neighbourPos.y = __ldg(&d_messages->location[i].y);
#ifdef _3D
                    neighbourPos.z = __ldg(&d_messages->location[i].z);
#endif
#else
                    neighbourPos.x = __ldg(&d_messages->locationX[i]);
                    neighbourPos.y = __ldg(&d_messages->locationY[i]);
#ifdef _3D
                    neighbourPos.z = __ldg(&d_messages->locationZ[i]);
#endif
#endif
#else//Read message data from tex cache (default)
#ifdef AOS_MESSAGES
#error TODO Read AOS Messages from explicit texture
#else
                    neighbourPos.x = tex1Dfetch<float>(d_tex_location[0], i);
                    neighbourPos.y = tex1Dfetch<float>(d_tex_location[1], i);
#ifdef _3D
                    neighbourPos.z = tex1Dfetch<float>(d_tex_location[2], i);
#endif
#endif
#endif
                    newLoc += reduceCircles(myLoc, neighbourPos, ct);
                }
            }
        }
        /**
         * End Same block
         */
    }
#endif
    //Export newLoc
    newLoc /= ct>0 ? ct : 1;
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
//__global__ void step_circles_model3(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut)
//{
//    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if (id >= d_locationMessageCount)
//        return;
//
//    const DIMENSIONS_VEC myLoc(locationMessagesIn->locationX[id], locationMessagesIn->locationY[id], locationMessagesIn->locationZ[id]);
//
//    DIMENSIONS_IVEC myGrid = getGridPosition(myLoc);
//    DIMENSIONS_VEC newLoc = DIMENSIONS_VEC(0);
//    unsigned int ct = 0;
//
//#pragma unroll
//    for (int x = -1; x <= 1; ++x)
//#pragma unroll
//        for (int y = -1; y <= 1; ++y)
//#pragma unroll
//            for (int z = -1; z <= 1; ++z)
//            {
//        DIMENSIONS_IVEC binPos = myGrid + DIMENSIONS_IVEC(x, y, z);
//        //Get bin  start/end index
//        unsigned int binHash = getHash(binPos);
//        unsigned int binIndex = tex1Dfetch<unsigned int>(d_tex_PBM, binHash);
//        unsigned int binIndexMax = tex1Dfetch<unsigned int>(d_tex_PBM, binHash + 1);
//
//        /**
//        * Start Same Block
//        */
//        if (binIndex < binIndexMax)
//        {
//            for (unsigned int i = binIndex; i < binIndexMax; ++i)
//            {
//                if (i != id)
//                {
//                    DIMENSIONS_VEC neighbourPos;
//
//                    neighbourPos.x = __ldg(&d_messages->locationX[i]);
//                    neighbourPos.y = __ldg(&d_messages->locationY[i]);
//                    neighbourPos.z = __ldg(&d_messages->locationZ[i]);
//
//                    newLoc += reduceCircles(myLoc, neighbourPos, ct);
//                }
//            }
//        }
//        /**
//        * End Same block
//        */
//            }
//    //Export newLoc
//    newLoc /= ct>0 ? ct : 1;
//    newLoc += myLoc;
//
//    newLoc = glm::clamp(newLoc, d_environmentMin, d_environmentMax);
//    locationMessagesOut->locationX[id] = newLoc.x;
//    locationMessagesOut->locationY[id] = newLoc.y;
//    locationMessagesOut->locationZ[id] = newLoc.z;
//}