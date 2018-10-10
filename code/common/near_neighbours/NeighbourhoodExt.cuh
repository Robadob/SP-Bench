#ifndef __NeighbourhoodExt_cuh__
#define __NeighbourhoodExt_cuh__

#include "Neighbourhood.cuh"

/**
 * This wraps SpatialPartition to allow tertiary data to be attached to messages
 */

template<class T>
class SpatialPartitionExt : private SpatialPartition
{
public:
    SpatialPartitionExt(unsigned int binCount, unsigned int maxAgents);
    SpatialPartitionExt(DIMENSIONS_VEC  environmentMin, DIMENSIONS_VEC environmentMax, unsigned int maxAgents, float interactionRad);
    ~SpatialPartitionExt();
    //Getters
    using SpatialPartition::d_getPBMIndex;
    using SpatialPartition::d_getPBMCount;
    using SpatialPartition::d_getLocationMessages;
    using SpatialPartition::d_getLocationMessagesSwap;
    using SpatialPartition::getLocationCount;
    //Setters
    using SpatialPartition::setLocationCount;
    //Util
    using SpatialPartition::buildPBM;
    virtual void swap() override;
    using SpatialPartition::getGridDim;
    using SpatialPartition::getEnvironmentMin;
    using SpatialPartition::getEnvironmentMax;
    using SpatialPartition::getEnvironmentDimensions;
    using SpatialPartition::getCellSize;
    using SpatialPartition::isValid;
    using SpatialPartition::getPos;
    using SpatialPartition::getGridPosition;
    using SpatialPartition::getHash;
#ifdef _DEBUG
    using SpatialPartition::assertSearch;
    using SpatialPartition::launchAssertPBMIntegerity;
#endif
    using SpatialPartition::getNeighbourhoodStats;
#ifdef _GL
    using SpatialPartition::getLocationTexNames;
    using SpatialPartition::getCountTexName;
#endif
private:
    void deviceAllocateExt(T **d_data);
    void deviceDeallocateExt(T *d_data);
    //Kernel launchers
    virtual void launchReorderLocationMessages() override;
    //Device pointers
    T *d_ext, *d_ext_swap;

};
#endif //__NeighbourhoodExt_cuh__
