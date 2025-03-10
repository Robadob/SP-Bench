#include <cuda_device_runtime_api.h>
#ifndef __NullModel_cuh__
#define __NullModel_cuh__

#include <cmath>
#include <memory>
#include "results.h"
#include "../core/CoreModel.cuh"
#include "NullKernels.cuh"
class NullModel : public CoreModel
{
public:
    NullModel(
        const unsigned int agents = 16384,
        const float density = 0.125f
        );
    //Density alt constructor
    NullModel(
        const float envWidth,
        const float interactionRad,
        const unsigned int clusterCount,
        const unsigned int agentsPerCluster,
        const float uniformDensity = 0
        );
    ~NullModel();
    //Returns the time taken
    const Time_Step step() override;
private:
    void launchStep();//Launches step_model kernel
    std::shared_ptr<SpatialPartition> spatialPartition;
    //If values are constant, might aswell make them public, save writing accessor methods
    DIMENSIONS_VEC *d_result;
    DIMENSIONS_VEC *h_result;
public:
    const DIMENSIONS_VEC *getResults();
    std::shared_ptr<SpatialPartition> getPartition() override { return spatialPartition; }
    const unsigned int width;
    const float density;
    const float interactionRad;
};
NullModel::NullModel(
    const unsigned int agents,
    const float density
    )
    : CoreModel(agents)
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f), DIMENSIONS_VEC((float)toWidth(agents, density)), agentMax, 1.0f))
    , d_result(nullptr)
    , h_result(nullptr)
    , width(toWidth(agents, density))
    , density(density)
    , interactionRad(1.0f)
{
#ifdef _DEBUG
    printf("Null Model: Agent Count(%d), Width(%d)\n", agentMax, width);
#endif
    CUDA_CALL(cudaMalloc(&d_result, agentMax*sizeof(DIMENSIONS_VEC)));
    CUDA_CALL(cudaMemset(d_result, 0, agentMax*sizeof(DIMENSIONS_VEC)));
    h_result = (DIMENSIONS_VEC*)malloc(agentMax*sizeof(DIMENSIONS_VEC));
}
NullModel::NullModel(
    const float envWidth,
    const float interactionRad,
    const unsigned int clusterCount,
    const unsigned int agentsPerCluster,
    const float uniformDensity
    )
    : CoreModel((clusterCount*agentsPerCluster)+toAgents((unsigned int)(envWidth/interactionRad), uniformDensity))//Calculate total agents
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f), DIMENSIONS_VEC(envWidth), agentMax, interactionRad))
    , d_result(nullptr)
    , h_result(nullptr)
    , width((unsigned int)envWidth)//Unsigned or float?
    , density(pow(envWidth, DIMENSIONS)/agentMax)
    , interactionRad(interactionRad)
{
#ifdef _DEBUG
    printf("Null Model (Density): Agent Count(%d), Width(%d)\n", agentMax, width);
#endif
    CUDA_CALL(cudaMalloc(&d_result, agentMax*sizeof(DIMENSIONS_VEC)));
    CUDA_CALL(cudaMemset(d_result, 0, agentMax*sizeof(DIMENSIONS_VEC)));
    h_result = (DIMENSIONS_VEC*)malloc(agentMax*sizeof(DIMENSIONS_VEC));
}
NullModel::~NullModel()
{
    cudaFree(d_result);
    free(h_result);
}

const DIMENSIONS_VEC *NullModel::getResults()
{
    CUDA_CALL(cudaMemcpy(h_result, d_result, agentMax*sizeof(DIMENSIONS_VEC), cudaMemcpyDeviceToHost));
    return h_result;
}
const Time_Step NullModel::step()
{
    cudaEvent_t start_overall, start_texture, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_texture);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Run single iteration of model
    launchStep();

    //End kernel timer/start texture reset timer
    cudaEventRecord(start_texture);

    //Rebuild PBM
    PBM_Time p = spatialPartition->buildPBM();

    //End overall timer
    cudaEventRecord(stop_overall);

    //Calculate return struct
    //cudaEventSynchronize(start_overall);//Only really required to synchronise the last event
    //cudaEventSynchronize(start_texture);
    cudaEventSynchronize(stop_overall);
    Time_Step rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.kernel, start_overall, start_texture);
    cudaEventElapsedTime(&rtn.texture, start_texture, stop_overall);
    rtn.pbm = p;
    return rtn;
}
int requiredSM_stepNullModel(int blockSize)
{
    return SpatialPartition::requiredSM(blockSize);
}
void NullModel::launchStep()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, step_null_model, requiredSM_stepNullModel, 0);//random 128
    // Round up according to array size
    int gridSize = (agentMax + blockSize - 1) / blockSize;
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
    LocationMessages *d_lm2 = spatialPartition->d_getLocationMessagesSwap();
    //Launch kernel
    step_null_model<<<gridSize, blockSize, requiredSM_stepNullModel(blockSize)>>>(d_lm, d_lm2, d_result);
    //Swap
    spatialPartition->swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}

#endif //__NullModel_cuh__
