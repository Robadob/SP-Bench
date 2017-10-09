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
        const float density = 0.125f,
        const float interactionRad=10.0f
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
    static unsigned int NullModel::toWidth(unsigned int agents, float density);
    static unsigned int NullModel::toAgents(unsigned int width, float density);
};


inline unsigned int NullModel::toWidth(unsigned int agents, float density)
{
#if DIMENSIONS == 2
    return (unsigned int)round(sqrt(agents/density));
#elif DIMENSIONS == 3
    return (unsigned int) round(cbrt(agents / density));
#else
#error DIMENSIONS must equal 2 or 3
#endif
}
inline unsigned int NullModel::toAgents(unsigned int width, float density)
{
    return (unsigned int)(round(pow(width, DIMENSIONS) * (double)density));
}
NullModel::NullModel(
    const unsigned int agents,
    const float density,
    const float interactionRad
    )
    : CoreModel(toAgents(toWidth(agents, density), density))
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f), DIMENSIONS_VEC(toWidth(agents, density)), agentMax, interactionRad))
    , d_result(nullptr)
    , h_result(nullptr)
    , width(toWidth(agents, density))
    , density(density)
    , interactionRad(interactionRad)
{
#ifdef _DEBUG
    printf("Null Model: Agent Count(%d), Width(%d)\n", agentMax, width);
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
    spatialPartition->buildPBM();

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

    return rtn;
}
int requiredSM_stepNullModel(int blockSize)
{
    return sizeof(LocationMessage)*blockSize;
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
