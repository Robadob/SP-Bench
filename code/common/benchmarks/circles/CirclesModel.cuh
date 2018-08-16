#ifndef __CirclesModel_cuh__
#define __CirclesModel_cuh__

#include <math.h>
#include "results.h"
#include "../core/CoreModel.cuh"
#include "CirclesKernels.cuh"
class CirclesModel : public CoreModel
{
public:
    CirclesModel(
        const unsigned int agents = 16384,
        const float density = 0.125f,
        const float forceModifier = 5.0
        );
    //Original constructor
    CirclesModel(
        const float density,
        const unsigned int width = 250,
        const float interactionRad = 10.0,
        const float attract = 5.0,
        const float repulse = 5.0
        );
    //Returns the time taken
    const Time_Step step() override;
private:
    void launchStep();//Launches step_model kernel
    std::shared_ptr<SpatialPartition> spatialPartition;
    //If values are constant, might aswell make them public, save writing accessor methods
public:
    std::shared_ptr<SpatialPartition> getPartition() override { return spatialPartition; }
    const unsigned int width;
    const float density;
    const float interactionRad;
    const float attract;//Redundant under sin model
    const float repulse;
};

//Required to remove extern for RDC=false
//extern __device__ __constant__ float d_attract;
//extern __device__ __constant__ float d_repulse;
__device__ __constant__ float d_attract;
__device__ __constant__ float d_repulse;

CirclesModel::CirclesModel(
    const unsigned int agents,
    const float density,
    const float forceModifier
    )
    : CoreModel(agents)
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f), DIMENSIONS_VEC((float)toWidth(agents, density)), agentMax, 1.0f))
    , width(toWidth(agents, density))
    , density(density)
    , interactionRad(1.0f)
    , attract(0.0f)
    , repulse(forceModifier)
{
    //Copy relevant parameters to constants
    CUDA_CALL(cudaMemcpyToSymbol(d_attract, &attract, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(d_repulse, &repulse, sizeof(float)));
}
CirclesModel::CirclesModel(
    const float density,
    const unsigned int width,
    const float interactionRad,
    const float attract,
    const float repulse
    )
    : CoreModel((int)(round(pow(width, DIMENSIONS) * (double)density)))
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f), DIMENSIONS_VEC((float)width), (int)round(pow(width, DIMENSIONS) * (double)density), interactionRad))
    , width(width)
    , density(density)
    , interactionRad(interactionRad)
    , attract(attract)
    , repulse(repulse)
{
    //Copy relevant parameters to constants
    CUDA_CALL(cudaMemcpyToSymbol(d_attract, &attract, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(d_repulse, &repulse, sizeof(float)));
    ///CUDA constants managed by Neighbourhood.cuh
    //float d_interactionRad == interactionRad
    //unsigned int d_locationMessageCount == agentMax (in this case where all agents submit a single location message)
    //glm::vec3  d_environmentMin == min env values (for wrapping purposes)
    //glm::vec3  d_environmentMax == max env values (for wrapping purposes)

}
const Time_Step CirclesModel::step()
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
int requiredSM_stepCirclesModel(int blockSize)
{
    return SpatialPartition::requiredSM(blockSize);
}
void CirclesModel::launchStep()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, step_circles_model, requiredSM_stepCirclesModel, 0);//random 128
    // Round up according to array size
    int gridSize = (agentMax + blockSize - 1) / blockSize;
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
    LocationMessages *d_lm2 = spatialPartition->d_getLocationMessagesSwap();
    //Launch kernel
    step_circles_model<<<gridSize, blockSize, requiredSM_stepCirclesModel(blockSize)>>>(d_lm, d_lm2);
    //Swap
    spatialPartition->swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}

#endif //__CirclesModel_cuh__
