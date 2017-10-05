#ifndef __Model_cuh__
#define __Model_cuh__

#include <math.h>
#include "results.h"
class CirclesModel : public Model
{
public:
    CirclesModel(
        const unsigned int width = 250,
        const float density = 1.0,
        const float interactionRad = 10.0,
        const float attract = 5.0,
        const float repulse = 5.0
        );
    //Returns the time taken
    const Time_Init initPopulation(const unsigned long long rngSeed = 12) override;
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
    const float attract;
    const float repulse;
};

extern __device__ __constant__ float d_attract;
extern __device__ __constant__ float d_repulse;

CirclesModel::CirclesModel(
    const unsigned int width,
    const float density,
    const float interactionRad,
    const float attract,
    const float repulse
    )
    : Model((int)(round(pow(width, DIMENSIONS) * (double)density)))
    , spatialPartition(std::make_shared<SpatialPartition>(DIMENSIONS_VEC(0.0f, 0.0f, 0.0f), DIMENSIONS_VEC(width, width, width), (int)round(pow(width, DIMENSIONS) * (double)density), interactionRad))
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

const Time_Init CirclesModel::initPopulation(const unsigned long long rngSeed)
{
    cudaEvent_t start_overall, start_kernel, start_pbm, start_free, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&start_pbm);
    cudaEventCreate(&start_free);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Generate curand
    curandState *d_rng;
    CUDA_CALL(cudaMalloc(&d_rng, agentMax*sizeof(curandState)));
    //Arbitrary thread block sizes (speed not too important during one off initialisation)
    unsigned int initThreads = 512;
	unsigned int initBlocks = (agentMax / initThreads) + 1;
    spatialPartition->setLocationCount(agentMax);
	if (rngSeed!=0)
	{
		init_curand << <initBlocks, initThreads >> >(d_rng, rngSeed);
		CUDA_CALL(cudaDeviceSynchronize());
	}

    //End curand timer/start kernel timer
    cudaEventRecord(start_kernel);

    //Generate initial states, and store in location messages
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
	if (rngSeed != 0)
		init_particles << <initBlocks, initThreads >> >(d_rng, d_lm);
	else
	{
		init_particles_uniform << <initBlocks, initThreads >> >(d_lm);
//Not sure why this ifdef is required
#ifdef _GL
		int bins = glm::compMul(spatialPartition->getGridDim());
		int spareP = agentMax%bins;
		fprintf(stderr, "Bins: %i, Agents: %i\n", bins, agentMax);
		if (spareP>0)
		{
			fprintf(stderr,"Warning %.1f%% of bins have an extra agent!\n", (spareP / (float)bins)*100);
		}
#endif
	}
    CUDA_CALL(cudaDeviceSynchronize());

    //End kernel timer/start pbm timer
    cudaEventRecord(start_pbm);

    //generate pbm
    spatialPartition->buildPBM();//This may error due to attempt to deallocate unallocated textures

    //End pbm timer/start free timer
    cudaEventRecord(start_free);

    //Free curand
    CUDA_CALL(cudaFree(d_rng));

    //End overall timer
    cudaEventRecord(stop_overall);

    //Calculate return struct
    //cudaEventSynchronize(start_overall);//Only really required to synchronise the last event
    //cudaEventSynchronize(start_kernel);
    //cudaEventSynchronize(start_pbm);
    //cudaEventSynchronize(start_free);
    cudaEventSynchronize(stop_overall);
    Time_Init rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.initCurand, start_overall, start_kernel);
    cudaEventElapsedTime(&rtn.kernel, start_kernel, start_pbm);
    cudaEventElapsedTime(&rtn.pbm, start_pbm, start_free);
    cudaEventElapsedTime(&rtn.freeCurand, start_free, stop_overall);

    return rtn;
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
int requiredSM_stepModel(int blockSize)
{
    return sizeof(LocationMessage)*blockSize;
}
void CirclesModel::launchStep()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, step_model, requiredSM_stepModel, 0);//random 128
    // Round up according to array size
    int gridSize = (agentMax + blockSize - 1) / blockSize;
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
    LocationMessages *d_lm2 = spatialPartition->d_getLocationMessagesSwap();
    //Launch kernel
    step_model<<<gridSize, blockSize, requiredSM_stepModel(blockSize) >> >(d_lm, d_lm2);
    //Swap
    spatialPartition->swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}

#endif
