#ifndef __Model_cuh__
#define __Model_cuh__

#include <math.h>
#include "results.h"
template <class T>
class Circles
{
public:
    Circles(
        const unsigned int width = 250,
        const float density = 1.0,
        const float interactionRad = 10.0,
        const float attract = 5.0,
        const float repulse = 5.0
        );
    //Returns the time taken
    const Time_Init initPopulation(const unsigned long long rngSeed = 12);
    const Time_Step step();
private:
    void launchStep();//Launches step_model kernel
    T *spatialPartition;
	unsigned int smCount;//Count of streaming multi processors on current GPU
    //If values are constant, might aswell make them public, save writing accessor methods
public:
    T *getPartition(){ return spatialPartition; }
    const unsigned int width;
    const float density;
    const float interactionRad;
    const float attract;
    const float repulse;
    const unsigned int agentMax;
};

extern __device__ __constant__ float d_attract;
extern __device__ __constant__ float d_repulse;

template <class T>
Circles<T>::Circles(
    const unsigned int width,
    const float density,
    const float interactionRad,
    const float attract,
    const float repulse
    )
    : width(width)
    , density(density)
    , interactionRad(interactionRad)
    , attract(attract)
    , repulse(repulse)
    , agentMax((int)(round(pow(width, DIMENSIONS) * (double)density)))
	, spatialPartition(new SpatialPartition(DIMENSIONS_VEC(0.0f, 0.0f, 0.0f), DIMENSIONS_VEC(width, width, width), (int)round(pow(width, DIMENSIONS) * (double)density), interactionRad))
{
    //Copy relevant parameters to constants
    CUDA_CALL(cudaMemcpyToSymbol(d_attract, &attract, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(d_repulse, &repulse, sizeof(float)));
    ///CUDA constants managed by Neighbourhood.cuh
    //float d_interactionRad == interactionRad
    //unsigned int d_locationMessageCount == agentMax (in this case where all agents submit a single location message)
    //glm::vec3  d_environmentMin == min env values (for wrapping purposes)
    //glm::vec3  d_environmentMax == max env values (for wrapping purposes)
	cudaDeviceProp properties;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&properties, device);
	this->smCount = properties.multiProcessorCount;
}

template <class T>
const Time_Init Circles<T>::initPopulation(const unsigned long long rngSeed)
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
template <class T>
const Time_Step Circles<T>::step()
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
template <class T>
void Circles<T>::launchStep()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, step_model, requiredSM_stepModel, 0);//random 128
    // Round up according to array size
    int gridSize = (agentMax + blockSize - 1) / blockSize;
	////If grid size is less than SMs, reduce block size
	//if ((unsigned int)gridSize < this->smCount*2)
	//{
	//	gridSize = this->smCount*5;
	//	blockSize = (agentMax / (this->smCount*2)) + 1;
	//}
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
    LocationMessages *d_lm2 = spatialPartition->d_getLocationMessagesSwap();
    //Launch kernel
#ifdef _local
	step_model <<<gridSize, blockSize>>>(d_lm, d_lm2);
#else
	step_model << <gridSize, blockSize, requiredSM_stepModel(blockSize) >> >(d_lm, d_lm2);//CHANGED: Don't sort particles
#endif
	CUDA_CHECK();
    //Swap
    spatialPartition->swap();//CHANGED: Don't sort particles
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}

#endif
