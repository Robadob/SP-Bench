#include "Neighbourhood.cuh"
#include "Circles.cuh"
#include "Visualisation/Visualisation.h"
#include "ParticleScene.h"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

__global__ void initLocations(
    LocationMessages *messages
    )
{

    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= d_locationMessageCount) return;

    //messages->locationX[index] = ;
   // messages->locationY[index] = ;
#ifdef _3D
   // messages->locationZ[index] = ;
#endif
}


int main()
{
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const unsigned int width = 50;
    const float density = 0.005f;
    const float interactionRad = 10.0f;
    const float attractionForce = 0.0001f;
    const float repulsionForce = 0.0001f;
    const unsigned long long iterations = 10000;

    Visualisation v("Visulisation Example", 1280, 720);
    Circles<SpatialPartition> model(width, density, interactionRad, attractionForce, repulsionForce);
    const Time_Init initTimes = model.initPopulation();//Need to init textures before creating the scene
    ParticleScene<SpatialPartition> *scene = new ParticleScene<SpatialPartition>(v, model);

    //Init model
    printf("Init Complete - Times\n");
    printf("CuRand init - %.3fs\n", initTimes.initCurand / 1000);
    printf("Main kernel - %.3fs\n", initTimes.kernel / 1000);
    printf("Build PBM   - %.3fs\n", initTimes.pbm / 1000);
    printf("CuRand free - %.3fs\n", initTimes.freeCurand / 1000);
    printf("Combined    - %.3fs\n", initTimes.overall / 1000);
    printf("\n");
    //Start visualisation
    //v.runAsync();
    //v.run();
    //Do iterations
    Time_Step_dbl average = {};//init

    printf("\n");
    for (unsigned long long i = 0; i < iterations; i++)
    {
        const Time_Step iterTime = model.step();
        //Pass count to visualisation
        scene->setCount(model.getPartition()->getLocationCount());
        //Calculate averages
        average.overall += iterTime.overall / iterations;
        average.kernel += iterTime.kernel / iterations;
        average.texture += iterTime.texture / iterations;
        v.render();
        printf("\r%6llu/%llu", i, iterations);
    }
    printf("Model complete - Average Times\n");
    printf("Main kernel - %.3fs\n", average.kernel / 1000);
    printf("Build PBM   - %.3fs\n", average.texture / 1000);
    printf("Combined    - %.3fs\n", average.overall / 1000);
    printf("\n");

    //Calculate final timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

    printf("Total Runtime: %.3fs\n", totalTime * 1000);

    v.run();

    //Wait for input before exit
    getchar();
    return 0;
}
