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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const unsigned int width = 50;
    const float density = 0.005;
    const float interactionRad = 10.0;
    const float attractionForce = 0.0001;
    const float repulsionForce = 0.0001;
    const unsigned long long iterations = 10000;
    Visualisation<ParticleScene> v = Visualisation<ParticleScene>("Visulisation Example", 1280, 720);
    v.setRenderAxis(true);
    Circles<SpatialPartition> model(width, density, interactionRad, attractionForce, repulsionForce);

    //getLocationTexNames() pASS TEX NAMES TO VISUALISATION
    GLuint *loc_texs = model.getPartition()->getLocationTexNames();
    v.getScene()->setTex(loc_texs);
    //Init model
    const Time_Init initTimes = model.initPopulation();
    printf("Init Complete - Times\n");
    printf("CuRand init - %.3fs\n", initTimes.initCurand * 1000);
    printf("Main kernel - %.3fs\n", initTimes.kernel * 1000);
    printf("Build PBM   - %.3fs\n", initTimes.pbm * 1000);
    printf("CuRand free - %.3fs\n", initTimes.freeCurand * 1000);
    printf("Combined    - %.3fs\n", initTimes.overall * 1000);
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
        v.getScene()->setCount(model.getPartition()->getLocationCount());
        //Calculate averages
        average.overall += iterTime.overall / iterations;
        average.kernel += iterTime.kernel / iterations;
        average.texture += iterTime.texture / iterations;
        v.renderStep();
        printf("\r%6llu/%llu", i, iterations);
    }
    printf("Model complete - Average Times\n");
    printf("Main kernel - %.3fs\n", average.kernel * 1000);
    printf("Build PBM   - %.3fs\n", average.texture * 1000);
    printf("Combined    - %.3fs\n", average.overall * 1000);
    printf("\n");

    //Calculate final timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

    printf("Total Runtime: %.3fs\n", totalTime * 1000);

    //Wait for input before exit
    getchar();
    return 0;
}
