#include "Neighbourhood.cuh"
#include "Circles.cuh"
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

    const unsigned int width = 250;
    const float density = 1.0;
    const float interactionRad = 10.0;
    const float attractionForce = 5.0;
    const float repulsionForce = 5.0;
    const unsigned long long iterations = 10000;
    Circles<SpatialPartition> model(width, density, interactionRad, attractionForce, repulsionForce);

    const Time_Init initTimes = model.initPopulation();
    printf("Init Complete - Times\n");
    printf("CuRand init - %.3fs\n", initTimes.initCurand * 1000);
    printf("Main kernel - %.3fs\n", initTimes.kernel * 1000);
    printf("Build PBM   - %.3fs\n", initTimes.pbm * 1000);
    printf("CuRand free - %.3fs\n", initTimes.freeCurand * 1000);
    printf("Combined    - %.3fs\n", initTimes.overall * 1000);
    printf("\n");

    Time_Step_dbl average = {};//init
    for (unsigned long long i = 0; i < iterations; i++)
    {
        const Time_Step iterTime = model.step();
        //Calculate averages
        average.overall += iterTime.overall / iterations;
        average.kernel += iterTime.kernel / iterations;
        average.texture += iterTime.texture / iterations;
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

//struct SpatialSOA
//{
//    float *locationX;
//    float *locationY;
//    float *locationZ;
//    //optional 
//    float *directionX;
//    float *directionY;
//    float *directionZ;
//    float *velocity;
//};
///*
// Device spatial stack of array
//*/
//struct D_SpatialSOA : SpatialSOA
//{
//    __host__ D_SpatialSOA(int len)
//    {
//        cudaMalloc(&locationX, len*sizeof(float));
//        cudaMalloc(&locationY, len*sizeof(float));
//        cudaMalloc(&locationZ, len*sizeof(float));
//        //optional
//        cudaMalloc(&directionX, len*sizeof(float));
//        cudaMalloc(&directionY, len*sizeof(float));
//        cudaMalloc(&directionZ, len*sizeof(float));
//        cudaMalloc(&velocity, len*sizeof(float));
//    }
//    __host__ ~D_SpatialSOA()
//    {
//        cudaFree(locationX);
//        cudaFree(locationY);
//        cudaFree(locationZ);
//        //optional
//        cudaFree(directionX);
//        cudaFree(directionY);
//        cudaFree(directionZ);
//        cudaFree(velocity);
//    }
//};
///*
//Device spatial stack of array
//*/
//struct H_SpatialSOA : SpatialSOA
//{
//    __host__ H_SpatialSOA(int len)
//    {
//        locationX = (float*)malloc(len*sizeof(float));
//        locationY = (float*)malloc(len*sizeof(float));
//        locationZ = (float*)malloc(len*sizeof(float));
//        //optional
//        directionX = (float*)malloc(len*sizeof(float));
//        directionY = (float*)malloc(len*sizeof(float));
//        directionZ = (float*)malloc(len*sizeof(float));
//        velocity = (float*)malloc(len*sizeof(float));
//    }
//    __host__ ~H_SpatialSOA()
//    {
//        free(locationX);
//        free(locationY);
//        free(locationZ);
//        //optional
//        free(directionX);
//        free(directionY);
//        free(directionZ);
//        free(velocity);
//    }
//};
