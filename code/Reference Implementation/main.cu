#include "Neighbourhood.cuh"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>


int main()
{
    //Define Neighbourhood Limits
    glm::vec3 envMin(-500.0, -500.0, -500.0);
    glm::vec3 envMax(500.0, 500.0, 500.0);
    unsigned int agentMax = 100000;
    //Allocate USP
    SpatialPartition sp(envMin, envMax, agentMax, 10.0);
    //Fill Neighbourhood
    LocationMessages *d_LM = sp.d_getLocationMessages();
       ///Some kernel
    //Sort/Construct USP
    sp.buildPBM();
    //Neighbourhood Search

    //Deallocate USP

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