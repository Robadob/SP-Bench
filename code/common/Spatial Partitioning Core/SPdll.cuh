#ifndef SPdll_cuh
#define SPdll_cuh

//Enable DLL exports
#define SP_EXPORTS
#ifdef SP_EXPORTS  
#define SP_API __declspec(dllexport)   
#else  
#define SP_API __declspec(dllimport)   
#endif 

#define _3D
#include "Neighbourhood.cuh"
#include <glm/glm.hpp>
namespace
{//anonymous namespace to store instance
    SpatialPartition *sp=nullptr;
}


void initSP(DIMENSIONS_VEC envMin, DIMENSIONS_VEC envMax, unsigned int maxAgents, float interactionRad){
    if (sp)
        delete sp;
    sp = new SpatialPartition(envMin, envMax, maxAgents, interactionRad);
}

void rebuildSP(){
    
}

//Return function pointer to device insertion method?
void insertSP(){
    
}

//Return function pointer to device search method?
void searchSP(){
    
}
void freeSP()
{
    delete sp;
    sp = nullptr;
}
#include "Neighbourhood.cu"
#include "NeighbourhoodKernels.cu"
#endif //SPdll_cuh