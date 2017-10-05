#define MOD_NAME "Individual Binning (default)"

//No options, this is default 


//Move below into a single include?
//Build the data-structure
#include "near_neighbours/Neighbourhood.cu"
#include "near_neighbours/NeighbourhoodKernels.cu"
//Build the Benchmark Models 
//Circles model (defines CIRCLES_MODEL)
#include "benchmarks/circles/Circles.cuh"
//Circles model (defines NULL_MODEL)
//#include "benchmarks/null/Null.cuh"
//Circles model (defines DENSITY_MODEL)
//#include "benchmarks/density/Density.cuh"

//Build the actual entry point
#include "main.cu"

//Build the other dependencies
#include "export.cpp"

//Do we also want visualisation?
#ifdef _GL
#include "ParticleScene.hpp"
#include "BuildVisualisation.h"
#endif