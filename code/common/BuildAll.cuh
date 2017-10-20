#ifndef __BuildAll_cuh__
#define __BuildAll_cuh__

#define _2D

//Build the data-structure
#include "near_neighbours/Neighbourhood.cu"
#include "near_neighbours/NeighbourhoodKernels.cu"
//Build the Benchmark Models 
#include "benchmarks/core/Core.cuh"
//Circles model (defines CIRCLES_MODEL)
#include "benchmarks/circles/Circles.cuh"
//Circles model (defines NULL_MODEL, DENSITY_MODEL)
#include "benchmarks/null/Null.cuh"

//Build the actual entry point
#include "main.cu"

//Build the other dependencies
#include "export.cpp"

//Do we also want visualisation?
#ifdef _GL
#include "ParticleScene.hpp"
#include "BuildVisualisation.h"
#endif

#endif //__BuildAll_cuh__