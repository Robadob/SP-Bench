#ifndef __BuildAll_cuh__
#define __BuildAll_cuh__

//#define _2D
#define _3D

/**
 * Config
 */
#define ATOMIC_PBM //Improves construction perf over large strips of empty bins

//#define GLOBAL_MESSAGES
//#define LDG_MESSAGES

//#define SHARED_BINSTATE //Defines storage of sm_message (and bin state) in shared memory

//#define STRIDED_MESSAGES //Harmful to perf

//#define GLOBAL_PBM //Negligible effect to perf
//#define LDG_PBM //Negligible effect to perf

//#define AOS_MESSAGES //This is how the CUDA_Particles example operates, also makes larger messages easier to do in a templated manner

/**
 * Build includes
 */
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