#ifndef __export_h__
#define __export_h__

#include "near_neighbours/Neighbourhood.cuh"
#include "results.h"
#include <memory>

/*
Exports the current population in a FLAMEGPU suitable format
*/
void exportPopulation(std::shared_ptr<SpatialPartition> s, const ArgData &args, const char *path);

void exportAgents(std::shared_ptr<SpatialPartition>  s, const char *path);
void exportNullAgents(std::shared_ptr<SpatialPartition> s, const char *path, const DIMENSIONS_VEC *results);

void exportSteps(const int argc, char **argv, const Time_Step *, const NeighbourhoodStats *, const unsigned int &stepCount, const char*path);
#endif //__export_h__