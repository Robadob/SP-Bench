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

#endif //__export_h__