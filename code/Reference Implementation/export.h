#include "Neighbourhood.cuh"
#include "results.h"

/*
Exports the current population in a FLAMEGPU suitable format
*/
void exportPopulation(SpatialPartition* s, ModelParams *model, char *path);