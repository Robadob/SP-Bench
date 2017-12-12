#define MOD_NAME "ModularStrips3D Binning"

#define MODULAR_STRIPS_3D
//#define NO_SYNC //Disables the bin switch sync, testing showed this harmed performance in the basic modular technique
//Build
#include "BuildAll.cuh"

#if defined(_2D) ||!defined(_3D)
#error ModularStrips3D only works in 3D
#endif