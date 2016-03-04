#ifndef __Circles_cuh__
#define __Circles_cuh__
#include "Neighbourhood.cuh"

template <class T>
class Circles
{
public:
    Circles(
        const unsigned int width = 250,
        const float density = 1.0,
        const float interactionRad = 10.0,
        const float attract = 5.0,
        const float repulse = 5.0
        );
private:
    T *spatialPartition;
    const unsigned int width;
    const float density;
    const float interactionRad;
    const float attract;
    const float repulse;
};

#endif