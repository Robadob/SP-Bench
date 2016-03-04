#include "Circles.cuh"

template <class T>
Circles<T>::Circles(
    const unsigned int width = 250,
    const float density = 1.0,
    const float interactionRad = 10.0,
    const float attract = 5.0,
    const float repulse = 5.0
    )
    : width(width)
    , density(density)
    , interactionRad(interactionRad)
    , attract(attract)
    , repulse(repulse)
    , spatialPartition(new T())
{
    //Copy relevant parameters to constants

    //Init 
    spatialPartition = ;
}