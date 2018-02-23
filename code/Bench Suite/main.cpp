#define _CRT_SECURE_NO_WARNINGS
#define _2D
#include <cstdio>
#include <memory>
#include <string>
#include "results.h"
#include <ctime>
#ifdef _MSC_BUILD 
#include <windows.h>
#else
#include <filesystem>
#endif

CirclesParams reset(const CirclesParams &start, const CirclesParams &end);
NullParams reset(const NullParams &start, const NullParams &end);
DensityParams reset(const DensityParams &start, const DensityParams &end);
CirclesParams interpolateParams2D(const CirclesParams &start, const CirclesParams &end1, const CirclesParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
NullParams interpolateParams2D(const NullParams &start, const NullParams &end1, const NullParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
DensityParams interpolateParams2D(const DensityParams &start, const DensityParams &end1, const DensityParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps);
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps);
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps);
void execString(const char* executable, CirclesParams model, char **rtn);
void execString(const char* executable, NullParams model, char **rtn);
template<class T>
bool run(const T &start, const T &end, const unsigned int steps, const char *runName);
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast);
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const CirclesParams &modelArgs);
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const NullParams &modelArgs);
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const DensityParams &modelArgs);
//Collated
template<class T>
bool runCollated(const T &start, const T &end, const unsigned int steps, const char *runName);
template<class T>
bool runCollated2D(const T &start, const T &end1, const T &end2, const unsigned int steps1, const unsigned int steps2, const char *runName);
void logCollatedHeader(FILE *out, const CirclesParams &modelArgs);
void logCollatedHeader(FILE *out, const NullParams &modelArgs);
void logCollatedHeader(FILE *out, const DensityParams &modelArgs);
void log(FILE *out, const NeighbourhoodStats *nsFirst);
void log(FILE *out, const CirclesParams *modelArgs);
void log(FILE *out, const NullParams *modelArgs);
void log(FILE *out, const DensityParams *modelArgs);
void log(FILE *out, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void log(FILE *out, const unsigned int agentCount);
//const char *TEST_NAMES[] = { "Default", "Strips", "Modular", "Morton", "MortonCompute", "Hilbert", "Peano" };
//const char *TEST_EXECUTABLES[] = { "Release-Mod-Default.exe", "Release-Mod-Strips.exe", "Release-Mod-Modular.exe", "Release-Mod-Morton.exe", "Release-Mod-MortonCompute.exe", "Release-Mod-Hilbert.exe", "Release-Mod-Peano.exe" };

const char *TEST_NAMES[] = { "Default", "Strips", "Modular", "StripsModularHybrid" };
const char *TEST_EXECUTABLES[] = { "Release-Mod-Default.exe", "Release-Mod-Strips.exe", "Release-Mod-Modular.exe", "Release-Mod-ModularStrips.exe" };

//const char *TEST_NAMES[] = { "StripsModularHybrid" };
//const char *TEST_EXECUTABLES[] = { "Release-Mod-ModularStrips.exe" };

//const char *TEST_NAMES[] = { "Default", "Strips", "Modular", "Default-AOS", "Strips-AOS", "Modular-AOS" };
//const char *TEST_EXECUTABLES[] = { "Release-Mod-Default.exe", "Release-Mod-Strips.exe", "Release-Mod-Modular.exe", "Release-Mod-Default-AOS.exe", "Release-Mod-Strips-AOS.exe", "Release-Mod-Modular-AOS.exe" };
//const char *TEST_NAMES[] = { 
//    "Default", "Strips", "Modular", 
//    "Default_GLOBAL", "Strips_GLOBAL", "Modular_GLOBAL", 
//    "Default_LDG", "Strips_LDG", "Modular_LDG" 
//};
//const char *TEST_EXECUTABLES[] = { 
//    "Release-Mod-Default.exe", "Release-Mod-Strips.exe", "Release-Mod-Modular.exe", 
//    "Release-Mod-Default_GLOBAL_PBM.exe", "Release-Mod-Strips_GLOBAL_PBM.exe", "Release-Mod-Modular_GLOBAL_PBM.exe", 
//    "Release-Mod-Default_LDG_PBM.exe", "Release-Mod-Strips_LDG_PBM.exe", "Release-Mod-Modular_LDG_PBM.exe" };
//const char *TEST_NAMES[] = { "Default", "Modular" };
//const char *TEST_EXECUTABLES[] = { "Release-Mod-Default.exe", "Release-Mod-Modular.exe" };
const char *DIR_X64 = "..\\bin\\x64\\";
int main(int argc, char* argv[])
{
#if defined(_3D)
    {//Problem Scale - Circles
        //Init step count
        const int steps = 11;        
        //Init model arg start
        CirclesParams start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 5.0f;
        start.width = 300;
        start.seed = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 5.0f;
        end.width = 300;
        end.seed = 250;
        run(start, end, steps, "CirclesProblemScale");
    }
    {//Neighbourhood Scale - Circles
        const int steps = 16;        
        //Init model arg start
        CirclesParams  start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 1.0f;
        start.width = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 15.0f;
        end.width = 100;
        //Init step count
        run(start, end, steps, "CirclesNeighbourhoodScale");
    }
#elif defined(_2D)
        //{//Null - Problem Scale 200
        //    const int steps = 100;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 10000;
        //    start.iterations = 500;
        //    start.density = 24.0f; //~192 neighbours
        //    start.seed = 1;
        //    //Init model arg end
        //    NullParams end = start;
        //    end.agents = 300000;
        //    runCollated(start, end, steps, "NullProblemScale200_uniformrng");
        //}
        //{//Null - Problem Scale 200
        //    const int steps = 100;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 10000;
        //    start.iterations = 500;
        //    start.density = 24.0f; //~192 neighbours
        //    start.seed = 0;
        //    //Init model arg end
        //    NullParams end = start;
        //    end.agents = 300000;
        //    runCollated(start, end, steps, "NullProblemScale200_uniform");
        //}
        //{//Null - Neighbour scale 50k
        //    const int steps = 100;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 100000;
        //    start.iterations = 500;
        //    start.density = 2.0f;
        //    start.seed = 1;
        //    //Init model arg end
        //    NullParams end = start;
        //    end.density = 50.0f;
        //    runCollated(start, end, steps, "NullNeighbourhoodScale100k_uniformrng");
        //}
        //{//Null - Neighbour scale 50k
        //    const int steps = 100;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 100000;
        //    start.iterations = 500;
        //    start.density = 2.0f;
        //    start.seed = 0;
        //    //Init model arg end
        //    NullParams end = start;
        //    end.density = 50.0f;
        //    runCollated(start, end, steps, "NullNeighbourhoodScale100k_uniform");
        //}
/**
 * 2D PARAM SWEEP of 2D Models
 */
    //{//2D parameter sweep of problem scale, neighbourhood scale
    //    {//Null
    //        const int steps1 = 30;
    //        const int steps2 = 49;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 10000;
    //        start.iterations = 500;
    //        start.density = 2.0f;
    //        start.seed = 1;
    //        //Init model arg end
    //        NullParams end1 = start;
    //        end1.agents = 300000;
    //        NullParams end2 = start;
    //        end2.density = 50.0f;
    //        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Neighbourhood-NonUniform");
    //    }
    //    {//Null
    //        const int steps1 = 30;
    //        const int steps2 = 49;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 10000;
    //        start.iterations = 500;
    //        start.density = 2.0f;
    //        start.seed = 0;
    //        //Init model arg end
    //        NullParams end1 = start;
    //        end1.agents = 300000;
    //        NullParams end2 = start;
    //        end2.density = 50.0f;
    //        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Neighbourhood-Uniform");
    //    }
    //}
    //{//XL Version: 2D parameter sweep of problem scale, neighbourhood scale
        //{//Null
        //    const int steps1 = 300;
        //    const int steps2 = 250;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 10000;
        //    start.iterations = 100;
        //    start.density = 2.0f;
        //    start.seed = 1;
        //    //Init model arg end
        //    NullParams end1 = start;
        //    end1.agents = 300000;
        //    NullParams end2 = start;
        //    end2.density = 50.0f;
        //    runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Neighbourhood-UniformRNG");
        //}
        //{//Null
        //    const int steps1 = 300;
        //    const int steps2 = 250;
        //    //Init model arg start
        //    NullParams  start = {};
        //    start.agents = 10000;
        //    start.iterations = 100;
        //    start.density = 2.0f;
        //    start.seed = 0;
        //    //Init model arg end
        //    NullParams end1 = start;
        //    end1.agents = 300000;
        //    NullParams end2 = start;
        //    end2.density = 50.0f;
        //    runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Neighbourhood-Uniform");
        //}
    //}
    //{//Extension of best case uniform random 2D Sweep
    //    {//Null
    //        const int steps1 = 75;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 10000;
    //        start.iterations = 100;
    //        start.density = 2.385542f;
    //        start.seed = 1;
    //        //Init model arg end
    //        NullParams end1 = start;
    //        end1.agents = 2000000;
    //        runCollated(start, end1, steps1, "Null2D-22neighbours-uniformrng");
    //    }
    //}
    {//2D sweep of 2D physical model
        //{//Circles - 2D UniformRNG
        //    const int steps1 = 50;
        //    const int steps2 = 30;
        //    //Init model arg start
        //    CirclesParams  start = {};
        //    start.agents = 300000;
        //    start.iterations = 500;
        //    start.density = 2.0f;
        //    start.forceModifier = 0.01f;
        //    start.seed = 1;
        //    //Init model arg end
        //    CirclesParams end1 = start;
        //    end1.agents = 10000;
        //    CirclesParams end2 = start;
        //    end2.density = 20.0f;
        //    runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Circles2D_UniformRNG");
        //}
        //{//Circles - 2D Uniform
        //    const int steps1 = 50;
        //    const int steps2 = 30;
        //    //Init model arg start
        //    CirclesParams  start = {};
        //    start.agents = 300000;
        //    start.iterations = 500;
        //    start.density = 2.0f;
        //    start.forceModifier = 0.01f;
        //    start.seed = 0;
        //    //Init model arg end
        //    CirclesParams end1 = start;
        //    end1.agents = 10000;
        //    CirclesParams end2 = start;
        //    end2.density = 20.0f;
        //    runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Circles2D_Uniform");
        //}
    }
/**
* 2D PARAM SWEEP of 3D Models with Hybrid
*/
{//2D parameter sweep of problem scale, neighbourhood scale (v2)
    {//Null
        const int steps1 = 75;
        const int steps2 = 100;
        //Init model arg start
        NullParams  start = {};
        start.agents = 68585;// 10000;
        start.iterations = 100;
        start.density = 1.1f;
        start.seed = 1;
        //Init model arg end
        NullParams end1 = start;
        end1.agents = 300000;
        NullParams end2 = start;
        end2.density = 10.0f;
        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Null3D_UniformRNGv2");
    }
    {//Null
        const int steps1 = 100;
        const int steps2 = 100;
        //Init model arg start
        NullParams  start = {};
        start.agents = 10000;
        start.iterations = 100;
        start.density = 1.1f;
        start.seed = 0;
        //Init model arg end
        NullParams end1 = start;
        end1.agents = 300000;
        NullParams end2 = start;
        end2.density = 10.0f;
        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Null3D_Uniformv2");
    }
}
//{//2D parameter sweep of problem scale, neighbourhood scale
//    {//Null
//        const int steps1 = 30;
//        const int steps2 = 49;
//        //Init model arg start
//        NullParams  start = {};
//        start.agents = 10000;
//        start.iterations = 500;
//        start.density = 2.0f;
//        start.seed = 1;
//        //Init model arg end
//        NullParams end1 = start;
//        end1.agents = 300000;
//        NullParams end2 = start;
//        end2.density = 50.0f;
//        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Null3D_UniformRNG");
//    }
//    {//Null
//        const int steps1 = 30;
//        const int steps2 = 49;
//        //Init model arg start
//        NullParams  start = {};
//        start.agents = 10000;
//        start.iterations = 500;
//        start.density = 2.0f;
//        start.seed = 0;
//        //Init model arg end
//        NullParams end1 = start;
//        end1.agents = 300000;
//        NullParams end2 = start;
//        end2.density = 50.0f;
//        runCollated2D(start, end1, end2, steps1, steps2, "2DSweep-Null3D_Uniform");
//    }
//}
    //{//2D sweep of 3D physical model
    //    {//Circles
    //        const int steps1 = 50;
    //        const int steps2 = 30;
    //        //Init model arg start
    //        CirclesParams  start = {};
    //        start.agents = 10000;
    //        start.iterations = 500;
    //        start.density = 2.0f;
    //        start.forceModifier = 0.01f;
    //        start.seed = 1;
    //        //Init model arg end
    //        CirclesParams end1 = start;
    //        end1.agents = 300000;
    //        CirclesParams end2 = start;
    //        end2.density = 20.0f;
    //        runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Circles3D_UniformRNG");
    //    }
    //    {//Circles
    //        const int steps1 = 50;
    //        const int steps2 = 30;
    //        //Init model arg start
    //        CirclesParams  start = {};
    //        start.agents = 10000;
    //        start.iterations = 500;
    //        start.density = 2.0f;
    //        start.forceModifier = 0.01f;
    //        start.seed = 0;
    //        //Init model arg end
    //        CirclesParams end1 = start;
    //        end1.agents = 300000;
    //        CirclesParams end2 = start;
    //        end2.density = 20.0f;
    //        runCollated2D(start, end1, end2, steps1, steps2, "2DSweepXL-Circles3D_Uniform");
    //    }
    //}
/**
* 3D NON-UNIFORM WITH HYBRID
*/
    //{//Null - Problem scale
    //    const int steps = 97;
    //    //Init model arg start
    //    NullParams  start = {};
    //    start.agents = 100000;
    //    start.iterations = 1000;
    //    start.density = 2.0f;
    //    start.seed = 1;
    //    //Init model arg end
    //    NullParams end = start;
    //    end.density = 50.0f;
    //    runCollated(start, end, steps, "AOS Neighbourhood");
    //}
    //{//Null - Problem scale
    //    const int steps = 97;
    //    //Init model arg start
    //    NullParams  start = {};
    //    start.agents = 100000;
    //    start.iterations = 1000;
    //    start.density = 2.0f;
    //    start.seed = 0;
    //    //Init model arg end
    //    NullParams end = start;
    //    end.density = 50.0f;
    //    runCollated(start, end, steps, "AOS Neighbourhood");
    //}
/**
 * 3D Uniform with hybrid
 */
    //{//Null - Problem scale
    //    const int steps = 97;
    //    //Init model arg start
    //    NullParams  start = {};
    //    start.agents = 100000;
    //    start.iterations = 1000;
    //    start.density = 2.0f;
    //    start.seed = 0;
    //    //Init model arg end
    //    NullParams end = start;
    //    end.density = 50.0f;
    //    runCollated(start, end, steps, "3D-Neighbourhood-Uniform");
    //}
    //{//Neighbourhood scale 50k, 100k, 200k agents
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 50000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 0;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-Uniform");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 50000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 1;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-NonUniform");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 200000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 0;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-Uniform");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 200000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 1;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-NonUniform");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 50000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 2;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-NonUniform2");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 100000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 2;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-NonUniform2");
    //    }
    //    {//Null - Problem scale
    //        const int steps = 97;
    //        //Init model arg start
    //        NullParams  start = {};
    //        start.agents = 200000;
    //        start.iterations = 1000;
    //        start.density = 2.0f;
    //        start.seed = 2;
    //        //Init model arg end
    //        NullParams end = start;
    //        end.density = 50.0f;
    //        runCollated(start, end, steps, "3D-Neighbourhood-NonUniform2");
    //    }
    //}
//////////    //{//Density - ClusterCount//Re-assess, dumb slow around step 20
//////////    //    const int steps = 100;
//////////    //    //Init model arg start
//////////    //    DensityParams  start = {};
//////////    //    start.iterations = 1000;
//////////    //    start.envWidth = 1000;
//////////    //    start.agents = 10000;
//////////    //    start.seed = 1;
//////////    //    start.interactionRad = 1.0f;
//////////    //    start.clusterRad = 5.0f;
//////////    //    start.clusterCount = 1;
//////////
//////////    //    //Init model arg end
//////////    //    DensityParams end = start;
//////////    //    end.clusterCount = 200;
//////////
//////////    //    runCollated(start, end, steps, "ClusterCount 10k");
//////////    //}
////////////completed
//////////    {//Null - Neighbour scale 100k
//////////        const int steps = 100;
//////////        //Init model arg start
//////////        NullParams  start = {};
//////////        start.agents = 100000;
//////////        start.iterations = 1000;
//////////        start.density = 1.0f;
//////////        start.interactionRad = 0.5f;
//////////        start.seed = 0;
//////////        //Init model arg end
//////////        NullParams end = start;
//////////        end.interactionRad = 10.0f;
//////////        runCollated(start, end, steps, "NullNeighbourhoodScale100k");
//////////    }
////////////Collected
//////////{//Null - Problem scale
//////////const int steps = 101;
////////////Init model arg start
//////////NullParams  start = {};
//////////start.agents = 1000;
//////////start.iterations = 1000;
//////////start.density = 1.0f;
//////////start.interactionRad = 2.0f;
//////////start.seed = 0;
////////////Init model arg end
//////////NullParams end = start;
//////////end.agents = 100000;
//////////runCollated(start, end, steps, "NullProblemScaleHD");
//////////}
//////////completed
////////    {//Null - Neighbour scale 25k
////////        const int steps = 100;
////////        //Init model arg start
////////        NullParams  start = {};
////////        start.agents = 25000;
////////        start.iterations = 1000;
////////        start.density = 1.0f;
////////        start.interactionRad = 0.5f;
////////        start.seed = 0;
////////        //Init model arg end
////////        NullParams end = start;
////////        end.interactionRad = 10.0f;
////////        runCollated(start, end, steps, "NullNeighbourhoodScale25k");
////////    }
//////////completed
////////    {//Null - Neighbour scale 50k
////////        const int steps = 100;
////////        //Init model arg start
////////        NullParams  start = {};
////////        start.agents = 50000;
////////        start.iterations = 1000;
////////        start.density = 1.0f;
////////        start.interactionRad = 0.5f;
////////        start.seed = 0;
////////        //Init model arg end
////////        NullParams end = start;
////////        end.interactionRad = 10.0f;
////////        runCollated(start, end, steps, "NullNeighbourhoodScale50k");
////////    }
    //////{//Density - ClusterCount
    //////    const int steps = 100;
    //////    //Init model arg start
    //////    DensityParams  start = {};
    //////    start.iterations = 1000;
    //////    start.envWidth = 1000;
    //////    start.agents = 10000;
    //////    start.seed = 1;
    //////    start.interactionRad = 1.0f;
    //////    start.clusterRad = 1.0f;
    //////    start.clusterCount = 1;

    //////    //Init model arg end
    //////    DensityParams end = start;
    //////    end.clusterCount = 200;

    //////    runCollated(start, end, steps, "ClusterCount 10k");
    ////////////}
    //////{//Density - ClusterCount
    //////    const int steps = 100;
    //////    //Init model arg start
    //////    DensityParams  start = {};
    //////    start.iterations = 1000;
    //////    start.envWidth = 1000;
    //////    start.agents = 100000;
    //////    start.seed = 1;
    //////    start.interactionRad = 1.0f;
    //////    start.clusterRad = 1.0f;
    //////    start.clusterCount = 200;

    //////    //Init model arg end
    //////    DensityParams end = start;
    //////    end.clusterCount = 1;

    //////    runCollated(start, end, steps, "ClusterCount 100k");
    //////}
////////    {//Density - ClusterRad
////////        const int steps = 100;
////////        //Init model arg start
////////        DensityParams  start = {};
////////        start.iterations = 1000;
////////        start.envWidth = 1000;
////////        start.agents = 10000;
////////        start.seed = 1;
////////        start.interactionRad = 1.0f;
////////        start.clusterRad = 0.1f;
////////        start.clusterCount = 1;
////////
////////        //Init model arg end
////////        DensityParams end = start;
////////        end.clusterRad = 50.0f;
////////
////////        runCollated(start, end, steps, "ClusterRad25-10k 100");
////////    }
    ////////{//Density - ClusterRad
    ////////    const int steps = 100;
    ////////    //Init model arg start
    ////////    DensityParams  start = {};
    ////////    start.iterations = 1000;
    ////////    start.envWidth = 100;
    ////////    start.agents = 100000;
    ////////    start.seed = 1;
    ////////    start.interactionRad = 1.0f;
    ////////    start.clusterRad = 0.1f;
    ////////    start.clusterCount = 50;

    ////////    //Init model arg end
    ////////    DensityParams end = start;
    ////////    end.clusterRad = 50.0f;

    ////////    runCollated(start, end, steps, "ClusterRad50-100k");
    ////////}
////////    //Ones likely to TDR
////////    //{//Neighbourhood Scale - Circles
////////    //    const int steps = 97;
////////    //    //Init model arg start
////////    //    CirclesParams  start = {};
////////    //    start.iterations = 1000;
////////    //    start.density = 10.0f;
////////    //    start.interactionRad = 1.0f;
////////    //    start.width = 250;
////////    //    start.seed = 0;
////////    //    start.attractionForce = 0.2f;
////////    //    start.repulsionForce = 0.2f;
////////    //    //Init model arg end
////////    //    CirclesParams end = start;
////////    //    end.interactionRad = 15.0f;
////////    //    //Init step count
////////    //    run(start, end, steps, "CirclesNeighbourhoodScaleHD");
////////    //}
////////    {//Null - Neighbour scale
////////        const int steps = 100;
////////        //Init model arg start
////////        NullParams  start = {};
////////        start.agents = 10000;
////////        start.iterations = 1000;
////////        start.density = 10.0f;
////////        start.interactionRad = 1.0f;
////////        start.seed = 0;
////////        //Init model arg end
////////        NullParams end = start;
////////        end.interactionRad = 15.0f;
////////        runCollated(start, end, steps, "NULLNeighbourhoodScaleHD");
////////    }
#endif
	printf("\nComplete\n");
}
template<class T>
bool run(const T &start, const T &end, const unsigned int steps, const char *runName)
{
    printf("Running %s\n", runName);
    const float stepsM1 = (float)steps - 1.0f;
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    NeighbourhoodStats nsFirst, nsLast;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //For each mod
    for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
    {
        //Create log
        std::string logPath("./out/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(TEST_NAMES[j]);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
#ifndef _MSC_BUILD
        if (!exists(logPath)) { // Check if folder exists
            create_directory(logPath.c_str()); // create folder
        }
#endif
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        FILE *log = fopen(logPath.c_str(), "w");
        if (!log)
        {
            fprintf(stderr, "Benchmark '%s' '%s' failed to create log file '%s'\n", runName, TEST_NAMES[j], logPath.c_str());
            return false;
        }
        //Create log header
        logHeader(log, start);
        //For each benchmark
        for (unsigned int i = 0; i < steps; i++)
        {
            printf("\rExecuting run %s %i/%i      ", TEST_NAMES[j], i, (int)stepsM1);
            //Interpolate model
            modelArgs = interpolateParams(start, end, i, steps);
            //Clear output structures
            memset(&modelParamsOut, 0, sizeof(T));
            memset(&stepRes, 0, sizeof(Time_Step_dbl));
            memset(&initRes, 0, sizeof(Time_Init));
            agentCount = 0;
            totalTime = 0;
            //executeBenchmark
            if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast))
            {
                fprintf(stderr, "\rBenchmark '%s' '%s' execution failed on stage %d/%d, exiting early.\n", runName, TEST_NAMES[j], i + 1, steps);
                return false;
            }
            //logResult
            logResult(log, &modelParamsOut, agentCount, &initRes, &stepRes, totalTime, &nsFirst, &nsLast);
        }
        printf("\n");
        //Close log
        fclose(log);
    }
    return true;
}

template<class T>
bool runCollated(const T &start, const T &end, const unsigned int steps, const char *runName)
{
    printf("Running %s\n", runName);
    const float stepsM1 = (float)steps - 1.0f;
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    NeighbourhoodStats nsFirst, nsLast;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //Create log
    FILE *log_F = nullptr;
    {
        std::string logPath("./out/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
#ifndef _MSC_BUILD
        if (!exists(logPath)) { // Check if folder exists
            create_directory(logPath.c_str()); // create folder
        }
#endif
        logPath = logPath.append("collated");
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        log_F = fopen(logPath.c_str(), "w");
        if (!log_F)
        {
            fprintf(stderr, "Collated benchmark '%s' failed to create log file '%s'\n", runName, logPath.c_str());
            return false;
        }
    }
    //Create log header
    logCollatedHeader(log_F, start);
    //For each step
    for (unsigned int i = 0; i < steps; i++)
    {
        //Interpolate model
        modelArgs = interpolateParams(start, end, i, steps);
        //Clear output structures
        memset(&modelParamsOut, 0, sizeof(T));
        memset(&stepRes, 0, sizeof(Time_Step_dbl));
        memset(&initRes, 0, sizeof(Time_Init));
        //For each mod
        for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
        {
            printf("\rExecuting run %s %i/%i %i/%llu      ", runName, i + 1, (int)steps, j + 1, sizeof(TEST_EXECUTABLES) / sizeof(char*));
            agentCount = 0;
            totalTime = 0;
            //executeBenchmark
            if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast))
            {
                fprintf(stderr, "\rBenchmark '%s' '%s' execution failed on stage %d/%d, exiting early.\n", runName, TEST_NAMES[j], i + 1, steps);
                return false;
            }
            //logResult
            log(log_F, &initRes, &stepRes, totalTime);
        }
        //AgentCount
        log(log_F, agentCount);
        //Neighbourhood stats
        log(log_F, &nsFirst);
        log(log_F, &nsLast);
        //Model Args
        log(log_F, &modelArgs);
        fputs("\n", log_F);
        fflush(log_F);
    }
    //Close log
    fclose(log_F);
    //Print confirmation to console
    printf("\rCompleted run %s %i/%i %llu/%llu         \n", runName, (int)steps, (int)steps, sizeof(TEST_EXECUTABLES) / sizeof(char*), sizeof(TEST_EXECUTABLES) / sizeof(char*));
    return true;
}
template<class T>
bool runCollated2D(const T &start, const T &end1, const T &end2, const unsigned int steps1, const unsigned int steps2, const char *runName)
{
    printf("Running %s\n", runName);
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    NeighbourhoodStats nsFirst, nsLast;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //Create log
    FILE *log_F = nullptr;
    {
        std::string logPath("./out/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
#ifndef _MSC_BUILD
        if (!exists(logPath)) { // Check if folder exists
            create_directory(logPath.c_str()); // create folder
        }
#endif
        logPath = logPath.append("collated");
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        log_F = fopen(logPath.c_str(), "w");
        if (!log_F)
        {
            fprintf(stderr, "Collated benchmark '%s' failed to create log file '%s'\n", runName, logPath.c_str());
            return false;
        }
    }
    //Create log header
    logCollatedHeader(log_F, start);
    //For each step
    for (unsigned int w = 0; w < steps1; w++)
    {
        //Interpolate model
        for (unsigned int v = 0; v < steps2; v++)
        {
            //Interpolate model
            modelArgs = interpolateParams2D(start, end1, end2, w, steps1, v, steps2);
            //Clear output structures
            memset(&modelParamsOut, 0, sizeof(T));
            memset(&stepRes, 0, sizeof(Time_Step_dbl));
            memset(&initRes, 0, sizeof(Time_Init));
            //For each mod
            for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
            {
                printf("\rExecuting run %s %d:%d/%d:%d %i/%llu      ", runName, w + 1, v + 1, steps1, steps2, j + 1, sizeof(TEST_EXECUTABLES) / sizeof(char*));
                agentCount = 0;
                totalTime = 0;
                //executeBenchmark
                if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast))
                {
                    fprintf(stderr, "\rBenchmark '%s' '%s' execution failed on stage %d:%d/%d:%d, exiting early.\n", runName, TEST_NAMES[j], w + 1, v + 1, steps1, steps2);
                    return false;
                }
                //logResult
                log(log_F, &initRes, &stepRes, totalTime);
            }
            //AgentCount
            log(log_F, agentCount);
            //Neighbourhood stats
            log(log_F, &nsFirst);
            log(log_F, &nsLast);
            //Model Args
            log(log_F, &modelArgs);
            fputs("\n", log_F);
            fflush(log_F);
        }
    }
    //Close log
    fclose(log_F);
    //Print confirmation to console
    printf("\rCompleted run %s %i/%i %llu/%llu         \n", runName, (int)steps1, (int)steps2, sizeof(TEST_EXECUTABLES) / sizeof(char*), sizeof(TEST_EXECUTABLES) / sizeof(char*));
    return true;
}
CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    CirclesParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    //modelArgs.width = start.width + (int)((step / stepsM1)*((int)end.width - (int)start.width));
    modelArgs.agents = start.agents + (int)((step / stepsM1)*((int)end.agents - (int)start.agents));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    //modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    //modelArgs.attractionForce = start.attractionForce + ((step / stepsM1)*(end.attractionForce - start.attractionForce));
    //modelArgs.repulsionForce = start.repulsionForce + ((step / stepsM1)*(end.repulsionForce - start.repulsionForce));
    modelArgs.forceModifier = start.forceModifier + ((step / stepsM1)*(end.forceModifier - start.forceModifier));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
CirclesParams reset(const CirclesParams &start, const CirclesParams &end)
{
    CirclesParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    //end1Reset.width = end.width - start.width;
    end1Reset.agents = end.agents - start.agents;
    end1Reset.density = end.density - start.density;
    //end1Reset.interactionRad = end.interactionRad - start.interactionRad;
    //end1Reset.attractionForce = end.attractionForce - start.attractionForce;
    //end1Reset.repulsionForce = end.repulsionForce - start.repulsionForce;
    end1Reset.forceModifier = end.forceModifier - start.forceModifier;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
NullParams reset(const NullParams &start, const NullParams &end)
{
    NullParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    end1Reset.agents = end.agents - start.agents;
    end1Reset.density = end.density - start.density;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
DensityParams reset(const DensityParams &start, const DensityParams &end)
{
    DensityParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    end1Reset.agentsPerCluster = end.agentsPerCluster - start.agentsPerCluster;
    end1Reset.envWidth = end.envWidth - start.envWidth;
    end1Reset.interactionRad = end.interactionRad - start.interactionRad;
    end1Reset.clusterCount = end.clusterCount - start.clusterCount;
    end1Reset.clusterRad = end.clusterRad - start.clusterRad;
    end1Reset.uniformDensity = end.uniformDensity - start.uniformDensity;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
CirclesParams interpolateParams2D(const CirclesParams &start, const CirclesParams &end1, const CirclesParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    CirclesParams end1Lerp = interpolateParams(CirclesParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    CirclesParams end2Lerp = interpolateParams(CirclesParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    CirclesParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    //modelArgs.width = start.width + end1Lerp.width + end2Lerp.width;
    modelArgs.agents = start.agents + end1Lerp.agents + end2Lerp.agents;
    modelArgs.density = start.density + end1Lerp.density + end2Lerp.density;
    //modelArgs.interactionRad = start.interactionRad + end1Lerp.interactionRad + end2Lerp.interactionRad;
    //modelArgs.attractionForce = start.attractionForce + end1Lerp.attractionForce + end2Lerp.attractionForce;
    //modelArgs.repulsionForce = start.repulsionForce + end1Lerp.repulsionForce + end2Lerp.repulsionForce;
    modelArgs.forceModifier = start.forceModifier + end1Lerp.forceModifier + end2Lerp.forceModifier;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
NullParams interpolateParams2D(const NullParams &start, const NullParams &end1, const NullParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    NullParams end1Lerp = interpolateParams(NullParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    NullParams end2Lerp = interpolateParams(NullParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    NullParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    modelArgs.agents = start.agents + end1Lerp.agents + end2Lerp.agents;
    modelArgs.density = start.density + end1Lerp.density + end2Lerp.density;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
DensityParams interpolateParams2D(const DensityParams &start, const DensityParams &end1, const DensityParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    DensityParams end1Lerp = interpolateParams(DensityParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    DensityParams end2Lerp = interpolateParams(DensityParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    DensityParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    modelArgs.agentsPerCluster = start.agentsPerCluster + end1Lerp.agentsPerCluster + end2Lerp.agentsPerCluster;
    modelArgs.envWidth = start.envWidth + end1Lerp.envWidth + end2Lerp.envWidth;
    modelArgs.interactionRad = start.interactionRad + end1Lerp.interactionRad + end2Lerp.interactionRad;
    modelArgs.clusterCount = start.clusterCount + end1Lerp.clusterCount + end2Lerp.clusterCount;
    modelArgs.clusterRad = start.clusterRad + end1Lerp.clusterRad + end2Lerp.clusterRad;
    modelArgs.uniformDensity = start.uniformDensity + end1Lerp.uniformDensity + end2Lerp.uniformDensity;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    NullParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agents = start.agents + (unsigned int)((step / stepsM1)*((unsigned int)end.agents - (unsigned int)start.agents));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    DensityParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agentsPerCluster = start.agentsPerCluster + (unsigned int)((step / stepsM1)*((unsigned int)end.agentsPerCluster - (unsigned int)start.agentsPerCluster));
    modelArgs.envWidth = start.envWidth + ((step / stepsM1)*(end.envWidth - start.envWidth));
    modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    modelArgs.clusterCount = start.clusterCount + (unsigned int)((step / stepsM1)*((int)end.clusterCount - (int)start.clusterCount));
    modelArgs.clusterRad = start.clusterRad + ((step / stepsM1)*(end.clusterRad - start.clusterRad));
    modelArgs.uniformDensity = start.uniformDensity + ((step / stepsM1)*(end.uniformDensity - start.uniformDensity));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast)
{
    char *command;
    bool rtn = true;
    execString(executable, modelArgs, &command);
    std::shared_ptr<FILE> pipe(_popen(command, "rb"), _pclose);
    if (!pipe.get()) rtn = false;
    if (rtn)
    {
        if (fread(modelparamOut, sizeof(T), 1, pipe.get()) != 1)
        {
            rtn = false; printf("\nReading model params failed.\n");
        }
        if (fread(agentCount, sizeof(unsigned int), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading agent count failed.\n");
        }
        if (fread(initRes, sizeof(Time_Init), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading init timings failed.\n");
        }
        if (fread(stepRes, sizeof(Time_Step_dbl), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading step timings failed.\n");
        }
        if (fread(totalTime, sizeof(float), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading total time failed.\n");
        }
        if (fread(nsFirst, sizeof(NeighbourhoodStats), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading first NeighbourhoodStats failed.\n");
        }
        if (fread(nsLast, sizeof(NeighbourhoodStats), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading last NeighbourhoodStats failed.\n");
        }
    }
    if (!rtn)
    {
        printf("Exec: %s\n", command);
    }
    return rtn;
}

void execString(const char* executable, CirclesParams modelArgs, char **rtn)
{
	std::string buffer("\"");
	buffer = buffer.append(DIR_X64);
	buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
	buffer = buffer.append("-pipe");
	buffer = buffer.append(" ");
	buffer = buffer.append("-device");
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(0));
	buffer = buffer.append(" ");
	buffer = buffer.append("-circles");
	buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.width));
    buffer = buffer.append(std::to_string(modelArgs.agents));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.density));
	buffer = buffer.append(" ");
	//buffer = buffer.append(std::to_string(modelArgs.interactionRad));
	//buffer = buffer.append(" ");
	//buffer = buffer.append(std::to_string(modelArgs.attractionForce));
	//buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.repulsionForce));
    buffer = buffer.append(std::to_string(modelArgs.forceModifier));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.iterations));
	if (modelArgs.seed!=12)
	{

		buffer = buffer.append(" ");
		buffer = buffer.append(" -seed");
		buffer = buffer.append(" ");
		buffer = buffer.append(std::to_string(modelArgs.seed));
	}
	const char *src = buffer.c_str();
	*rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
	memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
void execString(const char* executable, NullParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(DIR_X64);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(0));
    buffer = buffer.append(" ");
    buffer = buffer.append("-null");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agents));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.density));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append(" -seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
void execString(const char* executable, DensityParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(DIR_X64);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(0));
    buffer = buffer.append(" ");
    buffer = buffer.append("-density");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agentsPerCluster));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.envWidth));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterCount));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.interactionRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.uniformDensity));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append(" -seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
/**
 * Logging
 */
void logHeader(FILE *out, const CirclesParams &modelArgs)
{
	fputs("model", out);
    fputs(",,,,,", out);
    //fputs(",,,,,,,", out);
	fputs("init (s)", out);
	fputs(",,,,,,", out);
	fputs("step avg (s)", out);
	fputs(",,,", out);
	fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
	fputs("\n", out);
	//ModelArgs
    //fputs("width", out);
    fputs("agents", out);
	fputs(",", out);
	fputs("density", out);
	fputs(",", out);
	//fputs("interactionRad", out);
	//fputs(",", out);
	//fputs("attractionForce", out);
	//fputs(",", out);
    //fputs("repulsionForce", out);
    fputs("forceModifier", out);
	fputs(",", out);
	fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
	fputs("agents-out", out);
    fputs(",", out);
	//Init
	fputs("overall", out);
	fputs(",", out);
	fputs("initCurand", out);
	fputs(",", out);
	fputs("kernel", out);
	fputs(",", out);
	fputs("pbm", out);
	fputs(",", out);
	fputs("freeCurand", out);
	fputs(",", out);
	//Step avg
	fputs("overall", out);
	fputs(",", out);
	fputs("kernel", out);
	fputs(",", out);
	fputs("texture", out);
	fputs(",", out);
	//Total
	fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
	//ln
	fputs("\n", out);
}
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
	//fprintf(out, "%u,%f,%f,%f,%f,%llu,%llu,",
	//	modelArgs->width,
	//	modelArgs->density,
	//	modelArgs->interactionRad,
	//	modelArgs->attractionForce,
	//	modelArgs->repulsionForce,
	//	modelArgs->iterations,
 //       modelArgs->seed
	//	);
    fprintf(out, "%u,%f,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->forceModifier,
        modelArgs->iterations,
        modelArgs->seed
        );
	//Agent count
	fprintf(out, "%i,",
		agentCount
		);
	//Init
	fprintf(out, "%f,%f,%f,%f,%f,",
		initRes->overall/1000,
		initRes->initCurand / 1000,
		initRes->kernel / 1000,
		initRes->pbm / 1000,
		initRes->freeCurand / 1000
		);
	//Step avg
	fprintf(out, "%f,%f,%f,",
		stepRes->overall / 1000,
		stepRes->kernel / 1000,
		stepRes->texture / 1000
		);
	//Total
	fprintf(out, "%f,",
		totalTime /1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
        );
	//ln
	fputs("\n", out);
	fflush(out);
}
void logHeader(FILE *out, const NullParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("agents-out", out);
    fputs(",", out);
    //Init
    fputs("overall", out);
    fputs(",", out);
    fputs("initCurand", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("pbm", out);
    fputs(",", out);
    fputs("freeCurand", out);
    fputs(",", out);
    //Step avg
    fputs("overall", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("texture", out);
    fputs(",", out);
    //Total
    fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
    fprintf(out, "%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->iterations,
        modelArgs->seed
        );
    //Agent count
    fprintf(out, "%i,",
        agentCount
        );
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
        );
    //ln
    fputs("\n", out);
    fflush(out);
}
void logHeader(FILE *out, const DensityParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents per cluster", out);
    fputs(",", out);
    fputs("envWidth", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
    fputs(",", out);
    fputs("uniformDensity", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("agents-out", out);
    fputs(",", out);
    //Init
    fputs("overall", out);
    fputs(",", out);
    fputs("initCurand", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("pbm", out);
    fputs(",", out);
    fputs("freeCurand", out);
    fputs(",", out);
    //Step avg
    fputs("overall", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("texture", out);
    fputs(",", out);
    //Total
    fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
    fprintf(out, "%u,%f,%f,%u,%f,%f,%llu,%llu,",
        modelArgs->agentsPerCluster,
        modelArgs->envWidth,
        modelArgs->interactionRad,
        modelArgs->clusterCount,
        modelArgs->clusterRad,
        modelArgs->uniformDensity,
        modelArgs->iterations,
        modelArgs->seed
        );
    //Agent count
    fprintf(out, "%i,",
        agentCount
        );
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
        );
    //ln
    fputs("\n", out);
    fflush(out);
}

//Collated versions
void logCollatedHeader(FILE *out, const CirclesParams &modelArgs)
{
    //Row 1
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", TEST_NAMES[i]);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,", out);//Neighbourhood stats
    //fputs(",,,,,,", out);//Model Args
    fputs(",,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
    fputs("Model", out);
    //fputs(",,,,,,,", out);
    fputs(",,,,,", out);
    fputs("\n", out);
    //Row 3
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    //ModelArgs
    //fputs("width", out);
    fputs("agents", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    //fputs("interactionRad", out);
    //fputs(",", out);
    //fputs("attractionForce", out);
    //fputs(",", out);
    //fputs("repulsionForce", out);
    fputs("forceModifier", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}
void logCollatedHeader(FILE *out, const NullParams &modelArgs)
{
    //Row 1
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", TEST_NAMES[i]);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,", out);//Neighbourhood stats
    fputs(",,,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
    fputs("Model", out);
    fputs(",,,,,", out);
    fputs("\n", out);
    //Row 3
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fflush(out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}
void logCollatedHeader(FILE *out, const DensityParams &modelArgs)
{
    //Row 1
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", TEST_NAMES[i]);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,", out);//Neighbourhood stats
    fputs(",,,,,,,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,", out);
    fputs("Model", out);
    fputs(",,,,,,,,,", out);
    fputs("\n", out);
    //Row 3
    for (unsigned int i = 0; i < sizeof(TEST_NAMES) / sizeof(char*); ++i)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fflush(out);
    //ModelArgs
    fputs("agentsPerCluster", out);
    fputs(",", out);
    fputs("envWidth", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
    fputs(",", out);
    fputs("uniformDensity", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}
void log(FILE *out, const NeighbourhoodStats *nsFirst)
{
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average
        );
}
void log(FILE *out, const CirclesParams *modelArgs)
{
    //ModelArgs
    //fprintf(out, "%u,%f,%f,%f,%f,%llu,%llu,",
    //    modelArgs->width,
    //    modelArgs->density,
    //    modelArgs->interactionRad,
    //    modelArgs->attractionForce,
    //    modelArgs->repulsionForce,
    //    modelArgs->iterations,
    //    modelArgs->seed
    //    );
    fprintf(out, "%u,%f,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->forceModifier,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const NullParams *modelArgs)
{	
    //ModelArgs
    fprintf(out, "%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const DensityParams *modelArgs)
{	
    //ModelArgs
    fprintf(out, "%u,%f,%f,%u,%f,%f,%llu,%llu,",
        modelArgs->agentsPerCluster,
        modelArgs->envWidth,
        modelArgs->interactionRad,
        modelArgs->clusterCount,
        modelArgs->clusterRad,
        modelArgs->uniformDensity,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
}
void log(FILE *out, const unsigned int agentCount)
{
    fprintf(out, "%u,", agentCount);
}