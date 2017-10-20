#ifndef __Hilbert_h__
#define __Hilbert_h__
//https://link.springer.com/book/10.1007%2F978-1-4612-0871-6
//https://arxiv.org/abs/1109.2323

#if defined(_3D) && !defined(DIMENSIONS)
#define DIMENSIONS 3
#define DIMENSIONS_VEC glm::vec3
#define DIMENSIONS_IVEC glm::ivec3
#elif defined(_2D) && !defined(DIMENSIONS)
#define DIMENSIONS 2
#define DIMENSIONS_VEC glm::vec2
#define DIMENSIONS_IVEC glm::ivec2
#endif

/**
 * 2D Hilbert Space Filling Encode/Decode
 * Uses a texture cached look-up table
 */
#if defined(HILBERT) && defined(_2D)

#if defined(_2D)
unsigned int **h_lookup = nullptr;
#elif defined(_3D)
unsigned int ***h_lookup = nullptr;
#endif
unsigned int *d_lookup;
__device__ __constant__ cudaTextureObject_t d_tex_lookup;
namespace{
    DIMENSIONS_IVEC hilbertGridDims = DIMENSIONS_IVEC(0);
}

//rotate/flip a quadrant appropriately
__host__ void rot(const unsigned int &n, glm::ivec2 &pos, const unsigned int &rx, const unsigned int &ry) {
    if (ry == 0) {
        if (rx == 1) {
            pos.x = n - 1 - pos.x;
            pos.y = n - 1 - pos.y;
        }

        //Swap x and y
        int t = pos.x;
        pos.x = pos.y;
        pos.y = t;
    }
}

__host__ glm::ivec2 hilbertDecode(const unsigned int &d, const int &power) {
    glm::ivec2 rtn(0);
    int rx, ry, s, t = d;
    const unsigned int width = pow(2, power);
    for (s = 1; s<width; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, rtn, rx, ry);
        rtn.x += s * rx;
        rtn.y += s * ry;
        t /= 4;
    }
    return rtn;
}

//Build encode table, copy to GPU (const? tex?)
void initHilbert(DIMENSIONS_IVEC gridDims)
{
    const unsigned int EXPONENT_BASE = 2;
    //This might be precomputed, but repeating causes no harm
    const unsigned int maxDim = (unsigned int)glm::compMax(gridDims);                    //Get largest dimension
    const unsigned int maxDimExp = (unsigned int)ceil(log(maxDim) / log(EXPONENT_BASE)); //Find the ceiling of log_2 of maxDim an
    const unsigned int maxDim2 = (unsigned int)pow(EXPONENT_BASE, maxDimExp);            //Raise 2 to the power of max dim to get our actual grid dims
    gridDims = DIMENSIONS_IVEC(maxDim2);
    hilbertGridDims = gridDims;
    //Allocate Host 3D array to fill
#if defined(_2D)
    h_lookup = (unsigned int **)malloc(maxDim2*sizeof(unsigned int *));
#elif defined(_3D)
    h_lookup = (unsigned int ***)malloc(maxDim3*sizeof(unsigned int **));
#endif
    for (unsigned int i = 0; i < maxDim2; ++i)
    {
#if defined(_2D)
        h_lookup[i] = (unsigned int *)malloc(maxDim2*sizeof(unsigned int));
#elif defined(_3D)
        h_lookup[i] = (unsigned int **)malloc(maxDim3*sizeof(unsigned int *));
#endif
        for (unsigned int j = 0; j < maxDim2; ++j)
        {
#if defined(_2D)

            h_lookup[i][j] = UINT_MAX;//Uninitialised flag
#elif defined(_3D)
            h_lookup[i][j] = (unsigned int *)malloc(maxDim3*sizeof(unsigned int));
            for (unsigned int k = 0; k<maxDim3; ++k)
                h_lookup[i][j][k] = UINT_MAX;//Uninitialised flag
#endif
        }
    }
    //Run Peano decode to fill lookup
    const unsigned int binCount = glm::compMul(gridDims);
    for (unsigned int i = 0; i < binCount; ++i)
    {
        DIMENSIONS_IVEC c = hilbertDecode(i, maxDimExp);
#if defined(_2D)
        h_lookup[c.x][c.y] = i;
#elif defined(_3D)
        h_lookup[c.x][c.y][c.z] = i;
#endif
    }
    //Check all values have been set
    for (unsigned int i = 0; i < maxDim; ++i)
    {
        for (unsigned int j = 0; j < maxDim; ++j)
        {
            for (unsigned int k = 0; k < maxDim; ++k)
            {
#if defined(_2D)
                assert(h_lookup[i][j] != UINT_MAX);//Uninitialised flag check
#elif defined(_3D)
                assert(h_lookup[i][j][k] != UINT_MAX);//Uninitialised flag check
#endif
            }
        }
    }
    //Define CUDA texture memory
    //Define cuda array format
    //Allocate cuda array
    unsigned int size = binCount;
    CUDA_CALL(cudaMalloc(&d_lookup, size*sizeof(unsigned int)));
    //Define cuda resource from array
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_lookup;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = size*sizeof(unsigned int);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    CUDA_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    CUDA_CALL(cudaMemcpyToSymbol(d_tex_lookup, &texObj, sizeof(cudaTextureObject_t)));
    //Copy host look-up to device
    unsigned int offset = 0;
    for (unsigned int i = 0; i < maxDim; ++i)
    {
#if defined(_2D)
        CUDA_CALL(cudaMemcpy(d_lookup + offset, &h_lookup[i][0], maxDim*sizeof(unsigned int), cudaMemcpyHostToDevice));
        offset += maxDim;
#elif defined(_3D)
        for (unsigned int j = 0; j<maxDim; ++j)
        {
            CUDA_CALL(cudaMemcpy(d_lookup + offset, &h_lookup[i][j][0], maxDim*sizeof(unsigned int), cudaMemcpyHostToDevice));
            offset += maxDim;
        }
#endif
    }
}
void freeHilbert()
{
    if(h_lookup)
    {
        for (unsigned int i = 0; i<(unsigned int)hilbertGridDims.x; ++i)
        {
#if defined(_3D)
            for(unsigned int j = 0;j<(unsigned int)hilbertGridDims.y;++j)
            {
                free(h_lookup[i][j]);
            }
#endif
            free(h_lookup[i]);
        }
        free(h_lookup);
        h_lookup = nullptr;
    }
    cudaTextureObject_t texObj;
    CUDA_CALL(cudaMemcpyFromSymbol(&texObj, d_tex_lookup, sizeof(cudaTextureObject_t)));
    CUDA_CALL(cudaDestroyTextureObject(texObj));
    CUDA_CALL(cudaFree(d_lookup));
}
__device__ inline unsigned int to1d(const DIMENSIONS_IVEC &pos)
{
#if defined(_2D)
    return (d_gridDim.y * pos.x) + pos.y;
#elif defined(_3D)
    return (d_gridDim.y * d_gridDim.z) * pos.x
        + (d_gridDim.z * pos.y)
        + pos.z;
#endif
}

__device__ inline unsigned int d_hilbertEncode(const DIMENSIONS_IVEC &pos) {
    return tex1Dfetch<unsigned int>(d_tex_lookup, to1d(pos));
    //Is this necessary? (not if we're building a look up table)
}
__host__ unsigned int h_hilbertEncode(const DIMENSIONS_IVEC &pos)
{
#if defined(_2D)
    return h_lookup[pos.x][pos.y];
#elif defined(_3D)
    return h_lookup[pos.x][pos.y][pos.z];
#endif
}
#endif //defined(HILBERT) && defined(_2D)

/**
* 3D Hilbert Space Filling Encode/Decode
*/
#if defined(HILBERT) && defined(_3D)

#error Hilbert 3d not yet supported

#endif //defined(HILBERT) && defined(_3D)
#endif //__Hilbert_h__