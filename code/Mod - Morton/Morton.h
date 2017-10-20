#ifndef __Morton_h__
#define __Morton_h__
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
* 2D Morton Space Filling Encode/Decode
*/
#if defined(MORTON) && defined(_2D)

#if defined(_2D)
unsigned int **h_lookup = nullptr;
#elif defined(_3D)
unsigned int ***h_lookup = nullptr;
#endif
unsigned int *d_lookup;
__device__ __constant__ cudaTextureObject_t d_tex_lookup;
namespace{
    DIMENSIONS_IVEC mortonGridDims = DIMENSIONS_IVEC(0);
}

//https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
// Expands a 16-bit integer into 32 bits
// by inserting 1 zeros after each bit.
__host__ __device__ unsigned int expandBits16(unsigned int v)
{
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333;
    v = (v | (v << 1u)) & 0x55555555;
    return v;
}
DIMENSIONS_IVEC mortonDecode(const int &d) {
    //Convert a morton coded hash back into a grid square
    DIMENSIONS_IVEC ret = DIMENSIONS_IVEC(0);
    for (int i = 0; i < 32; i++)
    {
        int dim = (i % DIMENSIONS);//Dimension of shiftN (x,y)
        int shift = i / DIMENSIONS;//Which bit position within the return val
        ret[DIMENSIONS - 1 - dim] += (d&(1 << i)) >> (i - shift);
    }
    return ret;
}
//Build encode table, copy to GPU (const? tex?)
void initMorton(DIMENSIONS_IVEC gridDims)
{
    const unsigned int EXPONENT_BASE = 2;
    //This might be precomputed, but repeating causes no harm
    const unsigned int maxDim = (unsigned int)glm::compMax(gridDims);                    //Get largest dimension
    const unsigned int maxDimExp = (unsigned int)ceil(log(maxDim) / log(EXPONENT_BASE)); //Find the ceiling of log_2 of maxDim an
    const unsigned int maxDim2 = (unsigned int)pow(EXPONENT_BASE, maxDimExp);            //Raise 2 to the power of max dim to get our actual grid dims
    gridDims = DIMENSIONS_IVEC(maxDim2);
    mortonGridDims = gridDims;
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
        DIMENSIONS_IVEC c = mortonDecode(i);
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
void freeMorton()
{
    if(h_lookup)
    {
        for (unsigned int i = 0; i<(unsigned int)mortonGridDims.x; ++i)
        {
#if defined(_3D)
            for(unsigned int j = 0;j<(unsigned int)peanoGridDims.y;++j)
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
__device__ inline unsigned int d_mortonEncode(const DIMENSIONS_IVEC &pos)
{
    return tex1Dfetch<unsigned int>(d_tex_lookup, to1d(pos));
}
__host__ unsigned int h_mortonEncode(const DIMENSIONS_IVEC &pos)
{
#if defined(_2D)
    return h_lookup[pos.x][pos.y];
#elif defined(_3D)
    return h_lookup[pos.x][pos.y][pos.z];
#endif
}
#endif //defined(MORTON) && defined(_2D)

/**
* 3D Morton Space Filling Encode/Decode
*/
#if defined(MORTON) && defined(_3D)

#error Morton 3D lookup not yet implemented

#endif //defined(MORTON) && defined(_3D)
#endif //__Morton_h__