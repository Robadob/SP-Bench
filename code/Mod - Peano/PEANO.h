#ifndef __Peano_h__
#define __Peano_h__
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
* 2D Peano Space Filling Encode/Decode
*/
#if defined(PEANO) && defined(_2D)

#if defined(_2D)
unsigned int **h_lookup = nullptr;
#elif defined(_3D)
unsigned int ***h_lookup = nullptr;
#endif
unsigned int *d_lookup;
__device__ __constant__ cudaTextureObject_t d_tex_lookup;
namespace{
    DIMENSIONS_IVEC peanoGridDims = DIMENSIONS_IVEC(0);
}

DIMENSIONS_IVEC peanoDecode(const unsigned int &d, const int power) {
    glm::ivec2 pos = glm::ivec2(0);
    int s;
    unsigned int t = d;
    int _col, _row;
    //Iterate depth
    bool flipX = false, flipY = false;
    for (s = power - 1; s >= 0; s--) {
        const unsigned int _nine = t / (unsigned int)pow(9, s);  //Get the cell level position
        //Subtract value we've processed
        t -= _nine * (unsigned int)pow(9, s);
        //Calculate position within cell
        _col = _nine / 3;            //Calc column
        _row = _nine % 3;            //Calc row
        _row = _col == 1 ? 2 - _row : _row; //Flip centre row order
        //Flip cell based on outer position
        _row = (flipX && _row != 1) ? _row ^ 2 : _row;
        _col = (flipY && _col != 1) ? _col ^ 2 : _col;
        ////Diagonally mirror cells which match the initial orientation
        ////This produces the wiggley ('swastika') variant
        //if (wiggley&&s == 0)//Optionally also remove s==0
        //{
        //    if (!(flipX^flipY))//(!(flipX&&flipY)//Alternate flip that does half the flips
        //    {
        //        std::swap(_row, _col);
        //    }
        //}
        //Stack transform for next iteration
        flipX = flipX ^ (_col == 1);
        flipY = flipY ^ (_row == 1);
        //Calculate contribution to final coordinate
        _col = _col * (unsigned int)pow(3, s);
        _row = _row * (unsigned int)pow(3, s);
        //Don't flip
        pos += glm::ivec2(_col, _row);
    }
    assert(t == 0);
    return pos;
}

//Build encode table, copy to GPU (const? tex?)
void initPeano(DIMENSIONS_IVEC gridDims)
{
    const unsigned int EXPONENT_BASE = 3;
    //This might be precomputed, but repeating causes no harm
    const unsigned int maxDim = (unsigned int)glm::compMax(gridDims);                                  //Get largest dimension
    const unsigned int maxDimExp = (unsigned int)ceil(log(maxDim) / log(EXPONENT_BASE));  //Find the ceiling of log_3 of maxDim an
    const unsigned int maxDim3 = (unsigned int)pow(EXPONENT_BASE, maxDimExp);                         //Raise 3 to the power of max dim to get our actual grid dims
    gridDims = DIMENSIONS_IVEC(maxDim3);
    peanoGridDims = gridDims;
    //Allocate Host 3D array to fill
#if defined(_2D)
    h_lookup = (unsigned int **)malloc(maxDim3*sizeof(unsigned int *));
#elif defined(_3D)
    h_lookup = (unsigned int ***)malloc(maxDim3*sizeof(unsigned int **));
#endif
    for (unsigned int i = 0; i < maxDim3; ++i)
    {
#if defined(_2D)
        h_lookup[i] = (unsigned int *)malloc(maxDim3*sizeof(unsigned int));
#elif defined(_3D)
        h_lookup[i] = (unsigned int **)malloc(maxDim3*sizeof(unsigned int *));
#endif
        for (unsigned int j = 0; j < maxDim3; ++j)
        {
#if defined(_2D)

            h_lookup[i][j] = UINT_MAX;//Uninitialised flag
#elif defined(_3D)
            h_lookup[i][j] = (unsigned int *)malloc(maxDim3*sizeof(unsigned int));
            for(unsigned int k = 0;k<maxDim3;++k)
                h_lookup[i][j][k]= UINT_MAX;//Uninitialised flag
#endif
        }
    }
    //Run Peano decode to fill lookup
    const unsigned int binCount = glm::compMul(gridDims);
    for (unsigned int i = 0; i < binCount; ++i)
    {
        DIMENSIONS_IVEC c = peanoDecode(i, maxDimExp);
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
                assert(h_lookup[i][j][k]!=UINT_MAX);//Uninitialised flag check
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
        for(unsigned int j = 0;j<maxDim;++j)
        {
            CUDA_CALL(cudaMemcpy(d_lookup + offset, &h_lookup[i][j][0], maxDim*sizeof(unsigned int), cudaMemcpyHostToDevice));
            offset += maxDim;
        }
#endif
    }
}
void freePeano()
{
    if(h_lookup)
    {
        for (unsigned int i = 0; i<(unsigned int)peanoGridDims.x; ++i)
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
        h_lookup=nullptr;
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
// Calculates a 32-bit Peano code
__device__ unsigned int d_peanoEncode(const DIMENSIONS_IVEC &pos)
{
    return tex1Dfetch<unsigned int>(d_tex_lookup, to1d(pos));
    //Is this necessary? (not if we're building a look up table)
}
__host__ unsigned int h_peanoEncode(const DIMENSIONS_IVEC &pos)
{
#if defined(_2D)
    return h_lookup[pos.x][pos.y];
#elif defined(_3D)
    return h_lookup[pos.x][pos.y][pos.z];
#endif
}
#endif //defined(PEANO) && defined(_2D)

/**
* 3D Peano Space Filling Encode/Decode
*/
#if defined(PEANO) && defined(_3D)

#error Peano 3D not yet implemented (are we ever doing 3d Peano)?

#endif //defined(PEANO) && defined(_3D)
#endif //__Peano_h__