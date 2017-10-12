#ifndef __Morton_h__
#define __Morton_h__
//https://link.springer.com/book/10.1007%2F978-1-4612-0871-6
//https://arxiv.org/abs/1109.2323

/**
* 2D Morton Space Filling Encode/Decode
*/
#if defined(MORTON) && defined(_2D)
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
// Calculates a 32-bit Morton code
__host__ __device__ unsigned int mortonEncode(const glm::ivec3 &pos)
{
    //Pos should be clamped to 0<=x<65536
#ifdef _DEBUG
    assert(pos.x >= 0);
    assert(pos.x < 65536);
    assert(pos.y >= 0);
    assert(pos.y < 65536);
    assert(pos.z >= 0);
    assert(pos.z < 65536);
#endif
    return expandBits16((unsigned int)pos.x) | (expandBits16((unsigned int)pos.y)<<1);
}
__host__ __device__ DIMENSIONS_IVEC mortonDecode(const int &d) {
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
#endif //defined(MORTON) && defined(_2D)

/**
* 3D Morton Space Filling Encode/Decode
*/
#if defined(MORTON) && defined(_3D)
//https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ unsigned int expandBits10(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
// Calculates a 30-bit Morton code for the
__host__ __device__ unsigned int mortonEncode(const glm::ivec3 &pos)
{
    //Pos should be clamped to 0<=x<1024

#ifdef _DEBUG
    assert(pos.x >= 0);
    assert(pos.x < 1024);
    assert(pos.y >= 0);
    assert(pos.y < 1024);
    assert(pos.z >= 0);
    assert(pos.z < 1024);
#endif
    unsigned int xx = expandBits10((unsigned int)pos.x);
    unsigned int yy = expandBits10((unsigned int)pos.y);
    unsigned int zz = expandBits10((unsigned int)pos.z);
    return xx * 4 + yy * 2 + zz;
}
__host__ __device__ DIMENSIONS_IVEC mortonDecode(const int &d) {
    //Convert a morton coded hash back into a grid square
    DIMENSIONS_IVEC ret = DIMENSIONS_IVEC(0);
    for (int i = 0; i < 30; i++)
    {
        int dim = (i % DIMENSIONS);//Dimension of shiftN (x,y,z)
        int shift = i / DIMENSIONS;//Which bit position within the return val
        ret[DIMENSIONS - 1 - dim] += (d&(1 << i)) >> (i - shift);
    }
    return ret;
}
#endif //defined(MORTON) && defined(_3D)
#endif //__Morton_h__