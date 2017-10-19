#ifndef __Hilbert_h__
#define __Hilbert_h__
//https://link.springer.com/book/10.1007%2F978-1-4612-0871-6
//https://arxiv.org/abs/1109.2323

/**
 * 2D Hilbert Space Filling Encode/Decode
 */
#if defined(HILBERT) && defined(_2D)
//rotate/flip a quadrant appropriately
__host__ __device__ void rot(const unsigned int &n, glm::ivec3 &pos, const unsigned int &rx, const unsigned int &ry) {
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

__host__ __device__ glm::ivec2 hilbertDecode(const int &n, const int &d) {
    glm::ivec2 rtn;
    int rx, ry, s, t = d;
    pos->x = pos->y = 0;
    unsigned int width = pow(2, n);
    for (s = 1; s<width; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        rtn.x += s * rx;
        rtn.y += s * ry;
        t /= 4;
    }
    return rtn;
}

__host__ __device__ unsigned int hilbertEncode(const unsigned int &n, const glm::uvec2 &pos) {
    unsigned int rx, ry, s, d = 0;
    for (s = n / 2; s>0; s /= 2) {
        rx = (pos.x & s) > 0;
        ry = (pos.y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}
#endif //defined(HILBERT) && defined(_2D)

/**
* 3D Hilbert Space Filling Encode/Decode
*/
#if defined(HILBERT) && defined(_3D)
#error Hilbert 3d not yet supported
//rotate/flip a quadrant appropriately
__host__ __device__ void rot(const unsigned int &n, glm::ivec3 &pos, const unsigned int &rx, const unsigned int &ry, , const unsigned int &rz) {
    if (ry == 0) {
        if (rx == 1) {
            pos->x = n - 1 - pos.x;
            pos->y = n - 1 - pos.y;
        }

        //Swap x and y
        int t = pos.x;
        pos.x = pos.y;
        pos.y = t;
    }
}

__host__ __device__ glm::ivec2 hilbertDecode(const int &n, const int &d) {
    glm::ivec2 rtn;
    int rx, ry, s, t = d;
    pos->x = pos->y = 0;
    for (s = 1; s<n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        rtn.x += s * rx;
        rtn.y += s * ry;
        t /= 4;
    }
    return rtn;
}

__host__ __device__ unsigned int hilbertEncode(const unsigned int &n, const glm::uvec2 &pos) {
    unsigned int rx, ry, s, d = 0;
    for (s = n / 2; s>0; s /= 2) {
        rx = (pos.x & s) > 0;
        ry = (pos.y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}

#endif //defined(HILBERT) && defined(_3D)
#endif //__Hilbert_h__