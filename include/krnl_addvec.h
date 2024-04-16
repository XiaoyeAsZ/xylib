#ifndef __KRNL_ADDVEC__
#define __KRNL_ADDVEC__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE float
#define DATA_WIDTH 32
#define DATA_PACK_NUM 8
#define MAX_MATRIX_SIZE 128 * 128

#define MAX_TOKEN_LEN 128

extern "C"
{
    void KrnlAddVec(DATA_TYPE *VecA,
                    const unsigned int OffsetA,
                    DATA_TYPE *VecB,
                    const unsigned int OffsetB,
                    const unsigned int DimM,
                    const unsigned int DimN,
                    DATA_TYPE *VecRes,
                    const unsigned int OffsetVec);
}

#endif