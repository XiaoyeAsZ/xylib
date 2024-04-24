#ifndef __KRNL_ADDMAT__
#define __KRNL_ADDMAT__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<8>
#define DATA_WIDTH 8
#define DATA_PACK_NUM 32
#define MAX_MATRIX_SIZE 128 * 128

#define MAX_TOKEN_LEN 128

extern "C"
{
    void KrnlAddMat(DATA_TYPE *MatrixA,
                    const unsigned int OffsetA,
                    DATA_TYPE *MatrixB,
                    const unsigned int OffsetB,
                    const unsigned int DimM,
                    const unsigned int DimN,
                    DATA_TYPE *MatrixRes,
                    const unsigned int OffsetRes);
}

#endif