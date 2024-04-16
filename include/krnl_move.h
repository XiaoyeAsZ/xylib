#ifndef __KRNL_MOVE__
#define __KRNL_MOVE__

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
    void KrnlMove(DATA_TYPE *MatrixA,
                  const unsigned int OffsetA,
                  DATA_TYPE *MatrixB,
                  const unsigned int OffsetB,
                  const unsigned int Len);
}

#endif