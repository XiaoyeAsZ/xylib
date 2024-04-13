#ifndef __KRNL_DOTMAT__
#define __KRNL_DOTMAT__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE float
#define DATA_WIDTH 32
#define DATA_PACK_NUM 16
#define MAX_MATRIX_SIZE 128 * 128

extern "C"
{
    void KrnlDotMat(DATA_TYPE *MatrixA,
                  const unsigned int OffsetA,
                  DATA_TYPE *MatrixB,
                  const unsigned int OffsetB,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  DATA_TYPE *MatrixRes,
                  const unsigned int OffsetRes);
}

#endif