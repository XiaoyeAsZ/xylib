#ifndef __KRNL_TRANSPOSE__
#define __KRNL_TRANSPOSE__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<8>
#define DATA_WIDTH 8
#define DATA_PACK_NUM 32
#define MAX_MATRIX_SIZE 32 * 1024

typedef ap_int<8 * 32> PACK_INT8_32;

extern "C"
{
    void KrnlTranspose(const unsigned int DimM, const unsigned int DimN, DATA_TYPE *Matrix, DATA_TYPE *MatrixTrans);
}

#endif