#ifndef __KRNL_SILU__
#define __KRNL_SILU__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<32>
#define DATA_WIDTH 32
#define RES_TYPE ap_int<32>
#define RES_WIDTH 32
#define DATA_PACK_NUM 8
#define MAX_MATRIX_SIZE 128 * 128

extern "C"
{
    void KrnlSilu(DATA_TYPE *Matrix,
                  const unsigned int Offeset,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  DATA_TYPE *MatrixTrans);
}

#endif