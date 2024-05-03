#ifndef __KRNL_RMSNORM__
#define __KRNL_RMSNORM__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<8>
#define DATA_WIDTH 8
#define DATA_PACK_NUM 32
#define MAX_MATRIX_SIZE 64 * 64

#define INT8_PACK_NUM 32

#define MAX_TOKEN_LEN 128

typedef ap_int<8 * 32> PACK_INT8_32;

extern "C"
{
    void KrnlRMSNorm(
        PACK_INT8_32 * Matrix,
        PACK_INT8_32 * NormScale,
        PACK_INT8_32 * Res,
        unsigned int DimM,
        unsigned int DimN,
        float Scale1,
        float Scale2,
        float Scale3
    );
}

#endif