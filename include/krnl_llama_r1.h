#ifndef __KRNL_LLAMA_R1__
#define __KRNL_LLAMA_R1__

#include "hls_stream.h"
#include "types.hpp"
#include "ap_int.h"

using namespace blas;

typedef ap_int<8 * 32> pack_int8_32;
typedef ap_int<8 * 4096> pack_int8_4096

#define DATA_TYPE ap_int<8>
#define DATA_WIDTH 8

#define INT8_PACK_NUM 32

#define HEAD_NUM 32
#define EMBEDDING_DIM 4096

    extern "C"
{
    void LLAMA_R1(
        const unsigned int RowNum,
        const unsigned int ColNum,
        DATA_TYPE *MatrixInMem,
        DATA_TYPE *VectorInMem,
        RES_TYPE *ResInMem);
}

#endif