#ifndef __KRNL_GEMV__
#define __KRNL_GEMV__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE float
#define DATA_WIDTH 32
#define RES_TYPE ap_int<32>
#define RES_WIDTH 32
#define DATA_PACK_NUM 8
#define MAX_MATRIX_SIZE 128 * 128
#define MAX_VECTOR_SIZE 128 

void ReadFromMem(
    const unsigned int RowNum,
    const unsigned int ColNum,
    DATA_TYPE *MatrixInMem,
    DATA_TYPE *VectorInMem,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Matrix,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Vector);

void Gemv(
    const unsigned int RowNum,
    const unsigned int ColNum,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Matrix,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Vector,
    hls::stream<WideType<RES_TYPE, DATA_PACK_NUM>::t_TypeInt> &Res);

void WriteToMem(
    const unsigned int ColNum,
    RES_TYPE *ResInMem,
    hls::stream<WideType<RES_TYPE, DATA_PACK_NUM>::t_TypeInt> &Res);

extern "C"
{
    void KrnlGemv(
        const unsigned int RowNum,
        const unsigned int ColNum,
        DATA_TYPE *MatrixInMem,
        DATA_TYPE *VectorInMem,
        RES_TYPE *ResInMem);
}

#endif