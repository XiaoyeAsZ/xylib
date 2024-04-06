#ifndef __KRNL_GEMM__
#define __KRNL_GEMM__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<32>
#define DATA_WIDTH 32
#define RES_TYPE ap_int<32>
#define RES_WIDTH 32
#define DATA_PACK_NUM 8
#define MAX_MATRIX_SIZE 128 * 128

void ReadFromMem(
    const unsigned int DimM,
    const unsigned int DimN,
    const unsigned int DimK,
    DATA_TYPE *MatrixAInMem,
    DATA_TYPE *MatrixBInMem,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixATri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri);

void SystolicArray(
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixATri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &ResTri);

// void Gemm(
//     const unsigned int DimM,
//     const unsigned int DimN,
//     const unsigned int DimK,
//     hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixA,
//     hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixB,
//     hls::stream<WideType<RES_TYPE, DATA_PACK_NUM>::t_TypeInt> &Res);

// void WriteToMem(
//     const unsigned int DimM,
//     const unsigned int DimK,
//     RES_TYPE *ResInMem,
//     hls::stream<WideType<RES_TYPE, DATA_PACK_NUM>::t_TypeInt> &Res);

extern "C"
{
    void KrnlGemm(
        const unsigned int DimM,
        const unsigned int DimN,
        const unsigned int DimK,
        DATA_TYPE *MatrixAInMem,
        DATA_TYPE *MatrixBInMem,
        DATA_TYPE *MatrixResInMem);
}

#endif