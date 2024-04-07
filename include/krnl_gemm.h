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
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> *MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> *MatrixBTri);

void Mac(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
    hls::stream<DATA_TYPE> &AccRes);

void Mac(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTriNxt,
    hls::stream<DATA_TYPE> &AccRes);

void Mul(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Exit);

void Mul(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTriNxt,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Exit);

void Add(
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt> &Exit,
    hls::stream<DATA_TYPE> &AccRes);

void SystolicArray(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> *MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> *MatrixBTri,
    hls::stream<DATA_TYPE> *AccRes);

void WriteToMem(
    const unsigned int DimM,
    const unsigned int DimK,
    hls::stream<DATA_TYPE> *AccRes,
    DATA_TYPE *MatrixResInMem);

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