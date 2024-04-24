#ifndef __KRNL_GEMV_SINGLEPORT__
#define __KRNL_GEMV_SINGLEPORT__

#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE ap_int<8>
#define DATA_WIDTH 8
#define INT8_PACK_NUM 32
#define N_CHANNELS 16

#define DATA_PACK_NUM 32

#define MAX_MATRIX_SIZE 128 * 128
#define MAX_VECTOR_SIZE 128

typedef ap_int<8 * 32> PACK_INT8_32;
typedef ap_int<8 * 16> PACK_INT8_16;
typedef ap_int<32 * 32> PACK_INT32_32;

void ReadFromMem(
    PACK_INT8_32 *MatrixDDR1,
    PACK_INT8_32 *VecDDR0,
    hls::stream<PACK_INT8_32> *MatrixS,
    hls::stream<PACK_INT8_32> *VecS,
    unsigned int DimM,
    unsigned int DimN);

void Dot(
    hls::stream<PACK_INT8_32> &MatrixS,
    hls::stream<PACK_INT8_32> &VecS, 
    hls::stream<DATA_TYPE> &VecRes,
    unsigned int RowNum,
    unsigned int ColNum);

extern "C"
{
    void KrnlGemvSP(
        PACK_INT8_32 *MatrixDDR1,
        PACK_INT8_32 *VecDDR0,
        PACK_INT8_16 *VecResDDR1,
        unsigned int DimM,
        unsigned int DimN);
}

#endif