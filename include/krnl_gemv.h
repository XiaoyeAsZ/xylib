#ifndef __KRNL_GEMV__
#define __KRNL_GEMV__

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
    PACK_INT8_32 *MatrixHBM0,
    PACK_INT8_32 *MatrixHBM1,
    PACK_INT8_32 *MatrixHBM2,
    PACK_INT8_32 *MatrixHBM3,
    PACK_INT8_32 *MatrixHBM4,
    PACK_INT8_32 *MatrixHBM5,
    PACK_INT8_32 *MatrixHBM6,
    PACK_INT8_32 *MatrixHBM7,
    PACK_INT8_32 *MatrixHBM8,
    PACK_INT8_32 *MatrixHBM9,
    PACK_INT8_32 *MatrixHBM10,
    PACK_INT8_32 *MatrixHBM11,
    PACK_INT8_32 *MatrixHBM12,
    PACK_INT8_32 *MatrixHBM13,
    PACK_INT8_32 *MatrixHBM14,
    PACK_INT8_32 *MatrixHBM15,
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
    void KrnlGemv(
        PACK_INT8_32 *MatrixHBM0,
        PACK_INT8_32 *MatrixHBM1,
        PACK_INT8_32 *MatrixHBM2,
        PACK_INT8_32 *MatrixHBM3,
        PACK_INT8_32 *MatrixHBM4,
        PACK_INT8_32 *MatrixHBM5,
        PACK_INT8_32 *MatrixHBM6,
        PACK_INT8_32 *MatrixHBM7,
        PACK_INT8_32 *MatrixHBM8,
        PACK_INT8_32 *MatrixHBM9,
        PACK_INT8_32 *MatrixHBM10,
        PACK_INT8_32 *MatrixHBM11,
        PACK_INT8_32 *MatrixHBM12,
        PACK_INT8_32 *MatrixHBM13,
        PACK_INT8_32 *MatrixHBM14,
        PACK_INT8_32 *MatrixHBM15,
        PACK_INT8_32 *VecDDR0,
        PACK_INT8_16 *VecResDDR0,
        unsigned int DimM,
        unsigned int DimN);
}

#endif