#include "hls_stream.h"
#include "types.hpp"

using namespace blas;

#define DATA_TYPE float
#define DATA_WIDTH 32
#define DATA_PACK_NUM 8
#define MAX_MATRIX_SIZE 128 * 128

#define MAX_TOKEN_LEN 128

extern "C"
{
    void KrnlAddMat(DATA_TYPE MatrixA[MAX_MATRIX_SIZE],
                    const unsigned int OffsetA,
                    DATA_TYPE MatrixB[MAX_MATRIX_SIZE],
                    const unsigned int OffsetB,
                    const unsigned int DimM,
                    const unsigned int DimN,
                    DATA_TYPE MatrixRes[MAX_MATRIX_SIZE],
                    const unsigned int OffsetRes)
    {
        WideType<DATA_TYPE, DATA_PACK_NUM> DA;
        WideType<DATA_TYPE, DATA_PACK_NUM> DB;
        WideType<DATA_TYPE, DATA_PACK_NUM> DR;
        for (unsigned int IterAdd = 0; IterAdd < DimM * DimN / DATA_PACK_NUM; IterAdd++)
        {
// #pragma HLS PIPELINE
            DA = ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixA[OffsetA + IterAdd * DATA_PACK_NUM]))[0];
            DB = ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixB[OffsetB + IterAdd * DATA_PACK_NUM]))[0];
            for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
                DR[IterUnroll] = DA[IterUnroll] + DB[IterUnroll];
            ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixRes[OffsetRes + IterAdd * DATA_PACK_NUM]))[0] = DR;
        }
    }
}
