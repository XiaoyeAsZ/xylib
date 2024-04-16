#include "hls_stream.h"
#include "types.hpp"
#include "krnl_gemv.h"
#include "u280.h"
#include <iostream>

using namespace blas;

void ReadFromMem(
    const unsigned int RowNum,
    const unsigned int ColNum,
    DATA_TYPE *MatrixInMem,
    DATA_TYPE *VectorInMem,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Matrix,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Vector)
{

    for (unsigned int IterRow = 0; IterRow < RowNum; IterRow++)
    {
        for (unsigned int IterEntry = 0; IterEntry < ColNum / DATA_PACK_NUM; IterEntry++)
        {
#pragma HLS PIPELINE

            WideType<DATA_TYPE, DATA_PACK_NUM> MatrixT;
            WideType<DATA_TYPE, DATA_PACK_NUM> VectorT;

            MatrixT = ((WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt *)MatrixInMem)[IterRow * ColNum / DATA_PACK_NUM + IterEntry];
            VectorT = ((WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt *)VectorInMem)[IterEntry];

            Matrix.write(MatrixT);
            Vector.write(VectorT);
        }
    }
}

extern "C"
{
    void KrnlGemv(DATA_TYPE *Value)
    {

    }
}