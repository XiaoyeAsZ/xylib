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

void Gemv(
    const unsigned int RowNum,
    const unsigned int ColNum,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Matrix,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Vector,
    hls::stream<RES_TYPE> &Res)
{

    for (unsigned int IterRow = 0; IterRow < RowNum; IterRow++)
    {
        RES_TYPE ParSum = 0;
        for (unsigned int IterColPack = 0; IterColPack < ColNum / DATA_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            WideType<DATA_TYPE, DATA_PACK_NUM> X = Matrix.read();
            WideType<DATA_TYPE, DATA_PACK_NUM> Y = Vector.read();
            WideType<RES_TYPE, DATA_PACK_NUM> R;
            for (unsigned int IterUnrollMul = 0; IterUnrollMul < DATA_PACK_NUM; IterUnrollMul++)
            {
                R[IterUnrollMul] = X[IterUnrollMul] * Y[IterUnrollMul];
            }
            // FIFO_Mul2AddTree.write(R);

            for (unsigned int IterUnrolAdd = 0; IterUnrolAdd < DATA_PACK_NUM; IterUnrolAdd++)
            {
                ParSum += R[IterUnrolAdd];
            }
        }
        Res.write(ParSum);
    }
}

void WriteToMem(
    const unsigned int ColNum,
    RES_TYPE *ResInMem,
    hls::stream<RES_TYPE> &Res)
{
    for (unsigned int IterCol = 0; IterCol < ColNum; IterCol++)
    {
#pragma HLS PIPELINE
        ResInMem[IterCol] = Res.read();
    }
}

extern "C"
{
    void KrnlGemv(
        const unsigned int RowNum,
        const unsigned int ColNum,
        // int *MatrixInMem,
        // int *VectorInMem,
        // int *ResInMem
        DATA_TYPE MatrixInMem[MAX_MATRIX_SIZE],
        DATA_TYPE VectorInMem[MAX_VECTOR_SIZE],
        RES_TYPE ResInMem[MAX_VECTOR_SIZE])
    {
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, 1> Matrix;
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, 1> Vector;
        hls::stream<RES_TYPE, 1> Res;

#pragma HLS DATAFLOW
        ReadFromMem(RowNum, ColNum, MatrixInMem, VectorInMem, Matrix, Vector);
        Gemv(RowNum, ColNum, Matrix, Vector, Res);
        WriteToMem(ColNum, ResInMem, Res);
    }
}