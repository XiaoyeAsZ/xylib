/**
 *
 *
 *
 */

#include "hls_stream.h"
#include "types.hpp"
#include "krnl_gemm.h"
#include "u280.h"
#include <iostream>

#define INDEX_FROM_2D(x, y) x *DATA_PACK_NUM + y

void ReadFromMem(
    const unsigned int DimM,
    const unsigned int DimN,
    const unsigned int DimK,
    DATA_TYPE *MatrixAInMem,
    DATA_TYPE *MatrixBInMem,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixATri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri)
{
    WideType<DATA_TYPE, DATA_PACK_NUM> TriBuf[DATA_PACK_NUM];
#pragma HLS ARRAY_PARTITION variable = TriBuf dim = 1 complete

    for (unsigned int IterBlockM; IterBlockM < DimM / DATA_PACK_NUM; IterBlockM++)
    {
        for (unsigned int IterBlockN; IterBlockN < DimK / DATA_PACK_NUM; IterBlockN++)
        {
            WideType<DATA_TYPE, DATA_PACK_NUM> A;
            TriangSrl<DATA_TYPE, DATA_PACK_NUM> ATri;
            ATri.clear();
            for (unsigned int IterCol; IterCol < DimN; IterCol++)
            {
#pragma HLS PIPELINE
                for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
                {
                    A[IterEntry] = MatrixAInMem[(IterBlockM * DATA_PACK_NUM + IterEntry) * DimN + IterCol];
                }
                MatrixATri.write(ATri.shift(A));
            }

            WideType<DATA_TYPE, DATA_PACK_NUM> B;
            TriangSrl<DATA_TYPE, DATA_PACK_NUM> BTri;
            BTri.clear();
            for (unsigned int IterRow; IterRow < DimN; IterRow++)
            {
#pragma HLS PIPELINE
                B = *(WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixBInMem[IterRow * DimK + IterBlockN * DATA_PACK_NUM]);
                MatrixBTri.write(BTri.shift(B));
            }
        }
    }
}

// void SystolicArray(
//     hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixATri,
//     hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &MatrixBTri,
//     hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &ResTri)
// {
//     hls::stream<DATA_TYPE, 1> VerFIFO[DATA_PACK_NUM * DATA_PACK_NUM];
//     hls::stream<DATA_TYPE, 1> HorFIFO[DATA_PACK_NUM * DATA_PACK_NUM];
//     hls::stream<DATA_TYPE, 1> SumFIFO[DATA_PACK_NUM * DATA_PACK_NUM];
// #pragma HLS ARRAY_PARTITION variable = VerFIFO dim = 1 complete
// #pragma HLS ARRAY_PARTITION variable = HorFIFO dim = 1 complete
// #pragma HLS ARRAY_PARTITION variable = SumFIFO dim = 1 complete

//     for (unsigned int IterInit = 0; IterInit < DATA_PACK_NUM * DATA_PACK_NUM; IterInit++)
//     {
// #pragma HLS UNROLL
//         VerFIFO[IterInit].write(0);
//         HorFIFO[IterInit].write(0);
//         SumFIFO[IterInit].write(0);
//     }

//     while (1)
//     {
// #pragma HLS PIPELINE
//         WideType<DATA_TYPE, DATA_PACK_NUM> A = MatrixATri.read();
//         WideType<DATA_TYPE, DATA_PACK_NUM> B = MatrixBTri.read();

//         for (unsigned int IterHor = 0; IterHor < DATA_PACK_NUM; IterHor++)
//         {

//             for (unsigned int IterVer = 0; IterVer < DATA_PACK_NUM; IterVer++)
//             {
//                 DATA_TYPE PartialSum;
//                 DATA_TYPE LocalA;
//                 DATA_TYPE LocalB;
//                 if (IterHor == 0 && IterVer == 0)
//                 {
//                     LocalA = A[0];
//                     LocalB = B[0];
//                     PartialSum = 0;
//                 }
//                 else if (IterVer == 0)
//                 {
//                     LocalA = A[IterHor];
//                     LocalB = VerFIFO[INDEX_FROM_2D(IterHor - 1, IterVer)].read();
//                     PartialSum = 0;
//                 }
//                 else if (IterHor == 0)
//                 {
//                     LocalA = VerFIFO[INDEX_FROM_2D(IterHor, IterVer - 1)].read();
//                     LocalB = B[IterVer];
//                     PartialSum = SumFIFO[INDEX_FROM_2D(IterHor, IterVer - 1)].read();
//                 }
//                 else
//                 {
//                     LocalA = VerFIFO[INDEX_FROM_2D(IterHor, IterVer - 1)].read();
//                     LocalB = VerFIFO[INDEX_FROM_2D(IterHor - 1, IterVer)].read();
//                     PartialSum = SumFIFO[INDEX_FROM_2D(IterHor, IterVer - 1)].read();
//                 }

//                 if (IterHor != DATA_PACK_NUM - 1)
//                     VerFIFO[INDEX_FROM_2D(IterHor + 1, IterVer)].write(LocalB);

//                 if (IterVer != DATA_PACK_NUM - 1)
//                 {
//                     HorFIFO[INDEX_FROM_2D(IterHor, IterVer + 1)].write(LocalA);
//                 }

//                 if (IterVer == DATA_PACK_NUM - 1)
//                 {
//                     WideType<DATA_TYPE, DATA_PACK_NUM> MergeRes;
//                     for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
//                         MergeRes[IterUnroll] = PartialSum + LocalA * LocalB;
//                     ResTri.write(MergeRes);
//                 }
//                 else
//                     SumFIFO[INDEX_FROM_2D(IterHor, IterVer + 1)].write(PartialSum + LocalA * LocalB);
//             }
//         }
//     }
// }

extern "C"
{

    void KrnlGemm(
        const unsigned int DimM,
        const unsigned int DimN,
        const unsigned int DimK,
        DATA_TYPE MatrixAInMem[MAX_MATRIX_SIZE],
        DATA_TYPE MatrixBInMem[MAX_MATRIX_SIZE],
        RES_TYPE MatrixResInMem[MAX_MATRIX_SIZE])
    {

        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixATri;
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixBTri;

#pragma HLS DATAFLOW
        ReadFromMem(DimM, DimN, DimK, MatrixAInMem, MatrixBInMem, MatrixATri, MatrixBTri);
    }
}
