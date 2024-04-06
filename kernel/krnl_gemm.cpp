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

#define INDEX_FROM_2D(x, y, c) x *c + y

void ReadFromMem(
    const unsigned int DimM,
    const unsigned int DimN,
    const unsigned int DimK,
    DATA_TYPE *MatrixAInMem,
    DATA_TYPE *MatrixBInMem,
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> *MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> *MatrixBTri)
{
    TriangSrl<DATA_TYPE, DATA_PACK_NUM> BTri;
    BTri.clear();
    for (unsigned int IterBlockM = 0; IterBlockM < DimM / DATA_PACK_NUM; IterBlockM++)
    {
        for (unsigned int IterBlockN = 0; IterBlockN < DimK / DATA_PACK_NUM; IterBlockN++)
        {
            for (unsigned int IterCol = 0; IterCol < DimN; IterCol++)
            {
#pragma HLS PIPELINE
                for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
                {
                    DualTaggedType<DATA_TYPE> A;
                    A.m_val = MatrixAInMem[INDEX_FROM_2D(IterBlockM * DATA_PACK_NUM + IterEntry, IterCol, DimN)];
                    A.m_flush = (IterCol == 0);
                    A.m_exit = (IterCol == DimN - 1);
                    MatrixATaged[IterEntry].write(A);
                }
            }

            WideType<DATA_TYPE, DATA_PACK_NUM> B;

            for (unsigned int IterRow = 0; IterRow < DimN; IterRow++)
            {
#pragma HLS PIPELINE
                B = *(WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixBInMem[INDEX_FROM_2D(IterRow, IterBlockN * DATA_PACK_NUM, DimK)]);
                if (!(IterBlockM == 0 && IterBlockN == 0 && IterRow == 0))
                    MatrixBTri[0].write(BTri.shift(B));
            }
        }
    }
    for (unsigned int IterRemain = 0; IterRemain < DATA_PACK_NUM; IterRemain++)
        MatrixBTri[0].write(BTri.shift(WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt(0)));
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

void SystolicArray(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> *MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> *MatrixBTri,
    hls::stream<DATA_TYPE> &Res)
{

    //     for (unsigned int IterRowMac = 0; IterRowMac < DATA_PACK_NUM; IterRowMac++)
    //     {
    // #pragma HLS UNROLL
    //         WideType<DATA_TYPE, DATA_PACK_NUM> ASrl = 0;
    //         DualTaggedType A = MatrixATaged[IterRowMac].read();

    //         ASrl.shift(A.m_val);
    //     }

    for (int i = 0; i < 256 * 128 + 7; i++)
    {
        WideType<DATA_TYPE, DATA_PACK_NUM> t = MatrixBTri[0].read();
        for (int i = 0; i < DATA_PACK_NUM; i++)
        {
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;
    }
}

extern "C"
{

    void KrnlGemm(
        const unsigned int DimM,
        const unsigned int DimN,
        const unsigned int DimK,
        DATA_TYPE MatrixAInMem[MAX_MATRIX_SIZE],
        DATA_TYPE MatrixBInMem[MAX_MATRIX_SIZE],
        DATA_TYPE MatrixResInMem[MAX_MATRIX_SIZE])
    {

        hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt> MatrixATaged[DATA_PACK_NUM];
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixBTri[DATA_PACK_NUM];
        hls::stream<DATA_TYPE> Res;
#pragma HLS ARRAY_PARTITION variable = MatrixATaged dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = MatrixBTri dim = 1 complete

#pragma HLS DATAFLOW
        ReadFromMem(DimM, DimN, DimK, MatrixAInMem, MatrixBInMem, MatrixATaged, MatrixBTri);
        SystolicArray(MatrixATaged, MatrixBTri, Res);
    }
}
