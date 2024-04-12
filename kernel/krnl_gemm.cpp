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

#define INDEX_FROM_2D(x, y, c) (x) * (c) + (y)

void ReadFromMem(
    const unsigned int DimM,
    const unsigned int DimN,
    const unsigned int DimK,
    DATA_TYPE *MatrixAInMem,
    DATA_TYPE *MatrixBInMem,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixA,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixB)
{
    for (unsigned int IterBlockM = 0; IterBlockM < DimM / DATA_PACK_NUM; IterBlockM++)
    {
        for (unsigned int IterBlockN = 0; IterBlockN < DimK / DATA_PACK_NUM; IterBlockN++)
        {

            for (unsigned int IterCol = 0; IterCol < DimN; IterCol++)
            {
#pragma HLS PIPELINE
                WideType<DATA_TYPE, DATA_PACK_NUM> A;
                for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
                {
                    A[IterEntry] = MatrixAInMem[INDEX_FROM_2D(IterBlockM * DATA_PACK_NUM + IterEntry, IterCol, DimN)];
                }
                // std::cout << "read A(" << IterBlockM << "," << IterBlockN << "," << IterCol << ")\n";
                MatrixA.write(A);

                WideType<DATA_TYPE, DATA_PACK_NUM> B;
                B = *(WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixBInMem[INDEX_FROM_2D(IterCol, IterBlockN * DATA_PACK_NUM, DimK)]);
                // std::cout << "read B(" << IterBlockM << "," << IterBlockN << "," << IterCol << ")\n";
                MatrixB.write(B);
            }

            //             for (unsigned int IterCol = 0; IterCol < DimN; IterCol++)
            //             {
            // #pragma HLS PIPELINE
            //                 WideType<DATA_TYPE, DATA_PACK_NUM> A;
            //                 for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
            //                 {
            //                     A[IterEntry] = MatrixAInMem[INDEX_FROM_2D(IterBlockM * DATA_PACK_NUM + IterEntry, IterCol, DimN)];
            //                 }
            //                 std::cout << "read A(" << IterBlockM << "," << IterBlockN << "," << IterCol << ")\n";

            //                 MatrixA.write(A);
            //             }

            //             for (unsigned int IterRow = 0; IterRow < DimN; IterRow++)
            //             {
            // #pragma HLS PIPELINE
            //                 WideType<DATA_TYPE, DATA_PACK_NUM> B;
            //                 B = *(WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixBInMem[INDEX_FROM_2D(IterRow, IterBlockN * DATA_PACK_NUM, DimK)]);
            //                 std::cout << "read B(" << IterBlockM << "," << IterBlockN << "," << IterRow << ")\n";
            //                 MatrixB.write(B);
            //             }
        }
    }
}

void UnpackTri(
    const unsigned int DimM,
    const unsigned int DimN,
    const unsigned int DimK,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixA,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixB,
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> *MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTri)
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
                WideType<DATA_TYPE, DATA_PACK_NUM> TA = MatrixA.read();
                for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
                {
                    DualTaggedType<DATA_TYPE> A;
                    A.m_val = TA[IterEntry];
                    A.m_flush = (IterCol == DimN - 1);
                    A.m_exit = (IterBlockM == DimM / DATA_PACK_NUM - 1 && IterBlockN == DimK / DATA_PACK_NUM - 1 && IterCol == DimN - 1);
                    MatrixATaged[IterEntry].write(A);
                }

                WideType<DATA_TYPE, DATA_PACK_NUM> B;
                B = MatrixB.read();
                if (!(IterBlockM == 0 && IterBlockN == 0 && IterCol == 0))
                    MatrixBTri.write(BTri.shift(B));
                else
                    BTri.shift(B);
            }

            if (IterBlockM == DimM / DATA_PACK_NUM - 1 && IterBlockN == DimK / DATA_PACK_NUM - 1)
            {
                for (unsigned int IterRemain = 0; IterRemain < DATA_PACK_NUM; IterRemain++)
                    MatrixBTri.write(BTri.shift(WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt(0)));
            }

            //             for (unsigned int IterCol = 0; IterCol < DimN; IterCol++)
            //             {
            // #pragma HLS PIPELINE
            //                 WideType<DATA_TYPE, DATA_PACK_NUM> TA = MatrixA.read();
            //                 for (unsigned int IterEntry = 0; IterEntry < DATA_PACK_NUM; IterEntry++)
            //                 {
            //                     DualTaggedType<DATA_TYPE> A;
            //                     A.m_val = TA[IterEntry];
            //                     A.m_flush = (IterCol == DimN - 1);
            //                     A.m_exit = (IterBlockM == DimM / DATA_PACK_NUM - 1 && IterBlockN == DimK / DATA_PACK_NUM - 1 && IterCol == DimN - 1);
            //                     MatrixATaged[IterEntry].write(A);
            //                 }
            //             }

            //             WideType<DATA_TYPE, DATA_PACK_NUM> B;
            //             for (unsigned int IterRow = 0; IterRow < DimN; IterRow++)
            //             {
            // #pragma HLS PIPELINE
            //                 B = MatrixB.read();
            //                 if (!(IterBlockM == 0 && IterBlockN == 0 && IterRow == 0))
            //                     MatrixBTri.write(BTri.shift(B));
            //                 else
            //                     BTri.shift(B);
            //             }
        }
    }
}

void Mac(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTri,
    hls::stream<DATA_TYPE, DATA_PACK_NUM> &AccRes)
{

    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> SumBuf;
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Flush;
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Exit;

#pragma HLS DATAFLOW
    Mul(MatrixATaged, MatrixBTri, SumBuf, Flush, Exit);
    Add(SumBuf, Flush, Exit, AccRes);
}

void Mac(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTriNxt,
    hls::stream<DATA_TYPE, DATA_PACK_NUM> &AccRes)
{

    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> SumBuf;
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Flush;
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Exit;

#pragma HLS DATAFLOW
    Mul(MatrixATaged, MatrixBTri, MatrixBTriNxt, SumBuf, Flush, Exit);
    Add(SumBuf, Flush, Exit, AccRes);
}

void Mul(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Exit)
{

    bool exit;
    WideType<DATA_TYPE, DATA_PACK_NUM> ASrl = WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt(0);
    WideType<bool, DATA_PACK_NUM> AFlush = WideType<bool, DATA_PACK_NUM>::t_TypeInt(0);
    WideType<bool, DATA_PACK_NUM> AExit = WideType<bool, DATA_PACK_NUM>::t_TypeInt(0);

    DualTaggedType<DATA_TYPE> A;
    WideType<DATA_TYPE, DATA_PACK_NUM> B;
    WideType<DATA_TYPE, DATA_PACK_NUM> MulRes;
    do
    {
#pragma HLS PIPELINE
        A = MatrixATaged.read();
        exit = A.m_exit;
        ASrl.shift(A.m_val);
        AFlush.shift(A.m_flush);
        AExit.shift(A.m_exit);

        B = MatrixBTri.read();

        for (unsigned int IterMul = 0; IterMul < DATA_PACK_NUM; IterMul++)
        {
            MulRes[IterMul] = ASrl[IterMul] * B[IterMul];
        }
        SumBuf.write(MulRes);
        Flush.write(AFlush);
        Exit.write(AExit);

        if (exit)
        {
            for (unsigned int IterRemain = 0; IterRemain < DATA_PACK_NUM - 1; IterRemain++)
            {
                ASrl.shift(0);
                AFlush.shift(0);
                AExit.shift(0);

                B = MatrixBTri.read();
                for (unsigned int IterMul = 0; IterMul < DATA_PACK_NUM; IterMul++)
                {
                    MulRes[IterMul] = ASrl[IterMul] * B[IterMul];
                }

                SumBuf.write(MulRes);
                Flush.write(AFlush);
                Exit.write(AExit);
            }
        }

    } while (!exit);
}

void Mul(
    hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> &MatrixATaged,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTri,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &MatrixBTriNxt,
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Exit)
{

    bool exit;
    WideType<DATA_TYPE, DATA_PACK_NUM> ASrl = WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt(0);
    WideType<bool, DATA_PACK_NUM> AFlush = WideType<bool, DATA_PACK_NUM>::t_TypeInt(0);
    WideType<bool, DATA_PACK_NUM> AExit = WideType<bool, DATA_PACK_NUM>::t_TypeInt(0);

    DualTaggedType<DATA_TYPE> A;
    WideType<DATA_TYPE, DATA_PACK_NUM> B;
    WideType<DATA_TYPE, DATA_PACK_NUM> MulRes;

    do
    {
#pragma HLS PIPELINE
        A = MatrixATaged.read();
        exit = A.m_exit;
        ASrl.shift(A.m_val);
        AFlush.shift(A.m_flush);
        AExit.shift(A.m_exit);

        B = MatrixBTri.read();

        for (unsigned int IterMul = 0; IterMul < DATA_PACK_NUM; IterMul++)
        {
            MulRes[IterMul] = ASrl[IterMul] * B[IterMul];
        }
        SumBuf.write(MulRes);
        Flush.write(AFlush);
        Exit.write(AExit);

        MatrixBTriNxt.write(B);

        if (exit)
        {
            for (unsigned int IterRemain = 0; IterRemain < DATA_PACK_NUM - 1; IterRemain++)
            {
                ASrl.shift(0);
                AFlush.shift(0);
                AExit.shift(0);

                B = MatrixBTri.read();
                for (unsigned int IterMul = 0; IterMul < DATA_PACK_NUM; IterMul++)
                {
                    MulRes[IterMul] = ASrl[IterMul] * B[IterMul];
                }

                SumBuf.write(MulRes);
                Flush.write(AFlush);
                Exit.write(AExit);

                MatrixBTriNxt.write(B);
            }
        }

    } while (!exit);
}

void Add(
    hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &SumBuf,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Flush,
    hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> &Exit,
    hls::stream<DATA_TYPE, DATA_PACK_NUM> &AccRes)
{
    WideType<DATA_TYPE, DATA_PACK_NUM> SumVec = WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt(0);

    bool exit;
    do
    {
#pragma HLS PIPELINE
        WideType<DATA_TYPE, DATA_PACK_NUM> ReadVec = SumBuf.read();
        WideType<bool, DATA_PACK_NUM> ReadFlush = Flush.read();
        WideType<bool, DATA_PACK_NUM> ReadExit = Exit.read();
        exit = ReadExit[DATA_PACK_NUM - 1];

        for (unsigned int IterAcc = 0; IterAcc < DATA_PACK_NUM; IterAcc++)
        {
            SumVec[IterAcc] += ReadVec[IterAcc];
            if (ReadFlush[IterAcc])
            {
                AccRes.write(SumVec[IterAcc]);
                SumVec[IterAcc] = 0;
            }
        }

    } while (!exit);
}

void WriteToMem(
    const unsigned int DimM,
    const unsigned int DimK,
    hls::stream<DATA_TYPE, DATA_PACK_NUM> *AccRes,
    DATA_TYPE *MatrixResInMem)
{
    for (unsigned int IterBlockM = 0; IterBlockM < DimM / DATA_PACK_NUM; IterBlockM++)
    {
        for (unsigned int IterBlockN = 0; IterBlockN < DimK / DATA_PACK_NUM; IterBlockN++)
        {
            for (unsigned int IterRow = 0; IterRow < DATA_PACK_NUM; IterRow++)
            {
#pragma HLS PIPELINE
                WideType<DATA_TYPE, DATA_PACK_NUM> ResRow;
                for (unsigned int IterCol = 0; IterCol < DATA_PACK_NUM; IterCol++)
                {
                    ResRow[IterCol] = AccRes[IterRow].read();
                }
                ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixResInMem[INDEX_FROM_2D(IterBlockM * DATA_PACK_NUM + IterRow, IterBlockN * DATA_PACK_NUM, DimK)]))[0] = ResRow;
            }
        }
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

        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixA;
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixB;

        hls::stream<DualTaggedType<DATA_TYPE>::t_TypeInt, DATA_PACK_NUM> MatrixATaged[DATA_PACK_NUM];
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> MatrixBTri[DATA_PACK_NUM];

        hls::stream<DATA_TYPE, DATA_PACK_NUM> AccRes[DATA_PACK_NUM];

#pragma HLS ARRAY_PARTITION variable = MatrixATaged dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = MatrixBTri dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = AccRes dim = 1 complete
#pragma HLS stream variable = MatrixATaged depth = DATA_PACK_NUM
#pragma HLS stream variable = MatrixBTri[0] depth = DATA_PACK_NUM
#pragma HLS stream variable = AccRes depth = DATA_PACK_NUM

        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>, DATA_PACK_NUM> SumBuf[DATA_PACK_NUM];
        hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Flush[DATA_PACK_NUM];
        hls::stream<WideType<bool, DATA_PACK_NUM>::t_TypeInt, DATA_PACK_NUM> Exit[DATA_PACK_NUM];
#pragma HLS ARRAY_PARTITION variable = SumBuf dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = Flush dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = Exit dim = 1 complete

#pragma HLS DATAFLOW
        ReadFromMem(DimM, DimN, DimK, MatrixAInMem, MatrixBInMem, MatrixA, MatrixB);

        UnpackTri(DimM, DimN, DimK, MatrixA, MatrixB, MatrixATaged, MatrixBTri[0]);

        for (unsigned int IterMac = 0; IterMac < DATA_PACK_NUM - 1; IterMac++)
        {
#pragma HLS UNROLL
            Mac(MatrixATaged[IterMac], MatrixBTri[IterMac], MatrixBTri[IterMac + 1], AccRes[IterMac]);
        }
        Mac(MatrixATaged[DATA_PACK_NUM - 1], MatrixBTri[DATA_PACK_NUM - 1], AccRes[DATA_PACK_NUM - 1]);

        WriteToMem(DimM, DimK, AccRes, MatrixResInMem);
    }
}
