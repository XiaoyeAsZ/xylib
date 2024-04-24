#include "hls_stream.h"
#include "types.hpp"
#include "krnl_gemv_multiport.h"
#include "u280.h"
#include <iostream>

using namespace blas;

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
    unsigned int DimN)
{
#pragma HLS DATAFLOW
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterRead = 0; IterRead < DimM / INT8_PACK_NUM; IterRead++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadVec = VecDDR0[IterRead];
            for (unsigned int IterUnroll = 0; IterUnroll < N_CHANNELS; IterUnroll++)
            {
                VecS[IterUnroll].write(ReadVec);
            }
            for (int i = 0; i < 32; i++)
                std::cout << "read vec: " << ReadVec(i * 8 + 7, i * 8) << " ";
            std::cout << std::endl;
        }
    }

    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM0[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[0].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM1[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[1].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM2[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[2].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM3[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[3].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM4[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[4].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM5[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[5].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM6[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[6].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM7[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[7].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM8[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[8].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM9[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[9].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM10[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[10].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM11[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[11].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM12[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[12].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM13[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[13].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM14[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[14].write(ReadMat);
        }
    }
    for (unsigned int IterRow = 0; IterRow < DimM / N_CHANNELS; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < DimN / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 ReadMat = MatrixHBM15[IterRow * DimN / INT8_PACK_NUM + IterPack];
            MatrixS[15].write(ReadMat);
        }
    }
}

void Dot(
    hls::stream<PACK_INT8_32> &MatrixS,
    hls::stream<PACK_INT8_32> &VecS,
    hls::stream<DATA_TYPE> &VecRes,
    unsigned int RowNum,
    unsigned int ColNum)
{
    hls::stream<PACK_INT32_32> MulS;
    for (unsigned int IterRow = 0; IterRow < RowNum; IterRow++)
    {
        for (unsigned int IterPack = 0; IterPack < ColNum / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT8_32 MatrixE = MatrixS.read();
            PACK_INT8_32 VecE = VecS.read();
            PACK_INT32_32 MulRes;
            for (unsigned int IterMul = 0; IterMul < INT8_PACK_NUM; IterMul++)
            {
                ap_int<8> A = MatrixE(IterMul * 8 + 7, IterMul * 8);
                ap_int<8> B = VecE(IterMul * 8 + 7, IterMul * 8);
                MulRes(IterMul * 32 + 31, IterMul * 32) = A * B;
                // std::cout << "mul: " << A
                //           << " : " << B << " : " << ap_int<32>(MulRes(IterMul * 32 + 31, IterMul * 32)) << std::endl;
            }
            MulS.write(MulRes);
        }
    }

    for (unsigned int IterRow = 0; IterRow < RowNum; IterRow++)
    {
        ap_int<32> PSumStage0 = 0;
        ap_int<32> PSumStage1 = 0;
        for (unsigned int IterPack = 0; IterPack < ColNum / INT8_PACK_NUM; IterPack++)
        {
#pragma HLS PIPELINE
            PACK_INT32_32 AddE = MulS.read();
            for (unsigned int IterAdd = 0; IterAdd < INT8_PACK_NUM / 2; IterAdd++)
            {
                // std::cout << "add: " << PSumStage0 << " : " << ap_int<8>(AddE(IterAdd * 8 + 7, IterAdd * 8)) << " res: ";
                PSumStage0 += ap_int<32>(AddE(IterAdd * 32 + 31, IterAdd * 32));
            }
            PSumStage1 = PSumStage0;
            for (unsigned int IterAdd = INT8_PACK_NUM / 2; IterAdd < INT8_PACK_NUM; IterAdd++)
            {
                // std::cout << "add: " << PSumStage1 << " : " << ap_int<8>(AddE(IterAdd * 8 + 7, IterAdd * 8)) << " res: ";
                PSumStage1 += ap_int<32>(AddE(IterAdd * 32 + 31, IterAdd * 32));
            }
        }
        ap_int<8> MRes = (PSumStage1 / 16129.0) * 127;
        std::cout << "write: " << PSumStage1 << " : " << MRes << std::endl;
        VecRes.write(MRes);
    }
}

void WriteToMem(
    hls::stream<DATA_TYPE> *VecRes,
    PACK_INT8_16 *VecResDDR0,
    unsigned int DimM,
    unsigned int DimN)
{
    for (unsigned int IterVec = 0; IterVec < DimN / N_CHANNELS; IterVec++)
    {
        PACK_INT8_16 ResT;
        for (unsigned int IterUnroll = 0; IterUnroll < N_CHANNELS; IterUnroll++)
        {
#pragma HLS UNROLL
            ResT(IterUnroll * 8 + 7, IterUnroll * 8) = VecRes[IterUnroll].read();
        }
        VecResDDR0[IterVec] = ResT;
    }
}

extern "C"
{
    void KrnlGemvMP(
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
        unsigned int DimN)
    {
        hls::stream<PACK_INT8_32> MatrixS[16];
        hls::stream<PACK_INT8_32> VecS[16];
        hls::stream<DATA_TYPE> VecRes[16];
#pragma HLS ARRAY_PARTITION variable = MatrixS dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = VecS dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = VecRes dim = 1 complete

#pragma HLS DATAFLOW

        ReadFromMem(MatrixHBM0, MatrixHBM1, MatrixHBM2, MatrixHBM3,
                    MatrixHBM4, MatrixHBM5, MatrixHBM6, MatrixHBM7,
                    MatrixHBM8, MatrixHBM9, MatrixHBM10, MatrixHBM11,
                    MatrixHBM12, MatrixHBM13, MatrixHBM14, MatrixHBM15,
                    VecDDR0, MatrixS, VecS, DimM, DimN);

        for (unsigned int IterDot = 0; IterDot < N_CHANNELS; IterDot++)
        {
#pragma HLS UNROLL
            Dot(MatrixS[IterDot], VecS[IterDot], VecRes[IterDot], DimM / N_CHANNELS, DimN);
        }

        WriteToMem(VecRes, VecResDDR0, DimM, DimN);
    }
}