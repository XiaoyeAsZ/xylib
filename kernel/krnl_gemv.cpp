#include "hls_stream.h"
#include "types.hpp"
#include "krnl_gemv.h"
#include "u280.h"
#include <iostream>

using namespace blas;

void ReadFromMem(
    PACK_INT8_32 *Matrix,
    PACK_INT8_32 *Vec,
    hls::stream<PACK_INT8_32> &MatrixS,
    hls::stream<PACK_INT8_32> &VecS,
    unsigned int DimM,
    unsigned int DimN)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            std::cout<<"read: "<<IterRow<<" : "<<IterColPack<<std::endl;
            PACK_INT8_32 RMatrix = Matrix[IterRow * DimN / INT8_PACK_NUM + IterColPack];
            MatrixS.write(RMatrix);
            PACK_INT8_32 RVec = Vec[IterColPack];
            VecS.write(RVec);
        }
    }

//     for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
//     {
//         for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
//         {
// #pragma HLS PIPELINE
            
//         }
//     }
}

void Dot(
    hls::stream<PACK_INT8_32> &MatrixS,
    hls::stream<PACK_INT8_32> &VecS,
    hls::stream<DATA_TYPE> &ResS,
    unsigned int DimM,
    unsigned int DimN)
{
#pragma HLS DATAFLOW
    hls::stream<PACK_INT32_32> MulResS;
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        PACK_INT32_32 MulRes;
        ap_int<32> AddResStage0 = 0;
        for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            std::cout<<"readfroms: "<<IterRow<<" : "<<IterColPack<<std::endl;
            PACK_INT8_32 RMatrix = MatrixS.read();
            PACK_INT8_32 RVec = VecS.read();
            for (unsigned int IterMul = 0; IterMul < INT8_PACK_NUM; IterMul++)
            {
                ap_int<8> A = RMatrix(IterMul * 8 + 7, IterMul * 8);
                ap_int<8> B = RVec(IterMul * 8 + 7, IterMul * 8);
                MulRes(IterMul * 32 + 31, IterMul * 32) = A * B;
            }
            for (unsigned int IterMul = 0; IterMul < INT8_PACK_NUM; IterMul++)
            {
                AddResStage0 += MulRes(IterMul * 32 + 31, IterMul * 32);
            }
            // MulResS.write(MulRes);
        }
        ap_int<8> Requant = int(AddResStage0 / 127.0);
        ResS.write(Requant);
    }

//     for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
//     {
//         ap_int<32> AddResStage0 = 0;
//         ap_int<32> AddResStage1 = 0;
//         for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
//         {
// #pragma HLS PIPELINE
//             std::cout<<"add: "<<IterRow<<" : "<<IterColPack<<std::endl;
//             PACK_INT32_32 RMul = MulResS.read();
//             for (unsigned int IterAdd = 0; IterAdd < INT8_PACK_NUM / 2; IterAdd++)
//             {
//                 ap_int<32> MulEntry = RMul(IterAdd * 32 + 31, IterAdd * 32);
//                 AddResStage0 += MulEntry;
//             }
//             AddResStage1 = AddResStage0;
//             for (unsigned int IterAdd = INT8_PACK_NUM / 2; IterAdd < INT8_PACK_NUM; IterAdd++)
//             {
//                 ap_int<32> MulEntry = RMul(IterAdd * 32 + 31, IterAdd * 32);
//                 AddResStage1 += MulEntry;
//             }
//         }

        // ap_int<8> Requant = int(AddResStage1 / 127.0);
        // ResS.write(Requant);
    // }
}

void WriteToMem(
    hls::stream<DATA_TYPE> &ResS,
    PACK_INT8_32 *Res,
    unsigned int DimM,
    unsigned int DimN)
{
    // unsigned int NRound = (DimM + INT8_PACK_NUM - 1) / INT8_PACK_NUM;
    // for (unsigned int IterRound = 0; IterRound < NRound; IterRound++)
    // {
    //     unsigned int NEntry = (IterRound == NRound - 1) ? (DimM - IterRound * INT8_PACK_NUM) : INT8_PACK_NUM;
    //     if (NEntry == INT8_PACK_NUM)
    //     {
    //         PACK_INT8_32 REntry;
    //         for (unsigned int IterEntry = 0; IterEntry < NEntry; IterEntry++)
    //         {
    //             REntry(IterEntry * 8 + 7, IterEntry * 8) = ResS.read();
    //         }
    //         Res[IterRound] = REntry;
    //     }
    //     else
    //     {
    //         for (unsigned int IterEntry = 0; IterEntry < NEntry; IterEntry++)
    //         {
    //             ap_int<8> REntry = ResS.read();
    //             ((ap_int<8> *)(&Res[IterRound]))[IterEntry] = REntry;
    //         }
    //     }
    // }
    DATA_TYPE *ResD = (DATA_TYPE *)Res;
    for (unsigned int IterRes = 0; IterRes < DimM; IterRes++)
    {
        ResD[IterRes] = ResS.read();
    }
}

extern "C"
{
    void KrnlGemv(
        PACK_INT8_32 Matrix[MAX_MATRIX_SIZE],
        PACK_INT8_32 Vec[MAX_VECTOR_SIZE],
        PACK_INT8_32 *Res,
        unsigned int DimM,
        unsigned int DimN)
    {
        hls::stream<PACK_INT8_32> MatrixS;
        hls::stream<PACK_INT8_32> VecS;
        hls::stream<DATA_TYPE> ResS;

#pragma HLS DATAFLOW
        ReadFromMem(Matrix, Vec, MatrixS, VecS, DimM, DimN);
        Dot(MatrixS, VecS, ResS, DimM, DimN);
        WriteToMem(ResS, Res, DimM, DimN);
    }
}