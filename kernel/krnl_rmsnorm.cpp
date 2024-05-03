#include "hls_stream.h"
#include "types.hpp"
#include "krnl_rmsnorm.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

using namespace blas;

void ProcessBlock(
    PACK_INT8_32 *Matrix,
    PACK_INT8_32 *NormScale,
    unsigned int DimM,
    unsigned int DimN,
    hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt, 128> &MulS,
    hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt> &PowS,
    float Scale1,
    float Scale2)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            std::cout << IterRow << " : " << IterColPack << std::endl;
            WideType<float, INT8_PACK_NUM> MulRes;
            WideType<float, INT8_PACK_NUM> PowRes;
            PACK_INT8_32 MatrixPack = Matrix[IterRow * DimN / INT8_PACK_NUM + IterColPack];
            PACK_INT8_32 NormScalePack = NormScale[IterRow * DimN / INT8_PACK_NUM + IterColPack];
            for (unsigned int IterEntry = 0; IterEntry < INT8_PACK_NUM; IterEntry++)
            {
                ap_int<8> A = MatrixPack(IterEntry * 8 + 7, IterEntry * 8);
                ap_int<8> B = NormScalePack(IterEntry * 8 + 7, IterEntry * 8);
                MulRes[IterEntry] = A * B / Scale1 / Scale2;
                PowRes[IterEntry] = A * A / Scale1 / Scale1;

                // std::cout << "readA: " << A << " readB: " << B << " A*B: " << MulRes[IterEntry] << " A*A: " << PowRes[IterEntry] << std::endl;
            }
            MulS.write(MulRes);
            PowS.write(PowRes);
        }
    }
}

void Merge(
    unsigned int DimM,
    unsigned int DimN,
    hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt> &PowS,
    hls::stream<float> &PartialS)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        float RowSum = 0;
        for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            WideType<float, INT8_PACK_NUM> Entry = PowS.read();
            for (unsigned int IterEntry = 0; IterEntry < INT8_PACK_NUM; IterEntry++)
                RowSum += Entry[IterEntry];
        }
        PartialS.write(hls::sqrt(RowSum / DimN));
        std::cout << "write: " << RowSum << std::endl;
    }
}

void Div(
    PACK_INT8_32 *Res,
    unsigned int DimM,
    unsigned int DimN,
    hls::stream<float> &PartialS,
    hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt, 128> &MulS,
    float Scale3)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        float Rowsum = PartialS.read();
        for (unsigned int IterColPack = 0; IterColPack < DimN / INT8_PACK_NUM; IterColPack++)
        {
#pragma HLS PIPELINE
            WideType<float, INT8_PACK_NUM> Entry = MulS.read();
            PACK_INT8_32 ResEntry;
            for (unsigned int IterEntry = 0; IterEntry < INT8_PACK_NUM; IterEntry++)
                ResEntry(IterEntry * 8 + 7, IterEntry * 8) = int(Entry[IterEntry] / Rowsum * Scale3 + 0.5);
            Res[IterRow * DimN / INT8_PACK_NUM + IterColPack] = ResEntry;
        }
    }
}

extern "C"
{

    void KrnlRMSNorm(
        PACK_INT8_32 Matrix[128],
        PACK_INT8_32 NormScale[128],
        PACK_INT8_32 Res[128],
        unsigned int DimM,
        unsigned int DimN,
        float Scale1,
        float Scale2,
        float Scale3)
    {
        hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt, 128> MulS;
        hls::stream<WideType<float, INT8_PACK_NUM>::t_TypeInt> PowS;
        hls::stream<float> PartialS;

#pragma HLS DATAFLOW

        ProcessBlock(Matrix, NormScale, DimM, DimN, MulS, PowS, Scale1, Scale2);

        Merge(DimM, DimN, PowS, PartialS);

        Div(Res, DimM, DimN, PartialS, MulS, Scale3);
    }
}
