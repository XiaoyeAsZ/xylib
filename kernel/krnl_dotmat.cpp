#include "hls_stream.h"
#include "types.hpp"
#include "krnl_dotmat.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

extern "C"
{
    void KrnlDotMat(DATA_TYPE *MatrixA,
                    const unsigned int OffsetA,
                    DATA_TYPE *MatrixB,
                    const unsigned int OffsetB,
                    const unsigned int DimM,
                    const unsigned int DimN,
                    DATA_TYPE *MatrixRes,
                    const unsigned int OffsetRes)
    {

        WideType<DATA_TYPE, DATA_PACK_NUM> AIn;
        WideType<DATA_TYPE, DATA_PACK_NUM> BIn;
        WideType<DATA_TYPE, DATA_PACK_NUM> Res;

        for (unsigned int IterRound = 0; IterRound < DimM * DimN / DATA_PACK_NUM; IterRound++)
        {
#pragma HLS PIPELINE
            // std::cout << "round: " << IterRound << std::endl;
            AIn = ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixA[OffsetA + IterRound * DATA_PACK_NUM]))[0];
            BIn = ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixB[OffsetB + IterRound * DATA_PACK_NUM]))[0];

            for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
            {
                Res[IterUnroll] = AIn[IterUnroll] * BIn[IterUnroll];
                // std::cout << "mul " << AIn[IterUnroll] << " and " << BIn[IterUnroll] << " res : " << Res[IterUnroll] << std::endl;
            }

            ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixRes[OffsetRes + IterRound * DATA_PACK_NUM]))[0] = Res;
        }
    }
}