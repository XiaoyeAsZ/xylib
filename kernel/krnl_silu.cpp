#include "hls_stream.h"
#include "types.hpp"
#include "krnl_silu.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

extern "C"
{
    void KrnlSilu(DATA_TYPE *Matrix,
                  const unsigned int Offset,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  DATA_TYPE *MatrixRes,
                  const unsigned int OffsetRes,
                  float Scale1,
                  float Scale2)
    {
        for (unsigned int IterRound = 0; IterRound < DimM * DimN / DATA_PACK_NUM; IterRound++)
        {
#pragma HLS PIPELINE
            WideType<DATA_TYPE, DATA_PACK_NUM>
                Input = ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&Matrix[Offset + IterRound * DATA_PACK_NUM]))[0];
            WideType<DATA_TYPE, DATA_PACK_NUM> Res;
            for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
            {

                Res[IterUnroll] = int((Input[IterUnroll] * (1.0 / (1.0 + hls::exp(-Input[IterUnroll] / Scale1)))) * Scale2 + 0.5);
                // std::cout << "source: " << Input[IterUnroll] << " after: " << Res[IterUnroll] << " ";
            }
            // std::cout << std::endl;
            ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixRes[OffsetRes + IterRound * DATA_PACK_NUM]))[0] = Res;
        }
    }
}