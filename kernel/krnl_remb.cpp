#include "hls_stream.h"
#include "types.hpp"
#include "krnl_remb.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

#define INDEX_FROM_2D(x, y, c) (x) * (c) + (y)

extern "C"
{
    void KrnlREmb(DATA_TYPE *Matrix,
                  const unsigned int Offset,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  DATA_TYPE *MatrixPos,
                  const unsigned int OffsetP,
                  DATA_TYPE *MatrixRes,
                  const unsigned int OffsetRes)
    {
        assert(DimN >= DATA_PACK_NUM);

        for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
        {
            for (unsigned int IterBlock = 0; IterBlock < DimN / DATA_PACK_NUM; IterBlock++)
            {
#pragma HLS PIPELINE
                WideType<DATA_TYPE, DATA_PACK_NUM> DIn =
                    ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&Matrix[Offset + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0];

                WideType<DATA_TYPE, DATA_PACK_NUM> DW =
                    ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixPos[OffsetP + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0];

                WideType<DATA_TYPE, DATA_PACK_NUM> DOut;
                for (unsigned int IterR = 0; IterR < DATA_PACK_NUM; IterR += 2)
                {
                    DOut[IterR] =
                        DIn[IterR] * DW[IterR] - DIn[IterR + 1] * DW[IterR + 1];
                    DOut[IterR + 1] =
                        DIn[IterR] * DW[IterR + 1] + DIn[IterR + 1] * DW[IterR];
                }
                ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixRes[OffsetRes + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0] = DOut;
            }
        }
    };
}