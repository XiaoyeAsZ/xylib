#include "hls_stream.h"
#include "types.hpp"
#include "krnl_move.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

#define INDEX_FROM_2D(x, y, c) (x) * (c) + (y)

extern "C"
{
    void KrnlMove(DATA_TYPE *MatrixA,
                  const unsigned int OffsetA,
                  DATA_TYPE *MatrixB,
                  const unsigned int OffsetB,
                  const unsigned int Len)
    {
        for (unsigned int IterMove = 0; IterMove < Len / DATA_PACK_NUM; IterMove++)
        {
            WideType<DATA_TYPE, DATA_PACK_NUM> DIn =
                ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixA[OffsetA + IterMove * DATA_PACK_NUM]))[0];
            ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixB[OffsetB + IterMove * DATA_PACK_NUM]))[0] = DIn;
        }
    };
}