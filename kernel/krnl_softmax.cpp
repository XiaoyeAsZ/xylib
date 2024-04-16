#include "hls_stream.h"
#include "types.hpp"
#include "krnl_softmax.h"
#include "u280.h"
#include <iostream>
#include "hls_math.h"

#define INDEX_FROM_2D(x, y, c) (x) * (c) + (y)

#define MAX_FIFO_DEPTH 2048

void ProcessBlock(DATA_TYPE *MatrixA,
                  const unsigned int OffsetA,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Dd,
                  hls::stream<DATA_TYPE> &Bs,
                  hls::stream<DATA_TYPE> &Ds,
                  DATA_TYPE *res)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        DATA_TYPE RowSum = 0;
        for (unsigned int IterBlock = 0; IterBlock < DimN / DATA_PACK_NUM; IterBlock++)
        {
#pragma HLS PIPELINE
            WideType<DATA_TYPE, DATA_PACK_NUM> DIn =
                ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixA[OffsetA + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0];

            WideType<DATA_TYPE, DATA_PACK_NUM> DE;
            for (unsigned int IterExp = 0; IterExp < DATA_PACK_NUM; IterExp++)
            {
                DE[IterExp] = hls::exp(DIn[IterExp]);
            }

            DATA_TYPE PartialSum = 0;
            for (unsigned int IterAdd = 0; IterAdd < DATA_PACK_NUM; IterAdd++)
                PartialSum += DE[IterAdd];
            RowSum += PartialSum;

            WideType<DATA_TYPE, DATA_PACK_NUM> DV;
            for (unsigned int IterDiv = 0; IterDiv < DATA_PACK_NUM; IterDiv++)
            {
                DV[IterDiv] = DE[IterDiv] / PartialSum;
            }

            Dd.write(DV);
            Bs.write(PartialSum);
        }
        Ds.write(RowSum);
    }
}

void Merge(DATA_TYPE *MatrixRes,
           const unsigned int OffsetRes,
           const unsigned int DimM,
           const unsigned int DimN,
           hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt> &Dd,
           hls::stream<DATA_TYPE> &Bs,
           hls::stream<DATA_TYPE> &Ds)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        DATA_TYPE Rs = Ds.read();

        DATA_TYPE bs;
        for (unsigned int IterBlock = 0; IterBlock < DimN / DATA_PACK_NUM; IterBlock++)
        {
            WideType<DATA_TYPE, DATA_PACK_NUM> RD;
            WideType<DATA_TYPE, DATA_PACK_NUM> WD;
            RD = Dd.read();
            bs = Bs.read();
            for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
            {
                WD[IterUnroll] = RD[IterUnroll] * bs / Rs;
            }
            ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixRes[OffsetRes + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0] = WD;
        }
    }
}

extern "C"
{
    void KrnlSoftmax(DATA_TYPE MatrixA[MAX_MATRIX_SIZE],
                     const unsigned int OffsetA,
                     const unsigned int DimM,
                     const unsigned int DimN,
                     DATA_TYPE MatrixRes[MAX_MATRIX_SIZE],
                     const unsigned int OffsetRes)
    {
        hls::stream<WideType<DATA_TYPE, DATA_PACK_NUM>::t_TypeInt, MAX_FIFO_DEPTH> Dd;
        hls::stream<DATA_TYPE, MAX_FIFO_DEPTH> Bs;
        hls::stream<DATA_TYPE, MAX_FIFO_DEPTH> Ds;

#pragma HLS DATAFLOW
        ProcessBlock(MatrixA, OffsetA, DimM, DimN, Dd, Bs, Ds, MatrixRes);
        Merge(MatrixRes, OffsetRes, DimM, DimN, Dd, Bs, Ds);
    }
}