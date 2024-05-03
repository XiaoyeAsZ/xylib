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
                  hls::stream<WideType<float, DATA_PACK_NUM>::t_TypeInt> &Dd,
                  hls::stream<float> &Bs,
                  hls::stream<float> &Ds,
                  DATA_TYPE *res,
                  float Scale1)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        float RowSum = 0;
        for (unsigned int IterBlock = 0; IterBlock < DimN / DATA_PACK_NUM; IterBlock++)
        {
#pragma HLS PIPELINE
            WideType<DATA_TYPE, DATA_PACK_NUM> DIn =
                ((WideType<DATA_TYPE, DATA_PACK_NUM> *)(&MatrixA[OffsetA + INDEX_FROM_2D(IterRow, IterBlock * DATA_PACK_NUM, DimN)]))[0];

            WideType<float, DATA_PACK_NUM> DE;
            for (unsigned int IterExp = 0; IterExp < DATA_PACK_NUM; IterExp++)
            {
                DE[IterExp] = hls::exp(float(DIn[IterExp]) / Scale1);
            }

            float PartialSum = 0;
            for (unsigned int IterAdd = 0; IterAdd < DATA_PACK_NUM; IterAdd++)
                PartialSum += DE[IterAdd];
            RowSum += PartialSum;

            WideType<float, DATA_PACK_NUM> DV;
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
           hls::stream<WideType<float, DATA_PACK_NUM>::t_TypeInt> &Dd,
           hls::stream<float> &Bs,
           hls::stream<float> &Ds,
           float Scale2)
{
    for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
    {
        float Rs = Ds.read();

        float bs;
        for (unsigned int IterBlock = 0; IterBlock < DimN / DATA_PACK_NUM; IterBlock++)
        {
            WideType<float, DATA_PACK_NUM> RD;
            WideType<DATA_TYPE, DATA_PACK_NUM> WD;
            RD = Dd.read();
            bs = Bs.read();
            for (unsigned int IterUnroll = 0; IterUnroll < DATA_PACK_NUM; IterUnroll++)
            {
                WD[IterUnroll] = int(RD[IterUnroll] * bs / Rs * Scale2 + 0.5);
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
                     const unsigned int OffsetRes,
                     float Scale1,
                     float Scale2)
    {
        hls::stream<WideType<float, DATA_PACK_NUM>::t_TypeInt, MAX_FIFO_DEPTH> Dd;
        hls::stream<float, MAX_FIFO_DEPTH> Bs;
        hls::stream<float, MAX_FIFO_DEPTH> Ds;

#pragma HLS DATAFLOW
        ProcessBlock(MatrixA, OffsetA, DimM, DimN, Dd, Bs, Ds, MatrixRes, Scale1);
        Merge(MatrixRes, OffsetRes, DimM, DimN, Dd, Bs, Ds, Scale2);
    }
}