#include "hls_stream.h"
#include "types.hpp"
#include "krnl_transpose.h"
#include "u280.h"
#include <iostream>

extern "C"
{
    void KrnlTranspose(const unsigned int DimM, const unsigned int DimN, DATA_TYPE Matrix[MAX_MATRIX_SIZE], DATA_TYPE MatrixTrans[MAX_MATRIX_SIZE])
    {
        for (unsigned int IterRow = 0; IterRow < DimM; IterRow++)
        {
            for (unsigned int IterCol = 0; IterCol < DimN; IterCol++)
            {
                DATA_TYPE D = Matrix[IterRow * DimN + IterCol];
                MatrixTrans[IterCol * DimM + IterRow] = D;
            }
        }
    }
}