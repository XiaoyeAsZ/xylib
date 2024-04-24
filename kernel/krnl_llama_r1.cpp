#include "krnl_llama_r1.h"

void dot(
    pack_int8_32 *matrix_row,
    DATA_TYPE *vec,
    DATA_TYPE *res,
    unsigned int row_num,
    unsigned int col_num)
{
    pack_int8_32 mul_res;
    hls::stream<pack_int8_32> mul_s;

#pragma HLS DATAFLOW
    for (unsigned int irow = 0; irow < row_num; irow++)
    {
        for (unsigned int i = 0; i < col_num / INT8_PACK_NUM; i++)
        {
#pragma HLS PIPELINE
            pack_int8_32 r = matrix_row[irow * col_num / INT8_PACK_NUM + i];
            for (unsigned int imul = 0; imul < INT8_PACK_NUM; imul++)
                mul_res(imul * 8 + 7, imul * 8) = r[imul] * vec[i * INT8_PACK_NUM + imul];
            mul_s.write(mul_res);
        }
    }

    for (unsigned int irow = 0; irow < row_num; irow++)
    {
        ap_int<32> tmp = 0;
        for (unsigned int iadd = 0; iadd < col_num / INT8_PACK_NUM; iadd++)
        {
#pragma HLS PIPELINE
            pack_int8_32 t = mul_s.read();
            for (unsigned int ia = 0; ia < INT8_PACK_NUM; ia++)
                tmp += t(ia * 8 + 7, ia * 8);
        }
        res[irow] = tmp(7, 0);
    }
}

void gemv_qkvw_x16hbm_int8(
    pack_int8_32 *matrix_row_hbm0,
    pack_int8_32 *matrix_row_hbm1,
    pack_int8_32 *matrix_row_hbm2,
    pack_int8_32 *matrix_row_hbm3,
    pack_int8_32 *matrix_row_hbm4,
    pack_int8_32 *matrix_row_hbm5,
    pack_int8_32 *matrix_row_hbm6,
    pack_int8_32 *matrix_row_hbm7,
    pack_int8_32 *matrix_row_hbm8,
    pack_int8_32 *matrix_row_hbm9,
    pack_int8_32 *matrix_row_hbm10,
    pack_int8_32 *matrix_row_hbm11,
    pack_int8_32 *matrix_row_hbm12,
    pack_int8_32 *matrix_row_hbm13,
    pack_int8_32 *matrix_row_hbm14,
    pack_int8_32 *matrix_row_hbm15,
    pack_int8_32 *vec,
    hls::stream<pack_int8_4096> &q_s,
    pack_int8_32 *k_cache,
    pack_int8_32 *v_cache,
    unsigned int dim_m,
    unsigned int dim_n,
    bool load_vec,
    unsigned int des)
{
    DATA_TYPE vec_onchip[EMBEDDING_DIM];
    DATA_TYPE res_onchip[EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable = vec_onchip dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = res_onchip dim = 1 complete

    for (unsigned int ivec = 0; load_vec && ivec < EMBEDDING_DIM / INT8_PACK_NUM; ivec++)
    {
#pragma HLS PIPELINE
        pack_int8_32 vec_t = vec[ivec];
        for (unsigned int iunroll = 0; iunroll < INT8_PACK_NUM; iunroll++)
            vec_onchip[ivec * INT8_PACK_NUM + iunroll] = vec_t(iunroll * 8 + 7, iunroll * 8);
    }

    dot(matrix_row_hbm0, vec_onchip, &res_onchip[0 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm1, vec_onchip, &res_onchip[1 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm2, vec_onchip, &res_onchip[2 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm3, vec_onchip, &res_onchip[3 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm4, vec_onchip, &res_onchip[4 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm5, vec_onchip, &res_onchip[5 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm6, vec_onchip, &res_onchip[6 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm7, vec_onchip, &res_onchip[7 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm8, vec_onchip, &res_onchip[8 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm9, vec_onchip, &res_onchip[9 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm10, vec_onchip, &res_onchip[10 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm11, vec_onchip, &res_onchip[11 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm12, vec_onchip, &res_onchip[12 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm13, vec_onchip, &res_onchip[13 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm14, vec_onchip, &res_onchip[14 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);
    dot(matrix_row_hbm15, vec_onchip, &res_onchip[15 * dim_m / 16], dim_m / 16, EMBEDDING_DIM);

    pack_int8_32 res_pack;
    for (unsigned int ipack = 0; ipack < EMBEDDING_DIM / INT8_PACK_NUM; ipack++)
    {
#pragma HLS PIPELINE
        for (unsigned int iunroll = 0; iunroll < INT8_PACK_NUM; iunroll++)
        {
            res_pack(iunroll * 8 + 7, iunroll * 8) = res_onchip[ipack * INT8_PACK_NUM + iunroll];
        }
        if (des == 0)
        {
            q_s.write(res_pack);
        }
        else if (des == 1)
        {
            k_cache[ipack] = res_pack;
        }
        else if (des == 2)
        {
            v_cache[ipack] = res_pack;
        }
        else
            assert(0);
    }
}

extern "C"
{
    void LLAMA_R1(
        pack_int8_32 *qkvw_matrix_hbm0,
        pack_int8_32 *qkvw_matrix_hbm1,
        pack_int8_32 *qkvw_matrix_hbm2,
        pack_int8_32 *qkvw_matrix_hbm3,
        pack_int8_32 *qkvw_matrix_hbm4,
        pack_int8_32 *qkvw_matrix_hbm5,
        pack_int8_32 *qkvw_matrix_hbm6,
        pack_int8_32 *qkvw_matrix_hbm7,
        pack_int8_32 *qkvw_matrix_hbm8,
        pack_int8_32 *qkvw_matrix_hbm9,
        pack_int8_32 *qkvw_matrix_hbm10,
        pack_int8_32 *qkvw_matrix_hbm11,
        pack_int8_32 *qkvw_matrix_hbm12,
        pack_int8_32 *qkvw_matrix_hbm13,
        pack_int8_32 *qkvw_matrix_hbm14,
        pack_int8_32 *qkvw_matrix_hbm15,
        pack_int8_32 *qkvw_matrix_hbm16)
    {
        DATA_TYPE vec_onchip[EMBEDDING_DIM];
    }
}