#ifndef __LLAMA__
#define __LLAMA__

#define LAYER_NUM 32
#define HEAD_NUM 32
#define EMBEDDING_DIM 64
#define TEST_TOKEN_LEN 16
#define GEMM_MAX_MATRIX_M 4096
#define GEMM_MAX_MATRIX_N 4096
#define GEMM_MAX_MATRIX_K 4096

#include <vector>

struct TensorOnFPGA
{
    cl::Buffer Data;
    unsigned int Dim0;
    unsigned int Dim1;
};

template <typename DType>
struct TensorOnHost
{
    DType *Data;
    unsigned int Dim0;
    unsigned int Dim1;
};

// template <typename DType>
// class AttentionLayer
// {
// private:
//     unsigned int CacheLen = 0;
//     unsigned int MaxCacheLen = 16;

//     PonyTensor<DType> WQ;
//     PonyTensor<DType> WK;
//     PonyTensor<DType> WV;
//     PonyTensor<DType> KCache;
//     PonyTensor<DType> VCache;

// public:
//     AttentionLayer();
//     AttentionLayer(PonyTensor<DType> WQIn, PonyTensor<DType> WKIn, PonyTensor<DType> WVIn);
//     ~AttentionLayer();

//     PonyTensor<DType> forward(PonyTensor<DType> TokenInput, unsigned int TokenLen);
// };

// template <typename DType>
// class FFNLayer
// {
// private:
//     TensorOnHost<DType> WOnHost;
//     cl::Buffer WOnFpga;
//     cl::Buffer ResOnFPGA;

// public:
//     FFNLayer();
//     FFNLayer(TensorOnHost<DType> WIn);
//     ~FFNLayer();

//     cl::Buffer migrate(cl::Context &Ctx, cl::CommandQueue &Queue);
//     TensorOnFPGA forward(TensorOnFPGA TokenInput);
// };

// template <typename DType>
// class GemvRequest
// {
// private:
//     /* data */
// public:
//     GemvRequest(/* args */);
//     ~GemvRequest();

//     void run();
// };

template <typename DType>
class GemmRequest
{
private:
    cl::Kernel Kernel;
    cl::CommandQueue Q;

    cl::Buffer MatrixABuf;
    cl::Buffer MatrixBBuf;
    cl::Buffer MatrixResBuf;

public:
    GemmRequest(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue);
    ~GemmRequest(){};

    void run(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res, std::vector<cl::Event> &Events);
    void finish();
};

template <typename DType>
class GemmDispatch
{
private:
    std::vector<GemmRequest<DType>> Reqs;
    unsigned int ReqNumMax = 1;
    unsigned int Round = 0;

public:
    GemmDispatch(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue, unsigned int MaxReq);
    ~GemmDispatch(){};

    void request(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res, std::vector<cl::Event> &Events);
};

#endif