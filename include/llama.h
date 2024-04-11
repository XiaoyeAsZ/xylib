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

    TensorOnFPGA()
    {
        Dim0 = 0;
        dim1 = 0;
    }

    void ReleaseMem()
    {
        clReleaseMemObject(Data);
    }
};

template <typename DType>
struct TensorOnHost
{
    DType *Data;
    unsigned int Dim0;
    unsigned int Dim1;

    TensorOnHost()
    {
        Dim0 = 0;
        dim1 = 0;
    }
};

class OCLWrap
{
private:
    cl::Context Ctx;
    cl::CommandQueue Queue;

    cl::Kernel KrnlAdd;

public:
    template <typename DType>
    cl::Buffer AllocateReadBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateWriteBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateReadWriteBuffer(unsigned int Size);

    template <typename DType>
    TensorOnFPGA Mul(TensorOnFPGA Tensor0, TensorOnFPGA Tensor1);

    template <typename DType>
    TensorOnFPGA Add(TensorOnFPGA Tensor0, TensorOnFPGA Tensor1);

    template <typename DType>
    TensorOnFPGA Dot(TensorOnFPGA Tensor0, TensorOnFPGA Tensor1);

    template <typename DType>
    TensorOnFPGA Silu(TensorOnFPGA Tensor0);

    TensorOnFPGA Freq;
    void REmb(TensorOnFPGA Tensor0, TensorOnFPGA Tensor1, TensorOnFPGA Tensor2);

    void Append(TensorOnFPGA Base, TensorOnFPGA Item);

    TensorOnFPGA Trans(TensorOnFPGA Tensor0);

    TensorOnFPGA Softmax(TensorOnFPGA Tensor0);

}

template <typename DType>
class AttentionLayer
{
private:
    OCLWrap &OCL;

    std::vector<TensorOnHost<DType>> WQHost;
    std::vector<TensorOnHost<DType>> WKHost;
    std::vector<TensorOnHost<DType>> WVHost;
    std::vector<TensorOnHost<DType>> WOHost;
    std::vector<TensorOnFPGA> WQFPGA;
    std::vector<TensorOnFPGA> WKFPGA;
    std::vector<TensorOnFPGA> WVFPGA;
    std::vector<TensorOnFPGA> WOFPGA;
    std::vector<TensorOnFPGA> KCache;
    std::vector<TensorOnFPGA> VCache;

    unsigned int MaxCacheLen = 16;
    unsigned int CurLen = 0;

public:
    AttentionLayer();
    ~AttentionLayer();
};

template <typename DType>
class FeedForwardLayer
{
private:
    OCLWrap &OCL;

    TensorOnHost<DType> W0Host;
    TensorOnHost<DType> W1Host;
    TensorOnHost<DType> W2Host;
    std::vector<TensorOnFPGA> W0FPGA;
    std::vector<TensorOnFPGA> W1FPGA;
    std::vector<TensorOnFPGA> W2FPGA;

public:
    FeedForwardLayer();
    ~FeedForwardLayer();

    TensorOnFPGA operator()(TensorOnFPGA Input);
};

template <typename DType>
class RMSNormLayer
{
private:
public:
    RMSNormLayer();
    ~RMSNormLayer();
};

template <typename DType>
class TransformerBlock
{
private:
    OCLWrap &OCL;

    AttentionLayer<DType> Attn;
    FeedForwardLayer<DType> FFN;
    RMSNormLayer<DType> NormAttn;
    RMSNormLayer<DType> NormFFN;

public:
    TransformerBlock();
    ~TransformerBlock();

    TensorOnFPGA operator()(TensorOnFPGA Input);
};

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
    std::vector<cl::Event> Events;
    cl::Buffer MatrixABuf;
    cl::Buffer MatrixBBuf;
    cl::Buffer MatrixResBuf;

public:
    GemmRequest(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue);
    ~GemmRequest(){};

    void run(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res);
    void finish();
};

class KrnlDispatch
{
public:
    virtual void request() = 0;
};

template <typename DType>
class GemmDispatch : public KrnlDispatch
{
private:
    std::vector<GemmRequest<DType>> Reqs;
    unsigned int ReqNumMax = 1;
    unsigned int Round = 0;

public:
    GemmDispatch(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue, unsigned int MaxReq);
    ~GemmDispatch(){};

    void request(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res);
};

#endif