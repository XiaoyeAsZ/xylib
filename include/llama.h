#ifndef __LLAMA__
#define __LLAMA__

#define LAYER_NUM 1
#define HEAD_NUM 32
#define EMBEDDING_DIM 4096
#define HEAD_DIM 128
#define HIDDEN_DIM 11008
#define TEST_TOKEN_LEN 16
#define GEMM_MAX_MATRIX_M 4096
#define GEMM_MAX_MATRIX_N 4096
#define GEMM_MAX_MATRIX_K 4096

#define MAX_TOKEN_LEN 2048

// #define RANDOM_INPUT
#define DEBUG

#include <vector>

struct TensorOnFPGA
{
    cl::Buffer Data;
    unsigned int Offset;
    unsigned int Dim0;
    unsigned int Dim1;

    float Scale = 0;

    TensorOnFPGA()
    {
        Offset = 0;
        Dim0 = 0;
        Dim1 = 0;
    }

    TensorOnFPGA(cl::Buffer Data, unsigned int Offset, unsigned int Dim0, unsigned int Dim1)
    {
        this->Data = Data;
        this->Offset = Offset;
        this->Dim0 = Dim0;
        this->Dim1 = Dim1;
    }

    // void ReleaseMem()
    // {
    //     clReleaseMemObject(Data);
    // }

    TensorOnFPGA SubTensorRow(unsigned int Start, unsigned int Len)
    {
        assert(Len % Dim1 == 0);
        return TensorOnFPGA(Data, Start, Len / Dim1, Dim1);
    }
};

template <typename DType>
struct TensorOnHost
{
    DType *Data;
    unsigned int Dim0;
    unsigned int Dim1;

    float Scale = 0;

    TensorOnHost()
    {
        Dim0 = 0;
        Dim1 = 0;
    }

    TensorOnHost(DType *Data, unsigned int Dim0, unsigned int Dim1)
    {
        this->Data = Data;
        this->Dim0 = Dim0;
        this->Dim1 = Dim1;
    }

    TensorOnHost SubTensorRow(unsigned int Start, unsigned int Len)
    {
        assert(Len % Dim1 == 0);
        return TensorOnHost(&Data[Start], Len / Dim1, Dim1);
    }
};

struct QuantPack
{
    float Scale0;
    float Scale1;
    float Scale2;

    QuantPack(float s0, float s1, float s2)
    {
        this->Scale0 = s0;
        this->Scale1 = s1;
        this->Scale2 = s2;
    }
};

class OCLWrap
{
private:
    cl::Context Ctx;
    cl::Program Prog;
    cl::CommandQueue Queue;

    cl::Kernel KrnlGemm;
    cl::Kernel KrnlAddVec;
    cl::Kernel KrnlAddMat;
    cl::Kernel KrnlDotVec;
    cl::Kernel KrnlDotMat;
    cl::Kernel KrnlSilu;
    cl::Kernel KrnlREmb;
    cl::Kernel KrnlMove;
    cl::Kernel KrnlTranspose;
    cl::Kernel KrnlSoftmax;
    cl::Kernel KrnlRMSNorm;

    cl::Kernel KrnlGemv;

public:
    OCLWrap(cl::Context &Ctx, cl::Program Prog, cl::CommandQueue &Queue, std::ifstream &F);
    ~OCLWrap();

    template <typename DType>
    cl::Buffer AllocateReadBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateWriteBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateReadWriteBuffer(unsigned int Size);

    template <typename DType>
    void Map(TensorOnHost<DType> &Host, TensorOnFPGA &Device);

    template <typename DType>
    void Map(TensorOnFPGA &Device, TensorOnHost<DType> &Host);

    template <typename DType>
    TensorOnFPGA Mul(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent, float Scale);

    template <typename DType>
    TensorOnFPGA Add(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent, float Scale);

    template <typename DType>
    TensorOnFPGA Dot(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent, float Scale);

    template <typename DType>
    TensorOnFPGA Silu(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent, float Scale);

    TensorOnHost<ap_int<8>> freq;
    TensorOnFPGA Freq;
    template <typename DType>
    TensorOnFPGA REmb(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent, float ResScale);

    template <typename DType>
    void Move(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent);

    template <typename DType>
    TensorOnFPGA Trans(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent);

    template <typename DType>
    TensorOnFPGA Softmax(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent, float ResScale);

    template <typename DType>
    TensorOnFPGA RMSNorm(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events, cl::Event &RunEvent, float ResScale);

    void load_freq(std::ifstream &F);
};

template <typename DType>
class AttentionLayer
{
private:
    OCLWrap &OCL;

    std::vector<TensorOnHost<DType>> WQHost; // 32 * 4096 * 128
    std::vector<TensorOnHost<DType>> WKHost; // 32 * 4096 * 128
    std::vector<TensorOnHost<DType>> WVHost; // 32 * 4096 * 128
    TensorOnHost<DType> WOHost;              //  4096 * 4096

    std::vector<TensorOnFPGA> WQFPGA;
    std::vector<TensorOnFPGA> WKFPGA;
    std::vector<TensorOnFPGA> WVFPGA;
    TensorOnFPGA WOFPGA;
    std::vector<TensorOnFPGA> KCache;
    std::vector<TensorOnFPGA> VCache;

    unsigned int MaxCacheLen = 1024;
    unsigned int CurLen = 0;

    std::vector<QuantPack> OpQuantS;

public:
    AttentionLayer(OCLWrap &OCL);
    ~AttentionLayer();

    void load_weight(std::ifstream &F);
    void load_quant(std::ifstream &F);
    void migrate();
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class FeedForwardLayer
{
private:
    OCLWrap &OCL;

    TensorOnHost<DType> W0Host; // 4096 * 11008
    TensorOnHost<DType> W1Host; // 11008 * 4096
    TensorOnHost<DType> W2Host; // 4096 * 11008

    TensorOnFPGA W0FPGA;
    TensorOnFPGA W1FPGA;
    TensorOnFPGA W2FPGA;

    std::vector<QuantPack> OpQuantS;

public:
    FeedForwardLayer(OCLWrap &OCL);
    ~FeedForwardLayer();

    void load_weight(std::ifstream &F);
    void load_quant(std::ifstream &F);
    void migrate();
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class RMSNormLayer
{
private:
    OCLWrap &OCL;

    TensorOnHost<DType> WeightHost;
    TensorOnFPGA Weight;

public:
    RMSNormLayer(OCLWrap &OCL);
    ~RMSNormLayer();

    void load_weight(std::ifstream &F);
    void load_quant(std::ifstream &F);
    void migrate();
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events, float Scale);
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

    std::vector<QuantPack> OpQuantS;

public:
    TransformerBlock(OCLWrap &OCL);
    ~TransformerBlock();

    void load_weight(std::ifstream &F);
    void load_quant(std::ifstream &F);
    void migrate();
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class Transformer
{
private:
    OCLWrap &OCL;

    std::vector<TransformerBlock<DType>> Blocks;

public:
    Transformer(OCLWrap &OCL);
    ~Transformer();

    void load_weight(std::ifstream &F);
    void load_quant(std::ifstream &F);
    void migrate();
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

#endif