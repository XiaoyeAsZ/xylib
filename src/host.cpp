#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "ap_int.h"

#include "xcl2.hpp"
#include "cmdlineparser.h"
#include "llama.h"

using namespace sda;
using namespace sda::utils;

template <typename DType>
DType *GenerateRandomInput(unsigned int Len)
{
  DType *GeneratedInput = new DType[Len];
  for (unsigned int IterRnd = 0; IterRnd < Len; IterRnd++)
    GeneratedInput[IterRnd] = DType(int(float(rand() * 2 - RAND_MAX) / RAND_MAX * 127));
  return GeneratedInput;
}

template <typename DType>
TensorOnHost<DType> operator*(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim1 == MB.Dim0);
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MB.Dim1], MA.Dim0, MB.Dim1);
  for (int i = 0; i < MA.Dim0; i++)
  {
    for (int j = 0; j < MB.Dim1; j++)
    {
      ap_int<32> tmp = 0;
      for (int k = 0; k < MA.Dim1; k++)
        tmp += MA.Data[i * MA.Dim1 + k] * MB.Data[k * MB.Dim1 + j];
      Res.Data[i * MB.Dim1 + j] = int(tmp / 127.0);
    }
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> operator+(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim0 == MB.Dim0 && MA.Dim1 == MB.Dim1);
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MB.Dim1], MA.Dim0, MB.Dim1);
  for (int i = 0; i < MA.Dim0; i++)
  {
    for (int j = 0; j < MB.Dim1; j++)
    {
      Res.Data[i * MB.Dim1 + j] = MA.Data[i * MB.Dim1 + j] + MB.Data[i * MB.Dim1 + j];
    }
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> SiluTest(TensorOnHost<DType> &MA)
{
  assert(MA.Dim1 != 0);
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MA.Dim1], MA.Dim0, MA.Dim1);
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i++)
  {
    DType x = MA.Data[i];
    Res.Data[i] = (x * (1.0 / (1.0 + exp(float(-x) / 127.0)))) * 127;
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> DotTest(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim0 == MB.Dim0 && MA.Dim1 == MB.Dim1);
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MA.Dim1], MA.Dim0, MA.Dim1);
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i++)
  {
    ap_int<32> tmp = MA.Data[i] * MB.Data[i];
    Res.Data[i] = int(tmp / 127.0);
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> REmbTest(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim0 == MB.Dim0 && MA.Dim1 == MB.Dim1);
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MA.Dim1], MA.Dim0, MA.Dim1);
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i += 2)
  {
    ap_int<32> T1 = MA.Data[i] * MB.Data[i] - MA.Data[i + 1] * MB.Data[i + 1];
    Res.Data[i] = int(T1 / 127.0);
    ap_int<32> T2 = MA.Data[i] * MB.Data[i + 1] + MA.Data[i + 1] * MB.Data[i];
    Res.Data[i + 1] = int(T2 / 127.0);
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> TransTest(TensorOnHost<DType> &MA)
{
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MA.Dim1], MA.Dim1, MA.Dim0);
  for (int i = 0; i < MA.Dim0; i++)
  {
    for (int j = 0; j < MA.Dim1; j++)
      Res.Data[j * MA.Dim0 + i] = MA.Data[i * MA.Dim1 + j];
  }
  return Res;
}

template <typename DType>
TensorOnHost<DType> SoftmaxTest(TensorOnHost<DType> &MA)
{
  TensorOnHost<DType> Res(new DType[MA.Dim0 * MA.Dim1], MA.Dim0, MA.Dim1);
  for (int i = 0; i < MA.Dim0; i++)
  {
    float tmp = 0;
    for (int j = 0; j < MA.Dim1; j++)
    {
      tmp += exp(float(MA.Data[i * MA.Dim1 + j]) / 127.0);
    }
    for (int j = 0; j < MA.Dim1; j++)
    {
      Res.Data[i * MA.Dim1 + j] = int(exp(float(MA.Data[i * MA.Dim1 + j]) / 127.0) / tmp * 127);
    }
  }
  return Res;
}

template <typename DType>
bool operator==(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim0 == MB.Dim0 && MA.Dim1 == MB.Dim1);
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i++)
  {
    // printf("%d : %d\n", MA.Data[i], MB.Data[i]);
    if (MA.Data[i] != MB.Data[i])
    {
      std::cout<< MA.Data[i]<<" differ "<< MB.Data[i] <<std::endl;
      return 0;
    }
  }
  return 1;
}

/**
 * Class OCLWrap
 *
 */

OCLWrap::OCLWrap(cl::Context &Ctx, cl::Program Prog, cl::CommandQueue &Queue)
{
  cl_int err;

  this->Ctx = Ctx;
  this->Prog = Prog;
  this->Queue = Queue;

  OCL_CHECK(err, this->KrnlGemm = cl::Kernel(this->Prog, "KrnlGemm", &err));
  OCL_CHECK(err, this->KrnlDotMat = cl::Kernel(this->Prog, "KrnlDotMat", &err));
  OCL_CHECK(err, this->KrnlSilu = cl::Kernel(this->Prog, "KrnlSilu", &err));
  OCL_CHECK(err, this->KrnlREmb = cl::Kernel(this->Prog, "KrnlREmb", &err));
  OCL_CHECK(err, this->KrnlMove = cl::Kernel(this->Prog, "KrnlMove", &err));
  OCL_CHECK(err, this->KrnlTranspose = cl::Kernel(this->Prog, "KrnlTranspose", &err));
  OCL_CHECK(err, this->KrnlSoftmax = cl::Kernel(this->Prog, "KrnlSoftmax", &err));
  OCL_CHECK(err, this->KrnlAddMat = cl::Kernel(this->Prog, "KrnlAddMat", &err));

  OCL_CHECK(err, this->KrnlGemv = cl::Kernel(this->Prog, "KrnlGemv", &err));

  this->freq = TensorOnHost<ap_int<8>>(GenerateRandomInput<ap_int<8>>(MAX_TOKEN_LEN * HEAD_DIM), MAX_TOKEN_LEN, HEAD_DIM);
  this->Freq = TensorOnFPGA(this->AllocateReadBuffer<ap_int<8>>(MAX_TOKEN_LEN * HEAD_DIM), 0, MAX_TOKEN_LEN, HEAD_DIM);
  this->Map(freq, this->Freq);
};

OCLWrap::~OCLWrap(){};

unsigned int totalmem = 0;

template <typename DType>
cl::Buffer OCLWrap::AllocateReadBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;

  std::cout << "try to allocate: " << (Size) * sizeof(DType) / 1000000 << " MB RBuf" << std::endl;
  cl_mem_ext_ptr_t ext = {0};
  ext.banks = 0 | XCL_MEM_TOPOLOGY;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, (Size) * sizeof(DType), &ext, &err));
  OCL_CHECK(err, err = this->Queue.enqueueMigrateMemObjects({Buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  totalmem += (Size) * sizeof(DType) / 1000000;
  std::cout << "total: " << totalmem << std::endl;
  this->Queue.finish();
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  std::cout << "try to allocate: " << (Size) * sizeof(DType) / 1000000 << " MB WBuf" << std::endl;
  cl_mem_ext_ptr_t ext = {0};
  ext.banks = 0 | XCL_MEM_TOPOLOGY;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, (Size) * sizeof(DType), &ext, &err));
  OCL_CHECK(err, err = this->Queue.enqueueMigrateMemObjects({Buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  totalmem += (Size) * sizeof(DType) / 1000000;
  std::cout << "total: " << totalmem << std::endl;
  this->Queue.finish();
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateReadWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;

  std::cout << "try to allocate: " << (Size) * sizeof(DType) / 1000000 << " MB RWBuf" << std::endl;
  cl_mem_ext_ptr_t ext = {0};
  ext.banks = 0 | XCL_MEM_TOPOLOGY;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, (Size) * sizeof(DType), &ext, &err));
  OCL_CHECK(err, err = this->Queue.enqueueMigrateMemObjects({Buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  totalmem += (Size) * sizeof(DType) / 1000000;
  std::cout << "total: " << totalmem << std::endl;
  this->Queue.finish();
  return Buf;
};

template <typename DType>
void OCLWrap::Map(TensorOnHost<DType> &Host, TensorOnFPGA &Device)
{
  assert(Host.Dim0 == Device.Dim0 && Host.Dim1 == Device.Dim1);

  cl_int err;
  cl::Event Event;
  OCL_CHECK(err, err = this->Queue.enqueueWriteBuffer(Device.Data, CL_FALSE, 0,
                                                      (Host.Dim0 * Host.Dim1) * sizeof(DType),
                                                      Host.Data, nullptr, &Event));
  Event.wait();
};

template <typename DType>
void OCLWrap::Map(TensorOnFPGA &Device, TensorOnHost<DType> &Host)
{
  assert(Host.Dim0 == Device.Dim0 && Host.Dim1 == Device.Dim1);

  cl_int err;
  cl::Event Event;
  OCL_CHECK(err, err = this->Queue.enqueueReadBuffer(Device.Data, CL_FALSE, 0,
                                                     (Host.Dim0 * Host.Dim1) * sizeof(DType),
                                                     Host.Data, nullptr, &Event));
  Event.wait();
};

/**
 * TODO: MUL VV, VM, MV
 */
template <typename DType>
TensorOnFPGA OCLWrap::Mul(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                          std::vector<cl::Event> &Events, cl::Event &RunEvent)
{

  cl_int err;

  assert(Tensor0.Dim1 != 0 && Tensor1.Dim1 != 0 && Tensor0.Dim1 == Tensor1.Dim0);

  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor1.Dim1), 0, Tensor0.Dim0, Tensor1.Dim1);

  if (Tensor0.Dim0 == 1)
  {
    std::cout << "real vec mul\n";
    cl::Event Te;
    TensorOnFPGA T = this->Trans<ap_int<8>>(Tensor1, Events, Te);

    Events.push_back(Te);
    Events.back().wait();
    std::cout << "trans finish\n";

    TensorOnFPGA V = Tensor0;
    V.Dim0 = V.Dim1;
    V.Dim1 = 1;
    this->KrnlGemv.setArg(0, T.Data);
    this->KrnlGemv.setArg(1, V.Data);
    this->KrnlGemv.setArg(2, Buf.Data);
    this->KrnlGemv.setArg(3, T.Dim0);
    this->KrnlGemv.setArg(4, T.Dim1);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlGemv, &Events, &RunEvent));
  }
  else
  {
    this->KrnlGemm.setArg(0, Tensor0.Dim0);
    this->KrnlGemm.setArg(1, Tensor0.Dim1);
    this->KrnlGemm.setArg(2, Tensor1.Dim1);
    this->KrnlGemm.setArg(3, Tensor0.Data);
    this->KrnlGemm.setArg(4, Tensor1.Data);
    this->KrnlGemm.setArg(5, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlGemm, &Events, &RunEvent));
  }

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Add(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                          std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  assert(Tensor0.Dim0 == Tensor1.Dim0 && Tensor0.Dim1 == Tensor1.Dim1);

  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim0, Tensor0.Dim1);

  if (Tensor0.Dim1 == 0)
  {
    this->KrnlAddVec.setArg(0, Tensor0.Dim0);
    this->KrnlAddVec.setArg(1, Tensor0.Dim1);
    this->KrnlAddVec.setArg(2, Tensor0.Data);
    this->KrnlAddVec.setArg(3, Tensor1.Data);
    this->KrnlAddVec.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlAddVec, &Events, &RunEvent));
  }
  else
  {
    this->KrnlAddMat.setArg(0, Tensor0.Data);
    this->KrnlAddMat.setArg(1, Tensor0.Offset);
    this->KrnlAddMat.setArg(2, Tensor1.Data);
    this->KrnlAddMat.setArg(3, Tensor1.Offset);
    this->KrnlAddMat.setArg(4, Tensor0.Dim0);
    this->KrnlAddMat.setArg(5, Tensor0.Dim1);
    this->KrnlAddMat.setArg(6, Buf.Data);
    this->KrnlAddMat.setArg(7, Buf.Offset);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlAddMat, &Events, &RunEvent));
  }

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Dot(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                          std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  assert(Tensor0.Dim0 == Tensor1.Dim0 && Tensor0.Dim1 == Tensor1.Dim1);

  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim0, Tensor0.Dim1);

  if (Tensor0.Dim1 == 0)
  {
    this->KrnlDotVec.setArg(0, Tensor0.Dim0);
    this->KrnlDotVec.setArg(1, Tensor0.Dim1);
    this->KrnlDotVec.setArg(2, Tensor0.Data);
    this->KrnlDotVec.setArg(3, Tensor1.Data);
    this->KrnlDotVec.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlDotVec, &Events, &RunEvent));
  }
  else
  {
    this->KrnlDotMat.setArg(0, Tensor0.Data);
    this->KrnlDotMat.setArg(1, Tensor0.Offset);
    this->KrnlDotMat.setArg(2, Tensor1.Data);
    this->KrnlDotMat.setArg(3, Tensor1.Offset);
    this->KrnlDotMat.setArg(4, Tensor0.Dim0);
    this->KrnlDotMat.setArg(5, Tensor0.Dim1);
    this->KrnlDotMat.setArg(6, Buf.Data);
    this->KrnlDotMat.setArg(7, Buf.Offset);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlDotMat, &Events, &RunEvent));
  }

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Silu(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim0, Tensor0.Dim1);

  this->KrnlSilu.setArg(0, Tensor0.Data);
  this->KrnlSilu.setArg(1, Tensor0.Offset);
  this->KrnlSilu.setArg(2, Tensor0.Dim0);
  this->KrnlSilu.setArg(3, Tensor0.Dim1);
  this->KrnlSilu.setArg(4, Buf.Data);
  this->KrnlSilu.setArg(5, Buf.Offset);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlSilu, &Events, &RunEvent));

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::REmb(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                           std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim0, Tensor0.Dim1);

  assert(Tensor0.Dim0 == Tensor1.Dim0);
  this->KrnlREmb.setArg(0, Tensor0.Data);
  this->KrnlREmb.setArg(1, Tensor0.Offset);
  this->KrnlREmb.setArg(2, Tensor0.Dim0);
  this->KrnlREmb.setArg(3, Tensor0.Dim1);
  this->KrnlREmb.setArg(4, Tensor1.Data);
  this->KrnlREmb.setArg(5, Tensor1.Offset);
  this->KrnlREmb.setArg(6, Buf.Data);
  this->KrnlREmb.setArg(7, Buf.Offset);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlREmb, &Events, &RunEvent));
  return Buf;
}

template <typename DType>
void OCLWrap::Move(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                   std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;

  unsigned int MoveLen;
  if (Tensor0.Dim1 == 0)
    MoveLen = Tensor0.Dim0;
  else
    MoveLen = Tensor0.Dim0 * Tensor0.Dim1;

  assert(MoveLen <= Tensor1.Dim0 * Tensor1.Dim1);

  this->KrnlMove.setArg(0, Tensor0.Data);
  this->KrnlMove.setArg(1, Tensor0.Offset);
  this->KrnlMove.setArg(2, Tensor1.Data);
  this->KrnlMove.setArg(3, Tensor1.Offset);
  this->KrnlMove.setArg(4, MoveLen);

  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlMove, &Events, &RunEvent));
};

template <typename DType>
TensorOnFPGA OCLWrap::Trans(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim1, Tensor0.Dim0);

  this->KrnlTranspose.setArg(0, Tensor0.Dim0);
  this->KrnlTranspose.setArg(1, Tensor0.Dim1);
  this->KrnlTranspose.setArg(2, Tensor0.Data);
  this->KrnlTranspose.setArg(3, Buf.Data);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlTranspose, &Events, &RunEvent));

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Softmax(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf(this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), 0, Tensor0.Dim0, Tensor0.Dim1);

  this->KrnlSoftmax.setArg(0, Tensor0.Data);
  this->KrnlSoftmax.setArg(1, 0);
  this->KrnlSoftmax.setArg(2, Tensor0.Dim0);
  this->KrnlSoftmax.setArg(3, Tensor0.Dim1);
  this->KrnlSoftmax.setArg(4, Buf.Data);
  this->KrnlSoftmax.setArg(5, 0);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlSoftmax, &Events, &RunEvent));

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::RMSNorm(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

  this->KrnlRMSNorm.setArg(0, Tensor0.Data);
  this->KrnlRMSNorm.setArg(1, Tensor0.Offset);
  this->KrnlRMSNorm.setArg(2, Tensor0.Dim0);
  this->KrnlRMSNorm.setArg(3, Tensor0.Dim1);
  this->KrnlRMSNorm.setArg(4, Buf.Data);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlRMSNorm, &Events, &RunEvent));

  return Buf;
};

/**
 * Class AttentionLayer
 */

template <typename DType>
AttentionLayer<DType>::AttentionLayer(OCLWrap &OCL) : OCL(OCL)
{

  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    this->WQHost.push_back(TensorOnHost<DType>(nullptr, EMBEDDING_DIM, HEAD_DIM));
    this->WKHost.push_back(TensorOnHost<DType>(nullptr, EMBEDDING_DIM, HEAD_DIM));
    this->WVHost.push_back(TensorOnHost<DType>(nullptr, EMBEDDING_DIM, HEAD_DIM));

    this->WQFPGA.push_back(
        TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(EMBEDDING_DIM * HEAD_DIM), 0, EMBEDDING_DIM, HEAD_DIM));
    this->WKFPGA.push_back(
        TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(EMBEDDING_DIM * HEAD_DIM), 0, EMBEDDING_DIM, HEAD_DIM));
    this->WVFPGA.push_back(
        TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(EMBEDDING_DIM * HEAD_DIM), 0, EMBEDDING_DIM, HEAD_DIM));

    this->KCache.push_back(
        TensorOnFPGA(this->OCL.AllocateReadWriteBuffer<DType>(MAX_TOKEN_LEN * HEAD_DIM), 0, MAX_TOKEN_LEN, HEAD_DIM));
    this->VCache.push_back(
        TensorOnFPGA(this->OCL.AllocateReadWriteBuffer<DType>(MAX_TOKEN_LEN * HEAD_DIM), 0, MAX_TOKEN_LEN, HEAD_DIM));
  }
  std::cout << "allocate weigth finish\n";
  this->WOHost = TensorOnHost<DType>(nullptr, EMBEDDING_DIM, EMBEDDING_DIM);
  std::cout << "allocate init finish\n";
  this->WOFPGA =
      TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(EMBEDDING_DIM * EMBEDDING_DIM), 0, EMBEDDING_DIM, EMBEDDING_DIM);
  std::cout << "allocate o finish\n";
}

template <typename DType>
AttentionLayer<DType>::~AttentionLayer() {}

template <typename DType>
void AttentionLayer<DType>::load(std::ifstream &F)
{

  std::cout << "Loading: <AttentionLayer>\n";

#ifdef RANDOM_INPUT
  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    this->WQHost[IterHead].Data = GenerateRandomInput<DType>(this->WQHost[IterHead].Dim0 * this->WQHost[IterHead].Dim1);
    this->WKHost[IterHead].Data = GenerateRandomInput<DType>(this->WKHost[IterHead].Dim0 * this->WKHost[IterHead].Dim1);
    this->WVHost[IterHead].Data = GenerateRandomInput<DType>(this->WVHost[IterHead].Dim0 * this->WVHost[IterHead].Dim1);
  }
  this->WOHost.Data = GenerateRandomInput<DType>(this->WOHost.Dim0 * this->WOHost.Dim1);
#else

  this->WQHost[IterHead].Data = new DType[this->WQHost[IterHead].Dim0 * this->WQHost[IterHead].Dim1];
  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {

    for (unsigned int IterRead = 0; IterRead < this->WQHost[IterHead].Dim0 * this->WQHost[IterHead].Dim1; IterRead++)
      F >> this->WQHost[IterHead].Data[IterRead];
  }

  this->WKHost[IterHead].Data = new DType[this->WKHost[IterHead].Dim0 * this->WKHost[IterHead].Dim1];
  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    for (unsigned int IterRead = 0; IterRead < this->WKHost[IterHead].Dim0 * this->WKHost[IterHead].Dim1; IterRead++)
      F >> this->WKHost[IterHead].Data[IterRead];
  }

  this->WVHost[IterHead].Data = new DType[this->WVHost[IterHead].Dim0 * this->WVHost[IterHead].Dim1];
  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    for (unsigned int IterRead = 0; IterRead < this->WVHost[IterHead].Dim0 * this->WVHost[IterHead].Dim1; IterRead++)
      F >> this->WVHost[IterHead].Data[IterRead];
  }

  this->WOHost.Data = new DType[this->WOHost.Dim0 * this->WOHost.Dim1];
  for (unsigned int IterRead = 0; IterRead < this->WOHost.Dim0 * this->WOHost.Dim1; IterRead++)
    F >> this->WOHost.Data[IterRead];

#endif
}

template <typename DType>
void AttentionLayer<DType>::migrate()
{
#ifdef DEBUG
  std::cout << "Migrating: <AttentionLayer> ...\n";
#endif
  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    this->OCL.Map(this->WQHost[IterHead], this->WQFPGA[IterHead]);
    this->OCL.Map(this->WKHost[IterHead], this->WKFPGA[IterHead]);
    this->OCL.Map(this->WVHost[IterHead], this->WVFPGA[IterHead]);
  }

  this->OCL.Map(this->WOHost, this->WOFPGA);

#ifdef DEBUG
  std::cout << "Migrating: <AttentionLayer> Finish\n";
#endif
}

template <typename DType>
TensorOnFPGA AttentionLayer<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  std::vector<TensorOnFPGA> Heads;
  std::vector<cl::Event> Tail;

  

  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    std::vector<cl::Event> EventsInHead;
    cl::Event RunEvent;

    TensorOnFPGA H0 = this->OCL.Mul<DType>(Input, this->WQFPGA[IterHead], Events, RunEvent);
    EventsInHead.push_back(RunEvent);
#ifdef DEBUG
    std::cout << "ATTN Mul (Input * WQ)\n";
    EventsInHead.back().wait();
    TensorOnHost<DType> x0(new DType[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
    OCL.Map(Input, x0);
    TensorOnHost<DType> y0(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
    OCL.Map(H0, y0);
    TensorOnHost<DType> r0 = x0 * this->WQHost[IterHead];
    std::cout << (r0 == y0 ? "ATTN Mul Mul (Input * WQ) PASS!!!\n" : "ATTN Mul Mul (Input * WQ) ERROR!!!\n");
#endif

    TensorOnFPGA H1 = this->OCL.Mul<DType>(Input, this->WKFPGA[IterHead], EventsInHead, RunEvent);
    EventsInHead.push_back(RunEvent);
#ifdef DEBUG
    std::cout << "ATTN Mul (Input * WK)\n";
    EventsInHead.back().wait();
    TensorOnHost<DType> y1(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
    OCL.Map(H1, y1);
    TensorOnHost<DType> r1 = x0 * this->WKHost[IterHead];
    std::cout << (r1 == y1 ? "ATTN Mul (Input * WK) PASS!!!\n" : "ATTN Mul (Input * WK) ERROR!!!\n");
#endif

    TensorOnFPGA H2 = this->OCL.Mul<DType>(Input, this->WVFPGA[IterHead], EventsInHead, RunEvent);
    EventsInHead.push_back(RunEvent);
#ifdef DEBUG
    std::cout << "ATTN Mul (Input * WV)\n";
    EventsInHead.back().wait();
    TensorOnHost<DType> y2(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
    OCL.Map(H2, y2);
    TensorOnHost<DType> r2 = x0 * this->WVHost[IterHead];
    std::cout << (r2 == y2 ? "ATTN Mul (Input * WV) PASS!!!\n" : "ATTN Mul (Input * WV) ERROR!!!\n");
#endif

    TensorOnFPGA H3t = this->OCL.Freq.SubTensorRow(0, Input.Dim0 * HEAD_DIM);
    TensorOnFPGA H3 = this->OCL.REmb<DType>(H0, H3t, EventsInHead, RunEvent);
    EventsInHead.push_back(RunEvent);
#ifdef DEBUG
    std::cout << "ATTN REmb (Q)\n";
    EventsInHead.back().wait();
    TensorOnHost<DType> y3(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
    OCL.Map(H3, y3);
    TensorOnHost<DType> r3t = this->OCL.freq.SubTensorRow(0, Input.Dim0 * HEAD_DIM);
    TensorOnHost<DType> r3 = REmbTest(r0, r3t);
    std::cout << (r3 == y3 ? "ATTN REmb (Q) PASS!!!\n" : "ATTN REmb (Q) ERROR!!!\n");
#endif

    TensorOnFPGA H4t = this->OCL.Freq.SubTensorRow(0, Input.Dim0 * HEAD_DIM);
    TensorOnFPGA H4 = this->OCL.REmb<DType>(H1, H4t, EventsInHead, RunEvent);
    EventsInHead.push_back(RunEvent);
#ifdef DEBUG
    std::cout << "ATTN REmb (K)\n";
    EventsInHead.back().wait();
    TensorOnHost<DType> y4(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
    OCL.Map(H4, y4);
    TensorOnHost<DType> r4t = this->OCL.freq.SubTensorRow(0, Input.Dim0 * HEAD_DIM);
    TensorOnHost<DType> r4 = REmbTest(r1, r4t);
    std::cout << (r4 == y4 ? "ATTN REmb (K) PASS!!!\n" : "ATTN REmb (K) ERROR!!!\n");
#endif

        assert(this->CurLen + H3.Dim0 <= this->MaxCacheLen);
        assert(this->CurLen + H4.Dim0 <= this->MaxCacheLen);

        TensorOnFPGA tmp = this->KCache[IterHead].SubTensorRow(this->CurLen * HEAD_DIM, H3.Dim0 * H3.Dim1);
        this->OCL.Move<DType>(H3, tmp, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Move (KCache)\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y5(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
        OCL.Map(tmp, y5);
        std::cout << (y3 == y5 ? "ATTN Move (KCache) PASS!!!\n" : "ATTN Move (KCache) ERROR!!!\n");
    #endif

        tmp = this->VCache[IterHead].SubTensorRow(this->CurLen * HEAD_DIM, H4.Dim0 * H4.Dim1);
        this->OCL.Move<DType>(H4, tmp, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Move (VCache)\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y6(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
        OCL.Map(tmp, y6);
        std::cout << (y4 == y6 ? "ATTN Move (VCache) PASS!!!\n" : "ATTN Move (VCache) ERROR!!!\n");
    #endif

        TensorOnFPGA H5 = this->OCL.Trans<DType>(H4, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Trans (K)\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y7(new DType[Input.Dim0 * HEAD_DIM], HEAD_DIM, Input.Dim0);
        OCL.Map(H5, y7);
        TensorOnHost<DType> r7 = TransTest(r4);

    //     // for (int i = 0; i < 16; i++)
    //     // {
    //     //   for (int j = 0; j < HEAD_DIM; j++)
    //     //     std::cout << y7.Data[i * HEAD_DIM + j] << " ";
    //     //   std::cout << std::endl;
    //     // }

    //     // std::cout << std::endl;

    //     // for (int i = 0; i < 16; i++)
    //     // {
    //     //   for (int j = 0; j < HEAD_DIM; j++)
    //     //     std::cout << r7.Data[i * HEAD_DIM + j] << " ";
    //     //   std::cout << std::endl;
    //     // }
    //     std::cout << (r7 == y7 ? "ATTN Trans (K) PASS!!!\n" : "ATTN Trans (K) ERROR!!!\n");
    #endif

        TensorOnFPGA H6 = this->OCL.Mul<DType>(H3, H5, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Mul (Q K^T)\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y8(new DType[Input.Dim0 * Input.Dim0], Input.Dim0, Input.Dim0);
        OCL.Map(H6, y8);
        TensorOnHost<DType> r8 = r3 * r7;
        std::cout << (r8 == y8 ? "ATTN Mul (Q K^T) PASS!!!\n" : "ATTN Mul (Q K^T) ERROR!!!\n");
    #endif

        TensorOnFPGA H7 = this->OCL.Softmax<DType>(H6, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Softmax\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y9(new DType[Input.Dim0 * Input.Dim0], Input.Dim0, Input.Dim0);
        OCL.Map(H7, y9);
        TensorOnHost<DType> r9 = SoftmaxTest(r8);
        std::cout << (r9 == y9 ? "ATTN Softmax PASS!!!\n" : "ATTN Softmax ERROR!!!\n");
    #endif

        TensorOnFPGA H8 = this->OCL.Mul<DType>(H7, H2, EventsInHead, RunEvent);
        EventsInHead.push_back(RunEvent);
    #ifdef DEBUG
        std::cout << "ATTN Mul (score V)\n";
        EventsInHead.back().wait();
        TensorOnHost<DType> y10(new DType[Input.Dim0 * HEAD_DIM], Input.Dim0, HEAD_DIM);
        OCL.Map(H8, y10);
        TensorOnHost<DType> r10 = r9 * r2;
        std::cout << (r10 == y10 ? "ATTN Mul (score V) PASS!!!\n" : "ATTN Mul (score V) ERROR!!!\n");
    #endif

        Tail.push_back(RunEvent);

        Heads.push_back(H8);
  }

    assert(Tail.size() == HEAD_NUM);
    for (unsigned int IterSup = 0; IterSup < HEAD_NUM; IterSup++)
      Events.push_back(Tail[IterSup]);

    this->CurLen += Input.Dim0;

    assert(Heads.size() == HEAD_NUM);

    cl::Event RunEvent;

    TensorOnFPGA tmp = this->WOFPGA.SubTensorRow(0, HEAD_DIM * EMBEDDING_DIM);
    TensorOnFPGA Sum = this->OCL.Mul<DType>(Heads[0], tmp, Events, RunEvent);
    // Events.push_back(RunEvent);
    // #ifdef DEBUG
    //   std::cout << "ATTN Mul (output)\n";
    //   Events.back().wait();
    //   TensorOnHost<DType> x11(new DType[HEAD_DIM * EMBEDDING_DIM], HEAD_DIM, EMBEDDING_DIM);
    //   OCL.Map(tmp, x11);
    //   TensorOnHost<DType> y11(new DType[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
    //   OCL.Map(Sum, y11);
    //   TensorOnHost<DType> r11 = r10 * x11;
    //   std::cout << (r11 == y11 ? "ATTN Mul (output) PASS!!!\n" : "ATTN Mul (output) ERROR!!!\n");
    // #endif

    //   TensorOnFPGA T1 = this->OCL.Add<DType>(Sum, Sum, Events, RunEvent);
    //   Events.push_back(RunEvent);
    // #ifdef DEBUG
    //   std::cout << "ATTN Add\n";
    //   Events.back().wait();
    //   TensorOnHost<DType> y12(new DType[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
    //   OCL.Map(T1, y12);
    //   TensorOnHost<DType> r12 = y11 + y11;
    //   std::cout << (r12 == y12 ? "ATTN Add PASS!!!\n" : "ATTN Add ERROR!!!\n");
    // #endif

    for (unsigned int IterHead = 1; IterHead < HEAD_NUM; IterHead++)
    {
      tmp = this->WOFPGA.SubTensorRow(IterHead * HEAD_DIM * EMBEDDING_DIM, HEAD_DIM * EMBEDDING_DIM);
      TensorOnFPGA T0 =
          this->OCL.Mul<DType>(Heads[IterHead], tmp, Events, RunEvent);
      Events.push_back(RunEvent);

      TensorOnFPGA T1 = this->OCL.Add<DType>(Sum, T0, Events, RunEvent);
      Events.push_back(RunEvent);

      Sum = T1;
    }

    return Sum;
  // return TensorOnFPGA();
}

/**
 * Class FeedForwardLayer
 */

template <typename DType>
FeedForwardLayer<DType>::FeedForwardLayer(OCLWrap &OCL) : OCL(OCL)
{
  this->W0Host = TensorOnHost<DType>(nullptr, EMBEDDING_DIM, HIDDEN_DIM);
  this->W1Host = TensorOnHost<DType>(nullptr, HIDDEN_DIM, EMBEDDING_DIM);
  this->W2Host = TensorOnHost<DType>(nullptr, EMBEDDING_DIM, HIDDEN_DIM);

  this->W0FPGA = TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(this->W0Host.Dim0 * this->W0Host.Dim1),
                              0, this->W0Host.Dim0, this->W0Host.Dim1);
  this->W1FPGA = TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(this->W1Host.Dim0 * this->W1Host.Dim1),
                              0, this->W1Host.Dim0, this->W1Host.Dim1);
  this->W2FPGA = TensorOnFPGA(this->OCL.AllocateReadBuffer<DType>(this->W2Host.Dim0 * this->W2Host.Dim1),
                              0, this->W2Host.Dim0, this->W2Host.Dim1);
}

template <typename DType>
FeedForwardLayer<DType>::~FeedForwardLayer() {}

template <typename DType>
void FeedForwardLayer<DType>::load(std::ifstream &F)
{
#ifdef DEBUG
  std::cout << "Loading: <FeedForwardLayer>\n";
#endif

#ifdef RANDOM_INPUT
  this->W0Host.Data = GenerateRandomInput<DType>(this->W0Host.Dim0 * this->W0Host.Dim1);
  this->W1Host.Data = GenerateRandomInput<DType>(this->W1Host.Dim0 * this->W1Host.Dim1);
  this->W2Host.Data = GenerateRandomInput<DType>(this->W2Host.Dim0 * this->W2Host.Dim1);
#else
  this->W0Host.Data = new DType[this->W0Host.Dim0 * this->W0Host.Dim1];
  for (unsigned int IterRead = 0; IterRead < this->W0Host.Dim0 * this->W0Host.Dim1; IterRead++)
    F >> this->W0Host.Data[IterRead];

  this->W1Host.Data = new DType[this->W1Host.Dim0 * this->W1Host.Dim1];
  for (unsigned int IterRead = 0; IterRead < this->W1Host.Dim0 * this->W1Host.Dim1; IterRead++)
    F >> this->W1Host.Data[IterRead];

  this->W2Host.Data = new DType[this->W2Host.Dim0 * this->W2Host.Dim1];
  for (unsigned int IterRead = 0; IterRead < this->W2Host.Dim0 * this->W2Host.Dim1; IterRead++)
    F >> this->W2Host.Data[IterRead];
#endif
}

template <typename DType>
void FeedForwardLayer<DType>::migrate()
{
#ifdef DEBUG
  std::cout << "Migrating: <FeedForwardLayer>\n";
#endif

  this->OCL.Map(this->W0Host, this->W0FPGA);
  this->OCL.Map(this->W1Host, this->W1FPGA);
  this->OCL.Map(this->W2Host, this->W2FPGA);
}

template <typename DType>
TensorOnFPGA FeedForwardLayer<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  cl::Event RunEvent;

  TensorOnFPGA H0 = this->OCL.Mul<DType>(Input, this->W0FPGA, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Mul0\n";
  Events.back().wait();
  TensorOnHost<ap_int<8>> x0(new ap_int<8>[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
  OCL.Map(Input, x0);
  TensorOnHost<ap_int<8>> y0(new ap_int<8>[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H0, y0);
  TensorOnHost<ap_int<8>> r0 = x0 * this->W0Host;
  std::cout << (r0 == y0 ? "FFN Mul0 PASS!!!\n" : "FFN Mul0 ERROR!!!\n");
#endif

  TensorOnFPGA H1 = this->OCL.Silu<DType>(H0, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Silu\n";
  Events.back().wait();
  TensorOnHost<ap_int<8>> x1(new ap_int<8>[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H1, x1);
  TensorOnHost<ap_int<8>> r1 = SiluTest(r0);
  std::cout << (r1 == x1 ? "FFN Silu PASS!!!\n" : "FFN Silu ERROR!!!\n");
#endif

  TensorOnFPGA H2 = this->OCL.Mul<DType>(Input, this->W2FPGA, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Mul1\n";
  Events.back().wait();
  TensorOnHost<ap_int<8>> x2(new ap_int<8>[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H2, x2);
  TensorOnHost<ap_int<8>> r2 = x0 * this->W2Host;
  std::cout << (r2 == x2 ? "FFN Mul1 PASS!!!\n" : "FFN Mul1 ERROR!!!\n");
#endif

  TensorOnFPGA H3 = this->OCL.Dot<DType>(H1, H2, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Dot\n";
  Events.back().wait();
  TensorOnHost<ap_int<8>> x3(new ap_int<8>[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H3, x3);
  TensorOnHost<ap_int<8>> r3 = DotTest(r1, r2);
  std::cout << (r3 == x3 ? "FFN Dot PASS!!!\n" : "FFN Dot ERROR!!!\n");
#endif

  TensorOnFPGA H4 = this->OCL.Mul<DType>(H3, this->W1FPGA, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Mul2\n";
  Events.back().wait();
  TensorOnHost<ap_int<8>> x4(new ap_int<8>[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
  OCL.Map(H4, x4);
  TensorOnHost<ap_int<8>> r4 = x3 * this->W1Host;
  std::cout << (r4 == x4 ? "FFN Mul2 PASS!!!\n" : "FFN Mul2 ERROR!!!\n");
#endif

  return H4;
}

/**
 * Class RMSNormLayer
 */

template <typename DType>
RMSNormLayer<DType>::RMSNormLayer(OCLWrap &OCL) : OCL(OCL)
{
}

template <typename DType>
RMSNormLayer<DType>::~RMSNormLayer() {}

template <typename DType>
void RMSNormLayer<DType>::load(std::ifstream &F)
{
#ifdef DEBUG
  std::cout << "Loading: <RMSNormLayer>\n";
#endif
}

template <typename DType>
void RMSNormLayer<DType>::migrate()
{
#ifdef DEBUG
  std::cout << "Migrating: <RMSNormLayer>\n";
#endif
}

template <typename DType>
TensorOnFPGA RMSNormLayer<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  return this->OCL.RMSNorm<DType>(Input, Events);
}

/**
 * Class TransformerBlock
 */

template <typename DType>
TransformerBlock<DType>::TransformerBlock(OCLWrap &OCL) : OCL(OCL),
                                                          Attn(AttentionLayer<DType>(OCL)),
                                                          FFN(FeedForwardLayer<DType>(OCL)),
                                                          NormAttn(RMSNormLayer<DType>(OCL)),
                                                          NormFFN(RMSNormLayer<DType>(OCL))
{
}

template <typename DType>
TransformerBlock<DType>::~TransformerBlock() {}

template <typename DType>
void TransformerBlock<DType>::load(std::ifstream &F)
{
#ifdef DEBUG
  std::cout << "Loading: <TransformerBlock>\n";
#endif

  this->Attn.load(F);
  this->FFN.load(F);
  this->NormAttn.load(F);
  this->NormFFN.load(F);
}

template <typename DType>
void TransformerBlock<DType>::migrate()
{
  this->Attn.migrate();
  this->FFN.migrate();
  this->NormAttn.migrate();
  this->NormFFN.migrate();
}

template <typename DType>
TensorOnFPGA TransformerBlock<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  TensorOnFPGA H0 = this->NormAttn(Input, Events);
  TensorOnFPGA H1 = this->Attn(H0, Events);
  TensorOnFPGA H2 = this->OCL.Add<DType>(Input, H1, Events);
  TensorOnFPGA H3 = this->NormFFN(H2, Events);
  TensorOnFPGA H4 = this->FFN(H3, Events);
  TensorOnFPGA H5 = this->OCL.Add<DType>(H2, H4, Events);

  // Events.back().wait();
  // H0.ReleaseMem();
  // H1.ReleaseMem();
  // H2.ReleaseMem();
  // H3.ReleaseMem();
  // H4.ReleaseMem();

  return H5;
}

/**
 * Class Transformer
 */

template <typename DType>
Transformer<DType>::Transformer(OCLWrap &OCL) : OCL(OCL)
{

  for (unsigned int IterBlock = 0; IterBlock < LAYER_NUM; IterBlock++)
    this->Blocks.push_back(TransformerBlock<DType>(OCL));
}

template <typename DType>
Transformer<DType>::~Transformer() {}

template <typename DType>
void Transformer<DType>::load(std::ifstream &F)
{
#ifdef DEBUG
  std::cout << "Loading: <Transformer>\n";
#endif

  for (unsigned int IterBlock = 0; IterBlock < LAYER_NUM; IterBlock++)
    this->Blocks[IterBlock].load(F);
}

template <typename DType>
void Transformer<DType>::migrate()
{
  for (unsigned int IterBlock = 0; IterBlock < LAYER_NUM; IterBlock++)
    this->Blocks[IterBlock].migrate();
}

template <typename DType>
TensorOnFPGA Transformer<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  TensorOnFPGA X = Input;

#ifdef DEBUG
  for (unsigned int IterBlock = 0; IterBlock < 1; IterBlock++)
  {
    TensorOnFPGA H = this->Blocks[IterBlock](X, Events);

    // Events.back().wait();
    // X.ReleaseMem();

    X = H;
  }

#endif
  return X;
}

int main(int argc, char **argv)
{
  CmdLineParser parser;
  parser.addSwitch("--fpga", "-x", "FPGA binary (xclbin) file to use");

  parser.parse(argc, argv);
  string fpgaBinary = parser.value("fpga");

  if (fpgaBinary.size() == 0)
  {
    printf("ERROR: FPGA binary file (.xclbin) must be specified with the -x command line switch\n");
    return -1;
  }

  printf("FPGA binary       : %s\n", fpgaBinary.c_str());
  printf("Programming FPGA device\n");

  cl_int err;
  std::vector<cl::Device> devices = xcl::get_xil_devices();
  devices.resize(1);
  OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
  unsigned fileBufSize;
  char *fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

  // OCLWrap OCL(context, queue);
  // Transformer<short> Llama(OCL);
  // std::ifstream Ws("/home/chi/llama_fpga/weigths.dat", std::ios::binary | std::ios::in);
  // Llama.load(Ws);

  OCLWrap OCL(context, program, queue);
  // FeedForwardLayer<ap_int<8>> ffn(OCL);
  AttentionLayer<ap_int<8>> attn(OCL);
  std::ifstream Ws("/home/chi/llama_fpga/weigths.dat", std::ios::binary | std::ios::in);
  // ffn.load(Ws);
  // ffn.migrate();
  attn.load(Ws);
  attn.migrate();

  TensorOnHost<ap_int<8>> Input(GenerateRandomInput<ap_int<8>>(32 * EMBEDDING_DIM), 32, EMBEDDING_DIM);
  TensorOnFPGA _Input;
  _Input.Data = OCL.AllocateReadBuffer<ap_int<8>>(32 * EMBEDDING_DIM);
  _Input.Dim0 = Input.Dim0;
  _Input.Dim1 = Input.Dim1;
  OCL.Map(Input, _Input);

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  std::vector<cl::Event> Events;
  // TensorOnFPGA Output = ffn(_Input, Events);
  TensorOnFPGA Output = attn(_Input, Events);

  auto fpga_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
  printf("FPGA Time         : %10.4f s\n", fpga_duration.count());

  // std::cout << ((Res == ResHost) ? "PASS!!!" : "ERROR!!!") << std::endl;
}
