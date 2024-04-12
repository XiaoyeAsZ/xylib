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
    GeneratedInput[IterRnd] = DType(float(rand()) / RAND_MAX);
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
      DType tmp = 0;
      for (int k = 0; k < MA.Dim1; k++)
        tmp += MA.Data[i * MA.Dim1 + k] * MB.Data[k * MB.Dim1 + j];
      Res.Data[i * MB.Dim1 + j] = tmp;
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
    Res.Data[i] = x * (1.0 / (1.0 + exp(-x)));
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
    Res.Data[i] = MA.Data[i] * MB.Data[i];
  }
  return Res;
}

template <typename DType>
bool operator==(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  if (MA.Dim0 != MB.Dim0 || MA.Dim1 != MB.Dim1)
    return 0;
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i++)
  {
    if (abs(MA.Data[i] - MB.Data[i]) > 1e-5)
    {
      printf("%.10f differ %.10f\n", MA.Data[i], MB.Data[i]);
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
};

OCLWrap::~OCLWrap(){};

template <typename DType>
cl::Buffer OCLWrap::AllocateReadBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_READ_ONLY, (Size) * sizeof(DType), nullptr, &err));
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_WRITE_ONLY, (Size) * sizeof(DType), nullptr, &err));
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateReadWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_READ_WRITE, (Size) * sizeof(DType), nullptr, &err));
  OCL_CHECK(err, err = this->Queue.enqueueMigrateMemObjects({Buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
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

  this->KrnlGemm.setArg(0, Tensor0.Dim0);
  this->KrnlGemm.setArg(1, Tensor0.Dim1);
  this->KrnlGemm.setArg(2, Tensor1.Dim1);
  this->KrnlGemm.setArg(3, Tensor0.Data);
  this->KrnlGemm.setArg(4, Tensor1.Data);
  this->KrnlGemm.setArg(5, Buf.Data);
  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlGemm, &Events, &RunEvent));

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Add(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1,
                          std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  assert(Tensor0.Dim0 == Tensor1.Dim0 && Tensor0.Dim1 == Tensor1.Dim1);

  cl_int err;
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

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
    this->KrnlAddMat.setArg(0, Tensor0.Dim0);
    this->KrnlAddMat.setArg(1, Tensor0.Dim1);
    this->KrnlAddMat.setArg(2, Tensor0.Data);
    this->KrnlAddMat.setArg(3, Tensor1.Data);
    this->KrnlAddMat.setArg(4, Buf.Data);
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
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

  assert(Tensor0.Dim0 == Tensor1.Dim0);
  this->KrnlREmb.setArg(0, Tensor0.Data);
  this->KrnlREmb.setArg(1, Tensor0.Offset);
  this->KrnlREmb.setArg(2, Tensor0.Dim0);
  this->KrnlREmb.setArg(3, Tensor0.Dim1);
  this->KrnlREmb.setArg(4, Tensor1.Data);
  this->KrnlREmb.setArg(5, Tensor1.Offset);
  this->KrnlREmb.setArg(6, Buf.Data);
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
  this->KrnlMove.setArg(0, Tensor0.Offset);
  this->KrnlMove.setArg(0, Tensor1.Data);
  this->KrnlMove.setArg(0, Tensor1.Offset);
  this->KrnlMove.setArg(0, MoveLen);

  OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlMove, &Events, &RunEvent));
};

template <typename DType>
TensorOnFPGA OCLWrap::Trans(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events, cl::Event &RunEvent)
{
  cl_int err;
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

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
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

  this->KrnlSoftmax.setArg(0, Tensor0.Dim0);
  this->KrnlSoftmax.setArg(1, Tensor0.Dim1);
  this->KrnlSoftmax.setArg(2, Tensor0.Data);
  this->KrnlSoftmax.setArg(3, Buf.Data);
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
  }
  this->WOHost = TensorOnHost<DType>(nullptr, EMBEDDING_DIM, EMBEDDING_DIM);
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
TensorOnFPGA AttentionLayer<DType>::operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events)
{
  std::vector<TensorOnFPGA> Heads;

  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    std::vector<cl::Event> EventsInHead;

    TensorOnFPGA H0 = this->OCL.Mul<DType>(Input, this->WQFPGA[IterHead], EventsInHead);
    TensorOnFPGA H1 = this->OCL.Mul<DType>(Input, this->WKFPGA[IterHead], EventsInHead);
    TensorOnFPGA H2 = this->OCL.Mul<DType>(Input, this->WVFPGA[IterHead], EventsInHead);
    this->OCL.REmb<DType>(H0, H1, this->OCL.Freq, EventsInHead);

    assert(this->KCache[IterHead].Dim0 + H1.Dim0 <= this->MaxCacheLen);
    assert(this->VCache[IterHead].Dim0 + H2.Dim0 <= this->MaxCacheLen);
    this->OCL.Move<DType>(H1, this->KCache[IterHead].SubTensorRow(this->CurLen * HEAD_DIM, H1.Dim0 * H1.Dim1), EventsInHead);
    this->OCL.Move<DType>(H2, this->VCache[IterHead].SubTensorRow(this->CurLen * HEAD_DIM, H2.Dim0 * H2.Dim1), EventsInHead);

    TensorOnFPGA H3 = this->OCL.Trans<DType>(H1, EventsInHead);
    TensorOnFPGA H4 = this->OCL.Mul<DType>(H0, H3, EventsInHead);
    TensorOnFPGA H5 = this->OCL.Softmax<DType>(H4, EventsInHead);
    TensorOnFPGA H6 = this->OCL.Mul<DType>(H5, H2, EventsInHead);

    Events.push_back(EventsInHead.back());

    // EventsInHead.back().wait();
    // H0.ReleaseMem();
    // H1.ReleaseMem();
    // H2.ReleaseMem();
    // H3.ReleaseMem();
    // H4.ReleaseMem();

    Heads.push_back(H6);
  }

  this->CurLen += Input.Dim0;

  assert(Heads.size() == HEAD_NUM);
  TensorOnFPGA Sum = this->OCL.Mul<DType>(Heads[0], this->WOFPGA.SubTensorRow(0, EMBEDDING_DIM * EMBEDDING_DIM), Events);
  for (unsigned int IterHead = 1; IterHead < HEAD_NUM; IterHead++)
  {
    TensorOnFPGA T0 =
        this->OCL.Mul<DType>(Heads[IterHead],
                             this->WOFPGA.SubTensorRow(IterHead * EMBEDDING_DIM, EMBEDDING_DIM * EMBEDDING_DIM),
                             Events);
    TensorOnFPGA T1 = this->OCL.Add<DType>(Sum, T0, Events);
    // Sum.ReleaseMem();
    // T0.ReleaseMem();
    Sum = T1;
  }

  return Sum;
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
  TensorOnHost<float> x0(new float[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
  OCL.Map(Input, x0);
  TensorOnHost<float> y0(new float[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H0, y0);
  TensorOnHost<float> r0 = x0 * this->W0Host;
  std::cout << (r0 == y0 ? "FFN Mul0 PASS!!!\n" : "FFN Mul0 ERROR!!!\n");
#endif

  TensorOnFPGA H1 = this->OCL.Silu<DType>(H0, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Silu\n";
  Events.back().wait();
  TensorOnHost<float> x1(new float[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H1, x1);
  TensorOnHost<float> r1 = SiluTest(r0);
  std::cout << (r1 == x1 ? "FFN Silu PASS!!!\n" : "FFN Silu ERROR!!!\n");
#endif

  TensorOnFPGA H2 = this->OCL.Mul<DType>(Input, this->W2FPGA, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Mul1\n";
  Events.back().wait();
  TensorOnHost<float> x2(new float[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H2, x2);
  TensorOnHost<float> r2 = x0 * this->W2Host;
  std::cout << (r2 == x2 ? "FFN Mul1 PASS!!!\n" : "FFN Mul1 ERROR!!!\n");
#endif

  TensorOnFPGA H3 = this->OCL.Dot<DType>(H1, H2, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Dot\n";
  Events.back().wait();
  TensorOnHost<float> x3(new float[Input.Dim0 * HIDDEN_DIM], Input.Dim0, HIDDEN_DIM);
  OCL.Map(H3, x3);
  TensorOnHost<float> r3 = DotTest(r1, r2);
  std::cout << (r3 == x3 ? "FFN Dot PASS!!!\n" : "FFN Dot ERROR!!!\n");
#endif

  TensorOnFPGA H4 = this->OCL.Mul<DType>(H3, this->W1FPGA, Events, RunEvent);
  Events.push_back(RunEvent);
#ifdef DEBUG
  std::cout << "FFN Mul2\n";
  Events.back().wait();
  TensorOnHost<float> x4(new float[Input.Dim0 * EMBEDDING_DIM], Input.Dim0, EMBEDDING_DIM);
  OCL.Map(H4, x4);
  TensorOnHost<float> r4 = x3 * this->W1Host;
  std::cout << (r4 == x4 ? "FFN Mul2 PASS!!!\n" : "FFN Mul2 ERROR!!!\n");
#endif

  // Events.back().wait();
  // H0.ReleaseMem();
  // H1.ReleaseMem();
  // H2.ReleaseMem();
  // H3.ReleaseMem();
  // H4.ReleaseMem();

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
  FeedForwardLayer<float> ffn(OCL);
  std::ifstream Ws("/home/chi/llama_fpga/weigths.dat", std::ios::binary | std::ios::in);
  ffn.load(Ws);
  ffn.migrate();

  TensorOnHost<float> Input(GenerateRandomInput<float>(16 * EMBEDDING_DIM), 16, EMBEDDING_DIM);
  TensorOnFPGA _Input;
  _Input.Data = OCL.AllocateReadBuffer<float>(16 * EMBEDDING_DIM);
  _Input.Dim0 = Input.Dim0;
  _Input.Dim1 = Input.Dim1;
  OCL.Map(Input, _Input);

  std::vector<cl::Event> Events;
  TensorOnFPGA Output = ffn(_Input, Events);

  // std::cout << ((Res == ResHost) ? "PASS!!!" : "ERROR!!!") << std::endl;
}
