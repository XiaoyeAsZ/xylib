
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "xcl2.hpp"
#include "cmdlineparser.h"

using namespace sda;
using namespace sda::utils;

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
  devices.resize(1); // (arbitrarily) use the first Xilinx device that is found
  OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
  unsigned fileBufSize;
  char *fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

  auto *matrix_data = new int[128 * 128];
  auto *vector_data = new int[128];
  auto *res_data = new int[128];
  auto *ref_data = new int[128];
  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < 128; j++)
      matrix_data[i * 128 + j] = i + j;
  }
  for (int i = 0; i < 128; i++)
  {
    vector_data[i] = i;
    res_data[i] = 0;
  }

  for (int i = 0; i < 128; i++)
  {
    int tmp = 0;
    for (int j = 0; j < 128; j++)
      tmp += matrix_data[i * 128 + j] * vector_data[j];
    ref_data[i] = tmp;
  }

  printf("Running FPGA accelerator\n");

  cl::Buffer matrix_buf;
  cl::Buffer vector_buf;
  cl::Buffer result_buf;
  OCL_CHECK(err, matrix_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (128 * 128) * sizeof(int), nullptr, &err));
  OCL_CHECK(err, vector_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (128) * sizeof(int), nullptr, &err));
  OCL_CHECK(err, result_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, (128) * sizeof(int), nullptr, &err));

  cl::Event in1_event;
  cl::Event in2_event;
  cl::Event run_event;
  cl::Event out_event;
  std::vector<cl::Event> events;

  cl::Kernel kernel;
  OCL_CHECK(err, kernel = cl::Kernel(program, "KrnlGemv", &err));
  OCL_CHECK(err, err = kernel.setArg(0, 128));
  OCL_CHECK(err, err = kernel.setArg(1, 128));
  OCL_CHECK(err, err = kernel.setArg(2, matrix_buf));
  OCL_CHECK(err, err = kernel.setArg(3, vector_buf));
  OCL_CHECK(err, err = kernel.setArg(4, result_buf));

  size_t offset = 0;
  OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({matrix_buf, vector_buf, result_buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  queue.finish();

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  OCL_CHECK(err, err = queue.enqueueWriteBuffer(matrix_buf, CL_FALSE, offset, 128 * 128 * sizeof(int), matrix_data, nullptr, &in1_event));
  events.push_back(in1_event);
  OCL_CHECK(err, err = queue.enqueueWriteBuffer(vector_buf, CL_FALSE, offset, 128 * sizeof(int), vector_data, nullptr, &in2_event));
  events.push_back(in2_event);
  OCL_CHECK(err, err = queue.enqueueTask(kernel, &events, &run_event));
  events.push_back(run_event);
  OCL_CHECK(err, err = queue.enqueueReadBuffer(result_buf, CL_FALSE, offset, 128 * sizeof(int), res_data, &events, &out_event));
  events.push_back(out_event);

  events.back().wait();
  events.clear();

  auto fpga_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;

  bool flag = 1;
  for (int i = 0; i < 128; i++)
  {
    if (ref_data[i] != res_data[i])
      flag = 0;
  }

  std::cout << (flag ? "Test PASS !!!" : "ERROR !!!") << std::endl;

  printf("FPGA Time         : %10.4f s\n", fpga_duration.count());
}
