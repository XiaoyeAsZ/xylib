.PHONY: help

help::
	@echo  " Makefile Usage:"
	@echo  ""
	@echo  "  make build TARGET=<sw_emu/hw_emu/hw> "
	@echo  "  Command to generate the design for specified target"
	@echo  ""
	@echo  "  make run TARGET=<sw_emu/hw_emu/hw> "
	@echo  "  Command to generate and run the design for specified target"
	@echo  ""
	@echo  "  make clean TARGET=<sw_emu/hw_emu/hw> "
	@echo  "  Command to remove the generated non-hardware files for specified target"
	@echo  ""

################ Make File Describing host and kernel compile options
include ./make_options.mk

################ Below are the names for host executable and xclbin.
## Please keep it unchanged.
HOST_EXE := host.exe
XO_NAME := fpgabinary.$(TARGET)
XCLBIN := fpgabinary.$(TARGET).xclbin

################ Source Folder 
SRC_REPO := ./src
HEAD_REPO := ./include
KERNEL_REPO := ./kernel

################# If profiling is enabled pass host flag
ifeq ($(ENABLE_PROF),yes)
PROFILE_OPTIONS:=-c
endif

################ Build directory and number of images to process
ifeq ($(TARGET), sw_emu)
	BUILD_DIR ?= ./build_sw_emu
endif

ifeq ($(TARGET), hw_emu)
	BUILD_DIR ?= ./build_hw_emu
endif

ifeq ($(TARGET), hw)
	BUILD_DIR ?= ./build
endif


############## Host Application CPP dependencies
HOST_SRC_CPP := $(SRC_REPO)/host.cpp 
HOST_SRC_CPP += $(SRC_REPO)/xcl2.cpp
HOST_SRC_CPP += $(SRC_REPO)/cmdlineparser.cpp

############## Host Application header file dependencies
HOST_SRC_HEADERS := $(HEAD_REPO)/*.h
HOST_SRC_HEADERS += $(HEAD_REPO)/*.hpp


############## Kernel Source Files  Dependencies
# KERNEL_SRC_CPP = $(shell find . -type f -wholename '$(KERNEL_REPO)/krnl_*.cpp')
# KERNEL_SRC_HEADERS = $(shell find . -type f -wholename '$(HEAD_REPO)/krnl_*.h')
KERNEL_SRC_CPP := $(KERNEL_REPO)/krnl_gemv.cpp 
KERNEL_SRC_HEADERS := $(HEAD_REPO)/krnl_gemv.h
KERNEL_INCLUDES := -I$(HEAD_REPO)

############## Check the version of gcc avaiable and select
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
   ifndef XILINX_VIVADO
        $(error [ERROR]: g++ version older. Please use 5.0 or above.)
   else
      CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
      $(warning [WARNING]: g++ version is older. Using g++ provided by the Vitis tool : $(CXX))
   endif
endif

############## Set "HOST" Compiler Paths and Flags
CXXFLAGS += -I$(XILINX_XRT)/include/
CXXFLAGS += -I$(XILINX_VIVADO)/include/
CXXFLAGS += -I$(XILINX_HLS)/include/
CXXFLAGS += -I$(HEAD_REPO)
CXXFLAGS += -O3 -Wall -fmessage-length=0 -std=c++14

############## Set "HOST" Set Linker Paths and Flags
CXXLDFLAGS := -L$(XILINX_XRT)/lib/
CXXLDFLAGS += -lOpenCL -pthread -lrt -lstdc++ -lxilinxopencl -fopenmp

############## Kernel Compiler and Linker Flags
VPPFLAGS := -t $(TARGET)
VPPFLAGS += --platform $(PLATFORM) -R1 --save-temps
VPPFLAGS += --temp_dir $(BUILD_DIR)/$(VPP_TEMP_DIRS)
VPPFLAGS += --log_dir $(BUILD_DIR)/$(VPP_LOG_DIRS)
VPPFLAGS += --profile.data all:all:all:all
VPPFLAGS += --profile.trace_memory $(TRACE_DDR)
ifeq ($(ENABLE_STALL_TRACE),yes)
	VPPFLAGS += --profile.stall all:all:all
endif
VPPFLAGS += --config $(KERNEL_CONFIG_FILE)

create_dirs: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)


############## Host Executable File Generation
compile_host: $(BUILD_DIR)/$(HOST_EXE)

$(BUILD_DIR)/$(HOST_EXE): $(HOST_SRC_CPP) $(HOST_SRC_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(HOST_SRC_CPP) $(CXXLDFLAGS)
        #cp ./xrt.ini $(BUILD_DIR)/xrt.ini

############## Kernel XO and Xclbin File Generation
#Compile Kernel 

$(BUILD_DIR)/krnl_gemm.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlGemm $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_gemm.cpp -o $@

$(BUILD_DIR)/krnl_gemv.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlGemv $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_gemv.cpp -o $@

$(BUILD_DIR)/krnl_dotmat.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlDotMat $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_dotmat.cpp -o $@

$(BUILD_DIR)/krnl_silu.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlSilu $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_silu.cpp -o $@

$(BUILD_DIR)/krnl_remb.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlREmb $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_remb.cpp -o $@

$(BUILD_DIR)/krnl_move.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlMove $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_move.cpp -o $@

$(BUILD_DIR)/krnl_transpose.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlTranspose $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_transpose.cpp -o $@

$(BUILD_DIR)/krnl_softmax.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlSoftmax $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_softmax.cpp -o $@

$(BUILD_DIR)/krnl_addmat.xo: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
	v++ $(VPPFLAGS) -c -k KrnlAddMat $(KERNEL_INCLUDES) $(KERNEL_REPO)/krnl_addmat.cpp -o $@


# $(BUILD_DIR)/$(XO_NAME).xo: $(KERNEL_SRC_CPP) $(KERNEL_SRC_HEADERS)
# 	mkdir -p $(BUILD_DIR)
# 	mkdir -p $(BUILD_DIR)/$(VPP_TEMP_DIRS)
# 	mkdir -p $(BUILD_DIR)/$(VPP_LOG_DIRS)
# 	v++ $(VPPFLAGS) -c -k KrnlGemv   $(KERNEL_INCLUDES) $(KERNEL_SRC_CPP) -o $@



# Link Kernel
XO_LIST := $(BUILD_DIR)/krnl_gemm.xo $(BUILD_DIR)/krnl_gemv.xo $(BUILD_DIR)/krnl_dotmat.xo $(BUILD_DIR)/krnl_silu.xo $(BUILD_DIR)/krnl_remb.xo $(BUILD_DIR)/krnl_move.xo $(BUILD_DIR)/krnl_transpose.xo $(BUILD_DIR)/krnl_softmax.xo $(BUILD_DIR)/krnl_addmat.xo
$(BUILD_DIR)/$(XCLBIN): $(XO_LIST)
	v++ $(VPPFLAGS) -l -o $@ $(XO_LIST)



############## Emulation Files Generation
EMCONFIG_FILE = emconfig.json

$(BUILD_DIR)/$(EMCONFIG_FILE):
	 emconfigutil --nd 1  --platform $(PLATFORM) --od $(BUILD_DIR)

############## primary build targets

.PHONY: all clean  create_dirs

############## build the design without running host application
build: host xclbin emconfig
pre_build: host emconfig
host: $(BUILD_DIR)/$(HOST_EXE)
xclbin: $(BUILD_DIR)/$(XCLBIN)
emconfig: $(BUILD_DIR)/$(EMCONFIG_FILE)
############## build the design and then run host application
ifneq  ($(TARGET),hw)
run: build
	cp xrt.ini $(BUILD_DIR);
	cd $(BUILD_DIR) && XCL_EMULATION_MODE=$(TARGET) ./host.exe  -x ./$(XCLBIN) 
else
ifeq ($(USE_PRE_BUILT_XCLBIN),1)
run: pre_build
	cp xrt.ini $(BUILD_DIR);
	cd $(BUILD_DIR) && unset XCL_EMULATION_MODE && ./host.exe  -x  $(PRE_BUILT_XCLBIN_PATH)  $(PROFILE_OPTIONS) -f $(FILTER_TYPE) -r $(PARALLEL_ENQ_REQS) -n $(NUM_IMAGES)  $(INPUT_IMAGE_OPTION)
else
run: build
	cp xrt.ini $(BUILD_DIR);
	cd $(BUILD_DIR) && unset XCL_EMULATION_MODE && ./host.exe  -x ./$(XCLBIN) $(PROFILE_OPTIONS) -f $(FILTER_TYPE) -r $(PARALLEL_ENQ_REQS) -n $(NUM_IMAGES)  $(INPUT_IMAGE_OPTION)
endif
endif


## Clean generated files
clean:
	rm -rf $(BUILD_DIR) host*.exe
cleanall:
	@echo "Cleaning Software Emulation related host and kernel build file ...."
	rm -rf ./build_sw_emu host*.exe
	@echo "Cleaning Hardware Emulationrelated host and kernel build file ...."
	rm -rf ./build_hw_emu host*.exe
	@echo "Cleaning Hardware/System run related host and kernel buil file...."
	rm -rf ./build host*.exe
	rm -rf ./.Xil
	rm  -rf xcd.log