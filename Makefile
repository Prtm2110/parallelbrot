# Makefile for Mandelbrot Renderer with OpenCL and CUDA support
# Requires: OpenGL, OpenCL/CUDA, GLFW, GLEW

CC = g++
NVCC = nvcc
CFLAGS = -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS = -std=c++17 -O3 -arch=sm_50
LIBS = -lGL -lGLEW -lglfw -lOpenCL -lpthread
CUDA_LIBS = -lGL -lGLEW -lglfw -lcuda -lcudart -lpthread

# Detect system
UNAME_S := $(shell uname -s)

# Linux-specific settings
ifeq ($(UNAME_S),Linux)
    CFLAGS += -D_GNU_SOURCE
    LIBS += -lX11 -lXrandr -lXinerama -lXcursor -ldl
    CUDA_LIBS += -lX11 -lXrandr -lXinerama -lXcursor -ldl
endif

# macOS-specific settings (if you want to port later)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -I/opt/homebrew/include
    LIBS += -L/opt/homebrew/lib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
    CUDA_LIBS += -L/opt/homebrew/lib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
endif

# Source directories
SRC_DIR = src
OPENCL_DIR = $(SRC_DIR)/opencl
CUDA_DIR = $(SRC_DIR)/cuda
CPU_DIR = $(SRC_DIR)/cpu
BUILD_DIR = build

# Targets and sources with new paths
TARGET = $(BUILD_DIR)/mandelbrot_opencl_full
TARGET_C = $(BUILD_DIR)/mandelbrot_opencl_c
TARGET_SIMPLE = $(BUILD_DIR)/mandelbrot_opencl
TARGET_CPU = $(BUILD_DIR)/mandelbrot_cpu
TARGET_CUDA = $(BUILD_DIR)/mandelbrot_cuda

SOURCES = $(OPENCL_DIR)/mandelbrot_opencl_full.cpp
SOURCES_C = $(OPENCL_DIR)/mandelbrot_opencl_c.cpp
SOURCES_SIMPLE = $(OPENCL_DIR)/mandelbrot_opencl.cpp
SOURCES_CPU = $(CPU_DIR)/mandelbrot_cpu.cpp
SOURCES_CUDA = $(CUDA_DIR)/mandelbrot_cuda.cpp
CUDA_KERNEL = $(CUDA_DIR)/mandelbrot_kernel.cu
KERNEL_FILE = $(OPENCL_DIR)/mandelbrot_kernel.cl

.PHONY: all clean install-deps run debug c-version opencl cpu cuda check-opencl check-cuda install-cuda

all: $(TARGET_SIMPLE) $(TARGET_CPU)

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES) $(KERNEL_FILE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

$(TARGET_C): $(SOURCES_C) $(KERNEL_FILE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SOURCES_C) -o $(TARGET_C) $(LIBS)

$(TARGET_SIMPLE): $(SOURCES_SIMPLE) $(KERNEL_FILE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SOURCES_SIMPLE) -o $(TARGET_SIMPLE) $(LIBS)

$(TARGET_CPU): $(SOURCES_CPU) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SOURCES_CPU) -o $(TARGET_CPU) -lGL -lGLEW -lglfw -lpthread -lX11 -lXrandr -lXinerama -lXcursor -ldl

$(TARGET_CUDA): $(SOURCES_CUDA) $(CUDA_KERNEL) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_KERNEL) $(SOURCES_CUDA) -o $(TARGET_CUDA) $(CUDA_LIBS)

c-version: $(TARGET_C)

opencl: $(TARGET_SIMPLE)

# Deprecated alias for backward compatibility
simple: opencl
	@echo "Warning: 'make simple' is deprecated. Use 'make opencl' instead."

cpu: $(TARGET_CPU)

cuda: $(TARGET_CUDA)

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(BUILD_DIR)/*
	rmdir $(BUILD_DIR) 2>/dev/null || true

# Install dependencies on Ubuntu/Debian
install-deps-ubuntu:
	sudo apt update
	sudo apt install -y build-essential cmake pkg-config
	sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev
	sudo apt install -y libglfw3-dev libglew-dev
	sudo apt install -y opencl-headers ocl-icd-opencl-dev
	sudo apt install -y mesa-opencl-icd  # Mesa OpenCL implementation
	# For NVIDIA GPUs:
	# sudo apt install nvidia-opencl-dev

# Install CUDA dependencies on Ubuntu/Debian
install-cuda-ubuntu:
	@echo "Installing CUDA dependencies for Ubuntu/Debian..."
	sudo apt update
	sudo apt install -y build-essential cmake pkg-config
	sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev
	sudo apt install -y libglfw3-dev libglew-dev
	@echo "For CUDA support, install NVIDIA CUDA Toolkit:"
	@echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
	@echo "  sudo dpkg -i cuda-keyring_1.0-1_all.deb"
	@echo "  sudo apt update"
	@echo "  sudo apt install -y cuda-toolkit"
	@echo "Or visit: https://developer.nvidia.com/cuda-downloads"

# Install dependencies on Fedora/RHEL
install-deps-fedora:
	sudo dnf install -y gcc-c++ cmake pkgconfig
	sudo dnf install -y mesa-libGL-devel mesa-libGLU-devel
	sudo dnf install -y glfw-devel glew-devel
	sudo dnf install -y opencl-headers ocl-icd-devel
	sudo dnf install -y mesa-libOpenCL

# Install CUDA dependencies on Fedora/RHEL
install-cuda-fedora:
	@echo "Installing CUDA dependencies for Fedora/RHEL..."
	sudo dnf install -y gcc-c++ cmake pkgconfig
	sudo dnf install -y mesa-libGL-devel mesa-libGLU-devel
	sudo dnf install -y glfw-devel glew-devel
	@echo "For CUDA support, install NVIDIA CUDA Toolkit:"
	@echo "  sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo"
	@echo "  sudo dnf install -y cuda-toolkit"
	@echo "Or visit: https://developer.nvidia.com/cuda-downloads"

# Install dependencies on Arch Linux
install-deps-arch:
	sudo pacman -S --needed base-devel cmake pkgconf
	sudo pacman -S --needed mesa glu
	sudo pacman -S --needed glfw-x11 glew
	sudo pacman -S --needed opencl-headers ocl-icd
	sudo pacman -S --needed mesa-opencl-icd

# Install CUDA dependencies on Arch Linux
install-cuda-arch:
	@echo "Installing CUDA dependencies for Arch Linux..."
	sudo pacman -S --needed base-devel cmake pkgconf
	sudo pacman -S --needed mesa glu
	sudo pacman -S --needed glfw-x11 glew
	sudo pacman -S --needed cuda cuda-tools

run: $(TARGET_SIMPLE)
	$(TARGET_SIMPLE)

# Check OpenCL devices
check-opencl:
	@echo "Checking OpenCL devices:"
	@which clinfo >/dev/null 2>&1 && clinfo || echo "Install clinfo to see OpenCL device information"

# Check CUDA devices
check-cuda:
	@echo "Checking CUDA devices:"
	@which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found. Install NVIDIA drivers and CUDA toolkit."
	@which nvcc >/dev/null 2>&1 && echo "NVCC version: $$(nvcc --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*')" || echo "nvcc not found. Install CUDA toolkit."

# Performance test
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@time ./$(TARGET) --benchmark

help:
	@echo "Available targets:"
	@echo "  all                  - Build the OpenCL versions (opencl + cpu)"
	@echo "  simple               - Build main OpenCL GPU version"
	@echo "  cpu                  - Build CPU version"
	@echo "  cuda                 - Build CUDA GPU version"
	@echo "  c-version            - Build C-style OpenCL version"
	@echo "  debug                - Build with debug symbols"
	@echo "  clean                - Remove built files"
	@echo "  run                  - Build and run the main OpenCL version"
	@echo ""
	@echo "Built executables will be in the 'build/' directory:"
	@echo "  build/mandelbrot_opencl      - Main OpenCL version"
	@echo "  build/mandelbrot_cpu         - CPU version"
	@echo "  build/mandelbrot_cuda        - CUDA version (if built)"
	@echo "  build/mandelbrot_opencl_full - Full-featured OpenCL version"
	@echo ""
	@echo "Installation targets:"
	@echo "  install-deps-ubuntu  - Install OpenCL dependencies on Ubuntu/Debian"
	@echo "  install-deps-fedora  - Install OpenCL dependencies on Fedora/RHEL"
	@echo "  install-deps-arch    - Install OpenCL dependencies on Arch Linux"
	@echo "  install-cuda-ubuntu  - Install CUDA dependencies on Ubuntu/Debian"
	@echo "  install-cuda-fedora  - Install CUDA dependencies on Fedora/RHEL"
	@echo "  install-cuda-arch    - Install CUDA dependencies on Arch Linux"
	@echo ""
	@echo "Diagnostic targets:"
	@echo "  check-opencl         - Check available OpenCL devices"
	@echo "  check-cuda           - Check available CUDA devices"
	@echo "  help                 - Show this help message"

# Additional build configurations
release: CFLAGS += -DNDEBUG -march=native -flto
release: $(TARGET)

profile: CFLAGS += -pg -g
profile: LIBS += -pg
profile: $(TARGET)
