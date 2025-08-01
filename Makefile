# Makefile for OpenCL Mandelbrot Renderer
# Requires: OpenGL, OpenCL, GLFW, GLEW

CC = g++
CFLAGS = -std=c++17 -O3 -Wall -Wextra
LIBS = -lGL -lGLEW -lglfw -lOpenCL -lpthread

# Detect system
UNAME_S := $(shell uname -s)

# Linux-specific settings
ifeq ($(UNAME_S),Linux)
    CFLAGS += -D_GNU_SOURCE
    LIBS += -lX11 -lXrandr -lXinerama -lXcursor -ldl
endif

# macOS-specific settings (if you want to port later)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -I/opt/homebrew/include
    LIBS += -L/opt/homebrew/lib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
endif

TARGET = mandelbrot_opencl
TARGET_C = mandelbrot_opencl_c
TARGET_SIMPLE = mandelbrot_simple
TARGET_CPU = mandelbrot_cpu
SOURCES = mandelbrot_opencl.cpp
SOURCES_C = mandelbrot_opencl_c.cpp
SOURCES_SIMPLE = mandelbrot_simple.cpp
SOURCES_CPU = mandelbrot_cpu.cpp
KERNEL_FILE = mandelbrot_kernel.cl

.PHONY: all clean install-deps run debug c-version simple cpu

all: $(TARGET_SIMPLE) $(TARGET_CPU)

$(TARGET): $(SOURCES) $(KERNEL_FILE)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

$(TARGET_C): $(SOURCES_C) $(KERNEL_FILE)
	$(CC) $(CFLAGS) $(SOURCES_C) -o $(TARGET_C) $(LIBS)

$(TARGET_SIMPLE): $(SOURCES_SIMPLE) $(KERNEL_FILE)
	$(CC) $(CFLAGS) $(SOURCES_SIMPLE) -o $(TARGET_SIMPLE) $(LIBS)

$(TARGET_CPU): $(SOURCES_CPU)
	$(CC) $(CFLAGS) $(SOURCES_CPU) -o $(TARGET_CPU) -lGL -lGLEW -lglfw -lpthread -lX11 -lXrandr -lXinerama -lXcursor -ldl

c-version: $(TARGET_C)

simple: $(TARGET_SIMPLE)

cpu: $(TARGET_CPU)

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(TARGET) $(TARGET_C) $(TARGET_SIMPLE) $(TARGET_CPU)

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
	# For AMD GPUs:
	# sudo apt install rocm-opencl-dev

# Install dependencies on Fedora/RHEL
install-deps-fedora:
	sudo dnf install -y gcc-c++ cmake pkgconfig
	sudo dnf install -y mesa-libGL-devel mesa-libGLU-devel
	sudo dnf install -y glfw-devel glew-devel
	sudo dnf install -y opencl-headers ocl-icd-devel
	sudo dnf install -y mesa-libOpenCL

# Install dependencies on Arch Linux
install-deps-arch:
	sudo pacman -S --needed base-devel cmake pkgconf
	sudo pacman -S --needed mesa glu
	sudo pacman -S --needed glfw-x11 glew
	sudo pacman -S --needed opencl-headers ocl-icd
	sudo pacman -S --needed mesa-opencl-icd

run: $(TARGET_SIMPLE)
	./$(TARGET_SIMPLE)

# Check OpenCL devices
check-opencl:
	@echo "Checking OpenCL devices:"
	@which clinfo >/dev/null 2>&1 && clinfo || echo "Install clinfo to see OpenCL device information"

# Performance test
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@time ./$(TARGET) --benchmark

help:
	@echo "Available targets:"
	@echo "  all                 - Build the program"
	@echo "  debug               - Build with debug symbols"
	@echo "  clean               - Remove built files"
	@echo "  run                 - Build and run the program"
	@echo "  install-deps-ubuntu - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-fedora - Install dependencies on Fedora/RHEL"
	@echo "  install-deps-arch   - Install dependencies on Arch Linux"
	@echo "  check-opencl        - Check available OpenCL devices"
	@echo "  help                - Show this help message"

# Additional build configurations
release: CFLAGS += -DNDEBUG -march=native -flto
release: $(TARGET)

profile: CFLAGS += -pg -g
profile: LIBS += -pg
profile: $(TARGET)
