# Mandelbrot GPU Renderer

![OpenCL](https://img.shields.io/badge/OpenCL-✓%20Supported-green)
![CUDA](https://img.shields.io/badge/CUDA-✓%20Supported-green)
![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD%20%7C%20Intel-orange)

A high-performance GPU-accelerated Mandelbrot set renderer with multiple backend support: **OpenCL** for cross-platform compatibility and **CUDA** for maximum NVIDIA GPU performance, plus OpenGL for hardware-accelerated rendering.

![Mandelbrot Renderer Demo](docs/mandelbrot.gif)

## Quick Start

```bash
# Check what GPU hardware you have
make check-opencl     # Check OpenCL support
make check-cuda       # Check CUDA support (NVIDIA only)

# Install dependencies for your system
make install-deps-ubuntu     # Ubuntu/Debian OpenCL deps
make install-cuda-ubuntu     # Ubuntu/Debian CUDA deps (if NVIDIA)

# Build and run
make opencl                    # Build OpenCL version (recommended for all GPUs)
build/mandelbrot_opencl        # Run OpenCL version

# OR for NVIDIA GPUs with CUDA installed:
make cuda                      # Build CUDA version  
build/mandelbrot_cuda          # Run CUDA version (maximum performance)

# OR for comparison:
make cpu                       # Build CPU version
build/mandelbrot_cpu           # Run CPU version
```

## Features

- **Multiple GPU Backends**: 
  - **OpenCL**: Cross-platform GPU acceleration (NVIDIA, AMD, Intel)
  - **CUDA**: Optimized for NVIDIA GPUs with maximum performance
- **Hardware Rendering**: OpenGL for fast texture-based rendering
- **Interactive Navigation**: Real-time pan and zoom with mouse and keyboard
- **Multiple Implementations**: Different optimization levels and comparison versions
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **High Performance**: 10-100x faster than CPU implementations

## Available Programs

| Program | Backend | Description | Best For |
|---------|---------|-------------|----------|
| `build/mandelbrot_opencl` | OpenCL | Primary GPU-accelerated version | Cross-platform compatibility |
| `build/mandelbrot_cuda` | CUDA | NVIDIA-optimized version | Maximum NVIDIA GPU performance |
| `build/mandelbrot_cpu` | CPU | CPU-only implementation | Performance comparison |
| `build/mandelbrot_opencl_full` | OpenCL | Full-featured OpenCL version | Advanced OpenCL features |

## Project Structure

```
src/
├── opencl/                    # OpenCL implementations
│   ├── mandelbrot_opencl.cpp      # Main OpenCL version  
│   ├── mandelbrot_opencl_full.cpp # Full-featured OpenCL
│   ├── mandelbrot_opencl_c.cpp    # C-style OpenCL
│   └── mandelbrot_kernel.cl       # OpenCL compute kernel
├── cuda/                      # CUDA implementation
│   ├── mandelbrot_cuda.cpp        # CUDA version
│   └── mandelbrot_kernel.cu       # CUDA compute kernel
└── cpu/                       # CPU implementation
    └── mandelbrot_cpu.cpp         # CPU-only version
build/                         # Compiled executables
scripts/                       # Utility scripts
```

## Controls

All versions share the same controls:

- **Mouse drag**: Pan around the fractal
- **Mouse wheel**: Zoom in/out
- **Arrow keys**: Pan with keyboard
- **+/-**: Increase/decrease iteration count
- **C**: Cycle through color schemes (Ultra Fractal, Fire, Ocean, Psychedelic)
- **R**: Reset view to default
- **ESC**: Exit application

## Requirements

### OpenCL Requirements (for `build/mandelbrot_opencl`, `build/mandelbrot_opencl_full`)

#### Ubuntu/Debian:
```bash
make install-deps-ubuntu
```

#### Fedora/RHEL:
```bash
make install-deps-fedora
```

#### Arch Linux:
```bash
make install-deps-arch
```

### CUDA Requirements (for `mandelbrot_cuda`)

#### Ubuntu/Debian:
```bash
make install-cuda-ubuntu
```

#### Fedora/RHEL:
```bash
make install-cuda-fedora
```

#### Arch Linux:
```bash
make install-cuda-arch
```

### Manual Installation

**Common requirements:**
- OpenGL development libraries
- GLFW3 (window management)
- GLEW (OpenGL extension loading)

**For OpenCL:**
- OpenCL headers and runtime
- GPU-specific OpenCL drivers (NVIDIA, AMD, or Intel)

**For CUDA:**
- NVIDIA CUDA Toolkit (11.0+)
- NVIDIA GPU with Compute Capability 5.0+
- NVIDIA drivers (470+)

## Building

### Build All Programs
```bash
# Build OpenCL versions (default)
make

# Build specific versions
make opencl    # OpenCL GPU version (all platforms)
make cpu       # CPU version  
make cuda      # CUDA GPU version (NVIDIA only)

# Build with debug symbols
make debug

# Build optimized release version
make release

# Clean build files
make clean
```

### Check GPU Support
```bash
# Check OpenCL devices
make check-opencl

# Check CUDA devices
make check-cuda
```

## Running

### Quick Start
```bash
# Run OpenCL GPU version (recommended)
build/mandelbrot_opencl

# Run CUDA version (NVIDIA GPUs only)
build/mandelbrot_cuda

# Run CPU version (for comparison)
build/mandelbrot_cpu

# Or use make target
make run  # Builds and runs mandelbrot_opencl
```

### Choosing the Right Version

**Use `build/mandelbrot_opencl` (OpenCL) when:**
- You have any modern GPU (NVIDIA, AMD, Intel)
- You want cross-platform compatibility
- You're unsure which to choose

**Use `build/mandelbrot_cuda` (CUDA) when:**
- You have an NVIDIA GPU
- You want maximum performance
- You need CUDA-specific optimizations

**Use `build/mandelbrot_cpu` when:**
- You don't have a compatible GPU
- You want to compare performance
- Debugging or development

## GPU Device Support

### OpenCL Support
This program works with any OpenCL-compatible device:
- **NVIDIA GPUs**: Install NVIDIA OpenCL drivers
- **AMD GPUs**: Install ROCm or AMDGPU-PRO drivers  
- **Intel GPUs**: Intel OpenCL runtime
- **CPU fallback**: Most OpenCL implementations include CPU support

### CUDA Support  
The CUDA version requires:
- **NVIDIA GPU**: GeForce GTX 750 or newer (Compute Capability 5.0+)
- **NVIDIA CUDA Toolkit**: Version 11.0 or later
- **NVIDIA Drivers**: Version 470 or later

### Check Your Hardware
```bash
# Check OpenCL devices
make check-opencl

# Check CUDA devices  
make check-cuda

# Get detailed GPU info
nvidia-smi          # For NVIDIA GPUs
lspci | grep VGA    # List all graphics cards
```

### Performance Features:
- **Parallel Processing**: Utilizes hundreds/thousands of GPU cores
- **Memory Bandwidth**: Leverages high GPU memory bandwidth  
- **Real-time Interaction**: Smooth navigation even at high iteration counts
- **Optimized Kernels**: Hand-tuned for both OpenCL and CUDA

### Performance Tips:
- Use CUDA version for NVIDIA GPUs when possible
- Increase iteration count gradually to find sweet spot
- Lower resolution if experiencing lag
- Ensure GPU drivers are up to date

## Architecture

### Components

1. **OpenCL Implementation** (`src/opencl/mandelbrot_opencl.cpp`):
   - Cross-platform GPU acceleration
   - OpenCL kernel execution (`src/opencl/mandelbrot_kernel.cl`)
   - OpenGL rendering and window management
   - User input handling

2. **CUDA Implementation** (`src/cuda/mandelbrot_cuda.cpp`):
   - NVIDIA-optimized GPU acceleration
   - CUDA kernel execution (`src/cuda/mandelbrot_kernel.cu`)
   - CUDA-OpenGL interoperability for maximum performance
   - Same user interface as OpenCL version

3. **CPU Implementation** (`src/cpu/mandelbrot_cpu.cpp`):
   - CPU-only computation for comparison
   - Same controls and functionality
   - Multi-threaded CPU parallelization

4. **Compute Kernels**:
   - **OpenCL kernel** (`src/opencl/mandelbrot_kernel.cl`): Cross-platform GPU code
   - **CUDA kernel** (`src/cuda/mandelbrot_kernel.cu`): NVIDIA-optimized GPU code
   - Both include optimized Mandelbrot computation and coloring

### Data Flow

#### OpenCL Version:
1. User input updates view parameters (center, zoom, iterations)
2. OpenCL kernel computes Mandelbrot set on GPU
3. Results written to OpenGL texture via OpenCL-OpenGL interop
4. OpenGL renders fullscreen quad with texture
5. Display updates in real-time

#### CUDA Version:
1. User input updates view parameters (center, zoom, iterations)  
2. CUDA kernel computes Mandelbrot set on GPU
3. Results transferred via CUDA-OpenGL interoperability
4. OpenGL renders fullscreen quad with texture
5. Display updates in real-time (typically faster than OpenCL)

## Customization

### Kernel Selection

You can modify the main program to use different kernels:
- `mandelbrot`: Basic version
- `mandelbrot_smooth`: Better visual quality
- `mandelbrot_optimized`: Maximum performance

### Color Schemes

The kernels include multiple color mapping functions. You can:
- Modify the `mapColor` function in the kernel
- Add new color palettes
- Implement user-selectable color schemes

### Performance Tuning

- Adjust work group sizes for your GPU
- Modify iteration unrolling in optimized kernel
- Experiment with different precision levels

## Troubleshooting

### OpenCL Issues

1. **No OpenCL devices found**:
   - Install proper GPU drivers
   - Install OpenCL runtime for your hardware  
   - Check with `make check-opencl` or `clinfo` command
   - Try `mandelbrot_cpu` as fallback

2. **Compilation errors**:
   - Verify OpenCL headers are installed
   - Check compiler version (C++17 required)
   - Try: `sudo apt install opencl-headers ocl-icd-opencl-dev`

3. **Runtime errors**:
   - Ensure OpenGL context is created before OpenCL
   - Check GPU memory availability
   - Try running with lower iteration count

### CUDA Issues

1. **CUDA compilation fails**:
   - Check if CUDA toolkit is installed: `nvcc --version`
   - Install CUDA: `make install-cuda-ubuntu` (or your distro)
   - Verify NVIDIA drivers: `nvidia-smi`

2. **No CUDA devices found**:
   - Check GPU compatibility: `make check-cuda`
   - Ensure NVIDIA GPU with Compute Capability 5.0+
   - Update NVIDIA drivers to 470+ 
   - Try `build/mandelbrot_opencl` (OpenCL) as alternative

3. **CUDA runtime errors**:
   - Check GPU memory with `nvidia-smi`
   - Ensure no other CUDA applications are running
   - Try reducing window resolution

### Performance Issues

1. **Slow rendering**:
   - Try CUDA version for NVIDIA GPUs: `build/mandelbrot_cuda`
   - Reduce iteration count with `-/+` keys
   - Check if using integrated vs dedicated GPU
   - Monitor GPU usage with `nvidia-smi` or `radeontop`

2. **Input lag**:
   - Enable VSync in graphics drivers
   - Reduce window size for testing
   - Check system load with `htop`

3. **Build issues**:
   - Install dependencies: `make install-deps-ubuntu` 
   - For CUDA: `make install-cuda-ubuntu`
   - Check compiler: `gcc --version` (need GCC 7+)

### Quick Fixes

```bash
# Test different versions
build/mandelbrot_opencl   # Try OpenCL first
build/mandelbrot_cuda     # Try CUDA if you have NVIDIA
build/mandelbrot_cpu      # CPU fallback

# Check system compatibility  
make check-opencl     # Check OpenCL devices
make check-cuda       # Check CUDA devices
lspci | grep VGA      # List graphics cards

# Debug build for more error info
make debug
build/mandelbrot_opencl
```

## Summary

### Which Version Should I Use?

| Your Hardware | Recommended | Command | Performance |
|---------------|-------------|---------|-------------|
| **NVIDIA GPU + CUDA** | `mandelbrot_cuda` | `make cuda && build/mandelbrot_cuda` | ⭐⭐⭐⭐⭐ Best |
| **NVIDIA GPU (no CUDA)** | `mandelbrot_opencl` | `make opencl && build/mandelbrot_opencl` | ⭐⭐⭐⭐ Excellent |
| **AMD GPU** | `mandelbrot_opencl` | `make opencl && build/mandelbrot_opencl` | ⭐⭐⭐⭐ Excellent |
| **Intel GPU** | `mandelbrot_opencl` | `make opencl && build/mandelbrot_opencl` | ⭐⭐⭐ Good |
| **No GPU/Issues** | `mandelbrot_cpu` | `make cpu && build/mandelbrot_cpu` | ⭐⭐ Baseline |

### Installation Summary

```bash
# For most users (OpenCL):
make install-deps-ubuntu && make opencl && build/mandelbrot_opencl

# For NVIDIA users wanting maximum performance (CUDA):
make install-cuda-ubuntu && make cuda && build/mandelbrot_cuda
```

## Acknowledgments

Inspired by Javidx9's Mandelbrot on olcPixelGameEngine 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit improvements:
- Additional kernel optimizations
- New color schemes
- Platform-specific enhancements
- Bug fixes and documentation updates
