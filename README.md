# Mandelbrot GPU Renderer

✅ **Successfully converted** from CPU-based olcPixelGameEngine to **OpenGL** and **OpenCL** GPU acceleration!

A high-performance GPU-accelerated Mandelbrot set renderer using OpenCL for computation and OpenGL for rendering.



## Features

- **GPU Acceleration**: Uses OpenCL kernels for parallel computation on GPU
- **Hardware Rendering**: OpenGL for fast texture-based rendering
- **Interactive Navigation**: Real-time pan and zoom with mouse and keyboard
- **Multiple Kernels**: Different optimization levels and coloring schemes
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **High Performance**: Significantly faster than CPU-based implementations

## Controls

- **Mouse drag**: Pan around the fractal
- **Mouse wheel**: Zoom in/out
- **Arrow keys**: Pan with keyboard
- **+/-**: Increase/decrease iteration count
- **R**: Reset view to default
- **ESC**: Exit application

## Requirements

### System Dependencies

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

### Manual Installation

Required libraries:
- OpenGL development libraries
- OpenCL headers and runtime
- GLFW3 (window management)
- GLEW (OpenGL extension loading)

## Building

```bash
# Build the program
make

# Build with debug symbols
make debug

# Build optimized release version
make release

# Clean build files
make clean
```

## Running

```bash
# Run the GPU-accelerated version
./mandelbrot_simple

# Run the CPU comparison version
./mandelbrot_cpu

# Or use make target
make run
```

## OpenCL Device Support

This program works with:
- **NVIDIA GPUs**: Install NVIDIA OpenCL drivers
- **AMD GPUs**: Install ROCm or AMDGPU-PRO drivers
- **Intel GPUs**: Intel OpenCL runtime
- **CPU fallback**: Most OpenCL implementations include CPU support

Check your OpenCL devices:
```bash
make check-opencl
```

## Performance

The OpenCL implementation provides significant performance improvements over CPU implementations:

- **GPU vs CPU**: 10-100x faster depending on hardware
- **Parallel Processing**: Utilizes hundreds/thousands of GPU cores
- **Memory Bandwidth**: Leverages high GPU memory bandwidth
- **Real-time Interaction**: Smooth navigation even at high iteration counts

## Architecture

### Components

1. **Main Application** (`mandelbrot_simple.cpp`):
   - OpenGL setup and rendering
   - OpenCL initialization and execution
   - User input handling
   - Window management

2. **OpenCL Kernels** (`mandelbrot_kernel.cl`):
   - `mandelbrot_buffer`: GPU-parallel Mandelbrot computation
   - `mapColor`: Fractal coloring function
   - Optimized for GPU execution

3. **CPU Comparison** (`mandelbrot_cpu.cpp`):
   - CPU-only implementation for performance comparison
   - Same controls and functionality

### Data Flow

1. User input updates view parameters (center, zoom, iterations)
2. OpenCL kernel computes Mandelbrot set on GPU
3. Results written directly to OpenGL texture
4. OpenGL renders fullscreen quad with texture
5. Display updates in real-time

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
   - Check with `clinfo` command

2. **Compilation errors**:
   - Verify OpenCL headers are installed
   - Check compiler version (C++17 required)

3. **Runtime errors**:
   - Ensure OpenGL context is created before OpenCL
   - Check GPU memory availability

### Performance Issues

1. **Slow rendering**:
   - Reduce iteration count
   - Check if using integrated vs dedicated GPU
   - Monitor GPU memory usage

2. **Input lag**:
   - Enable VSync in graphics drivers
   - Reduce window size for testing

## License

Based on the original OneLoneCoder Mandelbrot example (OLC-3 License).
OpenCL/OpenGL implementation additions released under MIT License.

## Contributing

Feel free to submit improvements:
- Additional kernel optimizations
- New color schemes
- Platform-specific enhancements
- Bug fixes and documentation updates
