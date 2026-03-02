# parallelbrot

GPU-accelerated Mandelbrot set renderer with OpenCL (cross-platform), CUDA (NVIDIA), and CPU backends.

![Mandelbrot Renderer Demo](docs/mandelbrot.gif)

## Build & Run

Install dependencies first (Ubuntu/Debian):

```bash
make install-deps-ubuntu   # OpenCL deps
make install-cuda-ubuntu   # CUDA deps (NVIDIA only)
```

Then build and run:

```bash
make opencl && build/mandelbrot_opencl   # Any GPU (recommended)
make cuda   && build/mandelbrot_cuda     # NVIDIA only (fastest)
make cpu    && build/mandelbrot_cpu      # CPU fallback
make clean                               # Remove build artifacts
```

For other distros, replace `ubuntu` with `fedora` or `arch`.

## Controls

| Key / Input | Action |
|---|---|
| Mouse drag | Pan |
| Mouse wheel | Zoom |
| Arrow keys | Pan |
| `+` / `-` | Increase / decrease iterations |
| `C` | Cycle color schemes |
| `R` | Reset view |
| `ESC` | Quit |

## Requirements

- OpenGL, GLFW3, GLEW
- **OpenCL**: OpenCL headers + GPU drivers (NVIDIA / AMD ROCm / Intel)
- **CUDA**: CUDA Toolkit 11.0+, NVIDIA GPU (Compute Capability 5.0+), drivers 470+

Check hardware support:

```bash
make check-opencl
make check-cuda
```

## License

MIT — see [LICENSE](LICENSE).
