/*
    CUDA Mandelbrot Kernel
    High-performance GPU computation of the Mandelbrot set using CUDA
*/

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

// Color mapping function - creates beautiful fractal colors
__device__ float4 mapColor(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f); // Black for points in the set
    }
    
    float t = (float)iterations / (float)max_iterations;
    float r = 0.5f * sinf(3.0f * t) + 0.5f;
    float g = 0.5f * sinf(3.0f * t + 2.094f) + 0.5f; // 2*pi/3
    float b = 0.5f * sinf(3.0f * t + 4.188f) + 0.5f; // 4*pi/3
    
    return make_float4(r, g, b, 1.0f);
}

// Optimized Mandelbrot kernel with unrolled iterations
__device__ int mandelbrot_iterations(double real, double imag, int max_iterations) {
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;
    
    // Unroll the first few iterations for better performance
    #pragma unroll 4
    for (int i = 0; i < max_iterations && iterations < max_iterations; i += 4) {
        // Iteration 1
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 2
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 3
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 4
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
    }
    
    return iterations;
}

// CUDA kernel for Mandelbrot computation - buffer version
__global__ void mandelbrot_cuda_buffer(float4* output,
                                      int width,
                                      int height,
                                      double center_x,
                                      double center_y,
                                      double zoom,
                                      int max_iterations) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Compute Mandelbrot iterations
    int iterations = mandelbrot_iterations(real, imag, max_iterations);
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations);
    
    // Write to output buffer
    int index = y * width + x;
    output[index] = color;
}

// CUDA kernel for Mandelbrot computation - OpenGL interop version
__global__ void mandelbrot_cuda_texture(cudaSurfaceObject_t surface,
                                       int width,
                                       int height,
                                       double center_x,
                                       double center_y,
                                       double zoom,
                                       int max_iterations) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Compute Mandelbrot iterations
    int iterations = mandelbrot_iterations(real, imag, max_iterations);
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations);
    
    // Write to surface (OpenGL texture)
    surf2Dwrite(color, surface, x * sizeof(float4), y);
}

// Host function to launch the kernel
extern "C" {
    void launch_mandelbrot_kernel(float4* d_output,
                                 int width,
                                 int height,
                                 double center_x,
                                 double center_y,
                                 double zoom,
                                 int max_iterations) {
        
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        mandelbrot_cuda_buffer<<<gridSize, blockSize>>>(
            d_output, width, height, center_x, center_y, zoom, max_iterations
        );
        
        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }
    
    void launch_mandelbrot_texture_kernel(cudaSurfaceObject_t surface,
                                         int width,
                                         int height,
                                         double center_x,
                                         double center_y,
                                         double zoom,
                                         int max_iterations) {
        
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        mandelbrot_cuda_texture<<<gridSize, blockSize>>>(
            surface, width, height, center_x, center_y, zoom, max_iterations
        );
        
        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }
}
