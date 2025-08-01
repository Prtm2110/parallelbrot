/*
    OpenCL Mandelbrot Kernel
    High-performance GPU computation of the Mandelbrot set
*/

// Color mapping function - creates beautiful fractal colors
float4 mapColor(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return (float4)(0.0f, 0.0f, 0.0f, 1.0f); // Black for points in the set
    }
    
    float t = (float)iterations / (float)max_iterations;
    float r = 0.5f * sin(3.0f * t) + 0.5f;
    float g = 0.5f * sin(3.0f * t + 2.094f) + 0.5f; // 2*pi/3
    float b = 0.5f * sin(3.0f * t + 4.188f) + 0.5f; // 4*pi/3
    
    return (float4)(r, g, b, 1.0f);
}

// Buffer version - outputs to a global memory buffer
__kernel void mandelbrot_buffer(__global float4* output,
                               const int width,
                               const int height,
                               const double center_x,
                               const double center_y, 
                               const double zoom,
                               const int max_iterations) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Mandelbrot iteration
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;
    
    while (iterations < max_iterations) {
        double z_real_sq = z_real * z_real;
        double z_imag_sq = z_imag * z_imag;
        
        if (z_real_sq + z_imag_sq > 4.0) {
            break;
        }
        
        double temp = z_real_sq - z_imag_sq + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        
        iterations++;
    }
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations);
    
    // Write to output buffer
    int index = y * width + x;
    output[index] = color;
}

__kernel void mandelbrot(__write_only image2d_t output,
                        const int width,
                        const int height,
                        const double center_x,
                        const double center_y, 
                        const double zoom,
                        const int max_iterations) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Mandelbrot iteration
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;
    
    while (iterations < max_iterations) {
        double z_real_sq = z_real * z_real;
        double z_imag_sq = z_imag * z_imag;
        
        if (z_real_sq + z_imag_sq > 4.0) {
            break;
        }
        
        double temp = z_real_sq - z_imag_sq + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        
        iterations++;
    }
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations);
    
    // Write to output image
    write_imagef(output, (int2)(x, y), color);
}
