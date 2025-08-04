/*
	OpenCL Mandelbrot Renderer (Simple Version)
	High-performance GPU-accelerated Mandelbrot set visualization
	Uses OpenCL for computation and CPU for display conversion
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

class MandelbrotRenderer {
private:
    // Window properties
    GLFWwindow* window;
    int window_width = 1280;
    int window_height = 720;
    
    // OpenGL objects
    GLuint vao, vbo, texture, shader_program;
    
    // OpenCL objects
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem cl_buffer;
    
    // CPU buffer for results
    std::vector<float> cpu_buffer;
    
    // Mandelbrot parameters
    double center_x = -0.5;
    double center_y = 0.0;
    double zoom = 1.0;
    int max_iterations = 128;
    int color_scheme = 1; // 0=UltraFractal, 1=Fire, 2=Ocean, 3=Psychedelic (Fire default)
    
    // Mouse interaction
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool mouse_dragging = false;
    
    // Performance timing
    std::chrono::high_resolution_clock::time_point last_frame_time;
    
public:
    MandelbrotRenderer() {
        last_frame_time = std::chrono::high_resolution_clock::now();
        cpu_buffer.resize(window_width * window_height * 4); // RGBA
    }
    
    ~MandelbrotRenderer() {
        cleanup();
    }
    
    bool initialize() {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW\n";
            return false;
        }
        
        // Create window
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(window_width, window_height, "OpenCL Mandelbrot", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        
        // Set callbacks
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetCursorPosCallback(window, cursor_pos_callback);
        glfwSetKeyCallback(window, key_callback);
        
        // Initialize GLEW
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW\n";
            return false;
        }
        
        // Setup OpenGL
        if (!setupOpenGL()) {
            return false;
        }
        
        // Setup OpenCL
        if (!setupOpenCL()) {
            return false;
        }
        
        return true;
    }
    
    void run() {
        while (!glfwWindowShouldClose(window)) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_frame_time).count() / 1000.0;
            last_frame_time = current_time;
            
            update();
            render();
            
            glfwSwapBuffers(window);
            glfwPollEvents();
            
            // Update window title with performance info
            std::string title = "OpenCL Mandelbrot - Frame: " + std::to_string(frame_time) + "ms - Iterations: " + std::to_string(max_iterations);
            glfwSetWindowTitle(window, title.c_str());
        }
    }
    
private:
    bool setupOpenGL() {
        // Create vertex array object
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        // Create fullscreen quad
        float vertices[] = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
            -1.0f,  1.0f, 0.0f, 1.0f
        };
        
        unsigned int indices[] = {
            0, 1, 2,
            0, 2, 3
        };
        
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        GLuint ebo;
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        // Create texture
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, window_width, window_height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        // Create and compile shaders
        shader_program = createShaderProgram();
        if (shader_program == 0) {
            return false;
        }
        
        return true;
    }
    
    bool setupOpenCL() {
        cl_int err;
        
        // Get platforms
        cl_uint num_platforms;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            std::cerr << "No OpenCL platforms found\n";
            return false;
        }
        
        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get OpenCL platforms\n";
            return false;
        }
        
        // Try to find NVIDIA platform first
        platform = platforms[0];
        for (const auto& plat : platforms) {
            char name[256];
            clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
            std::cout << "Found platform: " << name << std::endl;
            if (strstr(name, "NVIDIA") != nullptr) {
                platform = plat;
                std::cout << "Selected NVIDIA platform" << std::endl;
                break;
            }
        }
        
        // Get devices
        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
            if (err != CL_SUCCESS || num_devices == 0) {
                std::cerr << "No OpenCL devices found\n";
                return false;
            }
        }
        
        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
        }
        
        device = devices[0];
        
        // Print device info
        char device_name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        std::cout << "Using OpenCL device: " << device_name << std::endl;
        
        // Create simple context (no OpenGL interop)
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL context: " << err << std::endl;
            return false;
        }
        
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL command queue: " << err << std::endl;
            return false;
        }
        
        // Load and build kernel
        std::string kernel_source = loadKernelSource("src/opencl/mandelbrot_kernel.cl");
        if (kernel_source.empty()) {
            return false;
        }
        
        const char* source_ptr = kernel_source.c_str();
        size_t source_size = kernel_source.length();
        
        program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL program: " << err << std::endl;
            return false;
        }
        
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL compilation error: " << log.data() << std::endl;
            return false;
        }
        
        kernel = clCreateKernel(program, "mandelbrot_buffer", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL kernel: " << err << std::endl;
            return false;
        }
        
        // Create OpenCL buffer
        size_t buffer_size = window_width * window_height * 4 * sizeof(float);
        cl_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL buffer: " << err << std::endl;
            return false;
        }
        
        return true;
    }
    
    GLuint createShaderProgram() {
        const char* vertex_shader_source = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* fragment_shader_source = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            uniform sampler2D mandelbrotTexture;
            
            void main() {
                vec4 color = texture(mandelbrotTexture, TexCoord);
                FragColor = color;
            }
        )";
        
        GLuint vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_shader_source);
        GLuint fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_shader_source);
        
        if (vertex_shader == 0 || fragment_shader == 0) {
            return 0;
        }
        
        GLuint program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);
        
        int success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char info_log[512];
            glGetProgramInfoLog(program, 512, nullptr, info_log);
            std::cerr << "Shader program linking failed: " << info_log << std::endl;
            return 0;
        }
        
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        
        return program;
    }
    
    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        
        int success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char info_log[512];
            glGetShaderInfoLog(shader, 512, nullptr, info_log);
            std::cerr << "Shader compilation failed: " << info_log << std::endl;
            return 0;
        }
        
        return shader;
    }
    
    std::string loadKernelSource(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open kernel file: " << filename << std::endl;
            return "";
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }
    
    void update() {
        // Handle continuous key input for smooth movement
        const double pan_speed = 2.0 / (zoom * window_width);
        
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            center_x -= pan_speed;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            center_x += pan_speed;
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            center_y += pan_speed;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            center_y -= pan_speed;
        }
    }
    
    void render() {
        cl_int err;
        
        // Set kernel arguments
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &window_width);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &window_height);
        err |= clSetKernelArg(kernel, 3, sizeof(double), &center_x);
        err |= clSetKernelArg(kernel, 4, sizeof(double), &center_y);
        err |= clSetKernelArg(kernel, 5, sizeof(double), &zoom);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &max_iterations);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &color_scheme);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set kernel arguments: " << err << std::endl;
            return;
        }
        
        // Execute kernel
        size_t global_work_size[2] = {(size_t)window_width, (size_t)window_height};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to execute kernel: " << err << std::endl;
            return;
        }
        
        // Read results back to CPU
        err = clEnqueueReadBuffer(queue, cl_buffer, CL_TRUE, 0, 
                                 cpu_buffer.size() * sizeof(float), 
                                 cpu_buffer.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read buffer: " << err << std::endl;
            return;
        }
        
        // Update OpenGL texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, window_width, window_height, 
                    0, GL_RGBA, GL_FLOAT, cpu_buffer.data());
        
        // Render with OpenGL
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader_program);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shader_program, "mandelbrotTexture"), 0);
        
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    
    void cleanup() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (cl_buffer) clReleaseMemObject(cl_buffer);
        if (context) clReleaseContext(context);
        
        if (shader_program) glDeleteProgram(shader_program);
        if (texture) glDeleteTextures(1, &texture);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        
        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }
    
    // Static callback functions
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        renderer->window_width = width;
        renderer->window_height = height;
        renderer->cpu_buffer.resize(width * height * 4);
        glViewport(0, 0, width, height);
        
        // Recreate texture with new size
        glBindTexture(GL_TEXTURE_2D, renderer->texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        
        // Recreate OpenCL buffer
        if (renderer->cl_buffer) {
            clReleaseMemObject(renderer->cl_buffer);
        }
        cl_int err;
        size_t buffer_size = width * height * 4 * sizeof(float);
        renderer->cl_buffer = clCreateBuffer(renderer->context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to recreate OpenCL buffer: " << err << std::endl;
        }
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        (void)xoffset; // Suppress unused warning
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
        // Get current mouse position
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        
        // Convert mouse position to complex plane coordinates (before zoom)
        double aspect_ratio = (double)renderer->window_width / renderer->window_height;
        double scale = 2.0 / renderer->zoom;
        
        // Normalize mouse coordinates to [-1, 1] range
        double norm_x = (2.0 * mouse_x / renderer->window_width) - 1.0;
        double norm_y = 1.0 - (2.0 * mouse_y / renderer->window_height); // Flip Y coordinate
        
        // Convert to complex plane coordinates
        double complex_x = renderer->center_x + norm_x * scale * aspect_ratio;
        double complex_y = renderer->center_y + norm_y * scale;
        
        // Apply zoom
        double zoom_factor = (yoffset > 0) ? 1.2 : 0.8;
        renderer->zoom *= zoom_factor;
        
        // Adjust center so the point under cursor remains fixed
        double new_scale = 2.0 / renderer->zoom;
        renderer->center_x = complex_x - norm_x * new_scale * aspect_ratio;
        renderer->center_y = complex_y - norm_y * new_scale;
    }
    
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        (void)mods; // Suppress unused warning
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                renderer->mouse_dragging = true;
                glfwGetCursorPos(window, &renderer->last_mouse_x, &renderer->last_mouse_y);
            } else if (action == GLFW_RELEASE) {
                renderer->mouse_dragging = false;
            }
        }
    }
    
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        if (renderer->mouse_dragging) {
            double dx = xpos - renderer->last_mouse_x;
            double dy = ypos - renderer->last_mouse_y;
            
            double pan_scale = 2.0 / (renderer->zoom * renderer->window_width);
            renderer->center_x -= dx * pan_scale;
            renderer->center_y += dy * pan_scale;
            
            renderer->last_mouse_x = xpos;
            renderer->last_mouse_y = ypos;
        }
    }
    
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        (void)scancode; // Suppress unused warning
        (void)mods;     // Suppress unused warning
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            switch (key) {
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, true);
                    break;
                case GLFW_KEY_EQUAL:  // + key
                case GLFW_KEY_KP_ADD:
                    renderer->max_iterations += 32;
                    if (renderer->max_iterations > 1024) renderer->max_iterations = 1024;
                    break;
                case GLFW_KEY_MINUS:
                case GLFW_KEY_KP_SUBTRACT:
                    renderer->max_iterations -= 32;
                    if (renderer->max_iterations < 32) renderer->max_iterations = 32;
                    break;
                case GLFW_KEY_R:
                    // Reset view
                    renderer->center_x = -0.5;
                    renderer->center_y = 0.0;
                    renderer->zoom = 1.0;
                    break;
                case GLFW_KEY_C:
                    // Cycle through color schemes
                    renderer->color_scheme = (renderer->color_scheme + 1) % 4;
                    std::cout << "Color scheme: ";
                    switch (renderer->color_scheme) {
                        case 0: std::cout << "Ultra Fractal" << std::endl; break;
                        case 1: std::cout << "Fire" << std::endl; break;
                        case 2: std::cout << "Ocean" << std::endl; break;
                        case 3: std::cout << "Psychedelic" << std::endl; break;
                    }
                    break;
            }
        }
    }
};

int main() {
    MandelbrotRenderer renderer;
    
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer\n";
        return -1;
    }
    
    std::cout << "OpenCL Mandelbrot Renderer\n";
    std::cout << "Controls:\n";
    std::cout << "  Mouse drag: Pan\n";
    std::cout << "  Mouse wheel: Zoom\n";
    std::cout << "  Arrow keys: Pan\n";
    std::cout << "  +/-: Increase/decrease iterations\n";
    std::cout << "  C: Change color scheme\n";
    std::cout << "  R: Reset view\n";
    std::cout << "  ESC: Exit\n\n";
    
    renderer.run();
    
    return 0;
}
