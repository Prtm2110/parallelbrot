/*
	CUDA Mandelbrot Renderer
	High-performance GPU-accelerated Mandelbrot set visualization using CUDA
	
	Features:
	- CUDA compute kernels for maximum GPU performance
	- OpenGL for hardware-accelerated rendering
	- CUDA-OpenGL interoperability for direct GPU-to-GPU data transfer
	- Interactive pan and zoom
	- Real-time parameter adjustment
	
	Controls:
	- Mouse drag: Pan
	- Mouse wheel: Zoom
	- +/-: Increase/decrease iterations
	- ESC: Exit
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// Forward declarations for CUDA kernel functions
extern "C" {
    void launch_mandelbrot_kernel(float4* d_output,
                                 int width,
                                 int height,
                                 double center_x,
                                 double center_y,
                                 double zoom,
                                 int max_iterations);
}

class CUDAMandelbrotRenderer {
private:
    // Window properties
    GLFWwindow* window;
    int window_width = 1280;
    int window_height = 720;
    
    // OpenGL objects
    GLuint vao, vbo, texture, shader_program;
    
    // CUDA objects
    float4* d_output_buffer;
    cudaGraphicsResource* cuda_gl_resource;
    cudaArray* cuda_array;
    bool use_gl_interop = true;
    
    // CPU buffer for non-interop mode
    std::vector<float> cpu_buffer;
    
    // Mandelbrot parameters
    double center_x = -0.5;
    double center_y = 0.0;
    double zoom = 1.0;
    int max_iterations = 128;
    
    // Mouse interaction
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool mouse_dragging = false;
    
    // Performance timing
    std::chrono::high_resolution_clock::time_point last_frame_time;
    
public:
    CUDAMandelbrotRenderer() {
        last_frame_time = std::chrono::high_resolution_clock::now();
        cpu_buffer.resize(window_width * window_height * 4); // RGBA
        d_output_buffer = nullptr;
        cuda_gl_resource = nullptr;
        cuda_array = nullptr;
    }
    
    ~CUDAMandelbrotRenderer() {
        cleanup();
    }
    
    bool initialize() {
        // Initialize CUDA first
        if (!setupCUDA()) {
            return false;
        }
        
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW\n";
            return false;
        }
        
        // Create window
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(window_width, window_height, "CUDA Mandelbrot", nullptr, nullptr);
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
        
        // Setup CUDA-OpenGL interop
        if (!setupCUDAGL()) {
            std::cout << "CUDA-OpenGL interop failed, falling back to buffer mode\n";
            use_gl_interop = false;
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
            std::string title = "CUDA Mandelbrot - Frame: " + std::to_string(frame_time) + "ms - Iterations: " + std::to_string(max_iterations);
            glfwSetWindowTitle(window, title.c_str());
        }
    }
    
private:
    bool setupCUDA() {
        cudaError_t error;
        
        // Initialize CUDA
        error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            std::cerr << "CUDA initialization failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Get device properties
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        
        // Allocate device memory for output buffer
        size_t buffer_size = window_width * window_height * sizeof(float4);
        error = cudaMalloc(&d_output_buffer, buffer_size);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        return true;
    }
    
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
    
    bool setupCUDAGL() {
        cudaError_t error;
        
        // Register OpenGL texture with CUDA
        error = cudaGraphicsGLRegisterImage(&cuda_gl_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (error != cudaSuccess) {
            std::cerr << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        std::cout << "CUDA-OpenGL interoperability enabled\n";
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
        
        // Check linking
        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
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
        
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
            return 0;
        }
        
        return shader;
    }
    
    void update() {
        // Launch CUDA kernel to compute Mandelbrot set
        launch_mandelbrot_kernel(
            d_output_buffer,
            window_width,
            window_height,
            center_x,
            center_y,
            zoom,
            max_iterations
        );
        
        // Copy result to OpenGL texture
        if (use_gl_interop) {
            // Use CUDA-OpenGL interop (faster)
            cudaGraphicsMapResources(1, &cuda_gl_resource);
            cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_gl_resource, 0, 0);
            
            // Copy from device buffer to OpenGL texture via CUDA array
            cudaMemcpy2DToArray(cuda_array, 0, 0, d_output_buffer, 
                               window_width * sizeof(float4),
                               window_width * sizeof(float4),
                               window_height,
                               cudaMemcpyDeviceToDevice);
            
            cudaGraphicsUnmapResources(1, &cuda_gl_resource);
        } else {
            // Fallback: copy to CPU then to OpenGL
            cudaMemcpy(cpu_buffer.data(), d_output_buffer, 
                      window_width * window_height * sizeof(float4), 
                      cudaMemcpyDeviceToHost);
            
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height, 
                           GL_RGBA, GL_FLOAT, cpu_buffer.data());
        }
    }
    
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(shader_program);
        glBindVertexArray(vao);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        // Set texture uniform
        glUniform1i(glGetUniformLocation(shader_program, "mandelbrotTexture"), 0);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    
    void cleanup() {
        if (d_output_buffer) {
            cudaFree(d_output_buffer);
            d_output_buffer = nullptr;
        }
        
        if (cuda_gl_resource) {
            cudaGraphicsUnregisterResource(cuda_gl_resource);
            cuda_gl_resource = nullptr;
        }
        
        if (shader_program) {
            glDeleteProgram(shader_program);
        }
        if (texture) {
            glDeleteTextures(1, &texture);
        }
        if (vbo) {
            glDeleteBuffers(1, &vbo);
        }
        if (vao) {
            glDeleteVertexArrays(1, &vao);
        }
        
        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }
    
    // Static callback functions
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        CUDAMandelbrotRenderer* renderer = static_cast<CUDAMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        renderer->window_width = width;
        renderer->window_height = height;
        
        // Reallocate CUDA buffer and texture
        // Note: In a full implementation, you'd want to handle this properly
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        CUDAMandelbrotRenderer* renderer = static_cast<CUDAMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
        double zoom_factor = 1.1;
        if (yoffset > 0) {
            renderer->zoom *= zoom_factor;
        } else {
            renderer->zoom /= zoom_factor;
        }
    }
    
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        CUDAMandelbrotRenderer* renderer = static_cast<CUDAMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
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
        CUDAMandelbrotRenderer* renderer = static_cast<CUDAMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
        if (renderer->mouse_dragging) {
            double dx = xpos - renderer->last_mouse_x;
            double dy = ypos - renderer->last_mouse_y;
            
            double scale = 4.0 / renderer->zoom;
            double aspect_ratio = (double)renderer->window_width / (double)renderer->window_height;
            
            renderer->center_x -= scale * aspect_ratio * dx / renderer->window_width;
            renderer->center_y += scale * dy / renderer->window_height;
            
            renderer->last_mouse_x = xpos;
            renderer->last_mouse_y = ypos;
        }
    }
    
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        CUDAMandelbrotRenderer* renderer = static_cast<CUDAMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            switch (key) {
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, true);
                    break;
                case GLFW_KEY_EQUAL:
                case GLFW_KEY_KP_ADD:
                    renderer->max_iterations = std::min(renderer->max_iterations + 32, 4096);
                    break;
                case GLFW_KEY_MINUS:
                case GLFW_KEY_KP_SUBTRACT:
                    renderer->max_iterations = std::max(renderer->max_iterations - 32, 32);
                    break;
                case GLFW_KEY_R:
                    renderer->center_x = -0.5;
                    renderer->center_y = 0.0;
                    renderer->zoom = 1.0;
                    renderer->max_iterations = 128;
                    break;
                case GLFW_KEY_LEFT:
                    renderer->center_x -= 0.1 / renderer->zoom;
                    break;
                case GLFW_KEY_RIGHT:
                    renderer->center_x += 0.1 / renderer->zoom;
                    break;
                case GLFW_KEY_UP:
                    renderer->center_y += 0.1 / renderer->zoom;
                    break;
                case GLFW_KEY_DOWN:
                    renderer->center_y -= 0.1 / renderer->zoom;
                    break;
            }
        }
    }
};

int main() {
    CUDAMandelbrotRenderer renderer;
    
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize CUDA Mandelbrot renderer\n";
        return -1;
    }
    
    std::cout << "CUDA Mandelbrot Renderer\n";
    std::cout << "Controls:\n";
    std::cout << "  Mouse drag: Pan\n";
    std::cout << "  Mouse wheel: Zoom\n";
    std::cout << "  Arrow keys: Pan\n";
    std::cout << "  +/-: Increase/decrease iterations\n";
    std::cout << "  R: Reset view\n";
    std::cout << "  ESC: Exit\n\n";
    
    renderer.run();
    
    return 0;
}
