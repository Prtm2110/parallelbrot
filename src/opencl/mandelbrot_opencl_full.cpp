/*
	OpenGL + OpenCL Mandelbrot Renderer
	High-performance GPU-accelerated Mandelbrot set visualization
	
	Features:
	- OpenCL compute kernels for GPU acceleration
	- OpenGL for hardware-accelerated rendering
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
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#include <GLFW/glfw3native.h>
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
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
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer cl_output_buffer;
    cl::ImageGL cl_texture;
    
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
    MandelbrotRenderer() {
        last_frame_time = std::chrono::high_resolution_clock::now();
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
        try {
            // Get platforms
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            
            if (platforms.empty()) {
                std::cerr << "No OpenCL platforms found\n";
                return false;
            }
            
            platform = platforms[0];
            
            // Get devices
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            if (devices.empty()) {
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            }
            
            if (devices.empty()) {
                std::cerr << "No OpenCL devices found\n";
                return false;
            }
            
            device = devices[0];
            
            std::cout << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            
            // Create context with OpenGL interop
            cl_context_properties properties[] = {
                CL_GL_CONTEXT_KHR, (cl_context_properties)glfwGetCurrentContext(),
                CL_GLX_DISPLAY_KHR, (cl_context_properties)glfwGetX11Display(),
                CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
                0
            };
            
            context = cl::Context(device, properties);
            queue = cl::CommandQueue(context, device);
            
            // Load and build kernel
            std::string kernel_source = loadKernelSource("src/opencl/mandelbrot_kernel.cl");
            if (kernel_source.empty()) {
                return false;
            }
            
            program = cl::Program(context, kernel_source);
            
            try {
                program.build();
            } catch (const cl::Error& err) {
                std::cerr << "OpenCL compilation error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
                return false;
            }
            
            kernel = cl::Kernel(program, "mandelbrot");
            
            // Create OpenCL image from OpenGL texture
            cl_texture = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture);
            
        } catch (const cl::Error& err) {
            std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")\n";
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
        try {
            // Acquire OpenGL texture
            std::vector<cl::Memory> gl_objects = {cl_texture};
            queue.enqueueAcquireGLObjects(&gl_objects);
            
            // Set kernel arguments
            kernel.setArg(0, cl_texture);
            kernel.setArg(1, window_width);
            kernel.setArg(2, window_height);
            kernel.setArg(3, center_x);
            kernel.setArg(4, center_y);
            kernel.setArg(5, zoom);
            kernel.setArg(6, max_iterations);
            
            // Execute kernel
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, 
                                     cl::NDRange(window_width, window_height), 
                                     cl::NullRange);
            
            // Release OpenGL texture
            queue.enqueueReleaseGLObjects(&gl_objects);
            queue.finish();
            
        } catch (const cl::Error& err) {
            std::cerr << "OpenCL execution error: " << err.what() << " (" << err.err() << ")\n";
        }
        
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
        glViewport(0, 0, width, height);
        
        // Recreate texture with new size
        glBindTexture(GL_TEXTURE_2D, renderer->texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        
        // Recreate OpenCL texture
        try {
            renderer->cl_texture = cl::ImageGL(renderer->context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, renderer->texture);
        } catch (const cl::Error& err) {
            std::cerr << "Failed to recreate OpenCL texture: " << err.what() << std::endl;
        }
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        auto* renderer = static_cast<MandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        double zoom_factor = (yoffset > 0) ? 1.2 : 0.8;
        renderer->zoom *= zoom_factor;
    }
    
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
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
    std::cout << "  R: Reset view\n";
    std::cout << "  ESC: Exit\n\n";
    
    renderer.run();
    
    return 0;
}
