/*
	CPU Mandelbrot Renderer (for comparison)
	Simple CPU implementation to compare with OpenCL version
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>

class CPUMandelbrotRenderer {
private:
    // Window properties
    GLFWwindow* window;
    int window_width = 1280;
    int window_height = 720;
    
    // OpenGL objects
    GLuint vao, vbo, texture, shader_program;
    
    // CPU buffer for results
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
    CPUMandelbrotRenderer() {
        last_frame_time = std::chrono::high_resolution_clock::now();
        cpu_buffer.resize(window_width * window_height * 4); // RGBA
    }
    
    ~CPUMandelbrotRenderer() {
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
        
        window = glfwCreateWindow(window_width, window_height, "CPU Mandelbrot", nullptr, nullptr);
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
            std::string title = "CPU Mandelbrot - Frame: " + std::to_string(frame_time) + "ms - Iterations: " + std::to_string(max_iterations);
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
    
    // Color scheme selection
    int color_scheme = 1; // 0 = Ultra Fractal, 1 = Fire, 2 = Ocean, 3 = Psychedelic (Fire default)
    
    // Enhanced color mapping functions with multiple beautiful palettes
    
    void mapColor_UltraFractal(int iterations, int max_iterations, float& r, float& g, float& b) {
        if (iterations == max_iterations) {
            r = g = b = 0.0f; // Black for points in the set
            return;
        }
        
        float t = (float)iterations / (float)max_iterations;
        float smooth_t = t + 1.0f - log(log(sqrt(4.0f)))/log(2.0f); // Smooth coloring
        smooth_t = fmod(smooth_t * 0.05f, 1.0f); // Slower color cycling
        
        if (smooth_t < 0.16f) {
            // Deep blue to cyan
            float local_t = smooth_t / 0.16f;
            r = 0.0f;
            g = local_t * 0.7f;
            b = 0.5f + local_t * 0.5f;
        } else if (smooth_t < 0.42f) {
            // Cyan to yellow
            float local_t = (smooth_t - 0.16f) / 0.26f;
            r = local_t;
            g = 0.7f + local_t * 0.3f;
            b = 1.0f - local_t;
        } else if (smooth_t < 0.6425f) {
            // Yellow to red
            float local_t = (smooth_t - 0.42f) / 0.2225f;
            r = 1.0f;
            g = 1.0f - local_t * 0.5f;
            b = 0.0f;
        } else if (smooth_t < 0.8575f) {
            // Red to magenta
            float local_t = (smooth_t - 0.6425f) / 0.215f;
            r = 1.0f;
            g = local_t * 0.5f;
            b = local_t;
        } else {
            // Magenta back to blue
            float local_t = (smooth_t - 0.8575f) / 0.1425f;
            r = 1.0f - local_t;
            g = 0.5f - local_t * 0.5f;
            b = 1.0f;
        }
    }
    
    void mapColor_Fire(int iterations, int max_iterations, float& r, float& g, float& b) {
        if (iterations == max_iterations) {
            r = g = b = 0.0f;
            return;
        }
        
        float t = (float)iterations / (float)max_iterations;
        t = pow(t, 0.8f); // Enhance contrast
        
        if (t < 0.25f) {
            // Black to dark red
            float local_t = t / 0.25f;
            r = local_t * 0.8f;
            g = 0.0f;
            b = 0.0f;
        } else if (t < 0.5f) {
            // Dark red to bright red
            float local_t = (t - 0.25f) / 0.25f;
            r = 0.8f + local_t * 0.2f;
            g = local_t * 0.3f;
            b = 0.0f;
        } else if (t < 0.75f) {
            // Red to orange/yellow
            float local_t = (t - 0.5f) / 0.25f;
            r = 1.0f;
            g = 0.3f + local_t * 0.7f;
            b = local_t * 0.5f;
        } else {
            // Orange to white
            float local_t = (t - 0.75f) / 0.25f;
            r = 1.0f;
            g = 1.0f;
            b = 0.5f + local_t * 0.5f;
        }
    }
    
    void mapColor_Ocean(int iterations, int max_iterations, float& r, float& g, float& b) {
        if (iterations == max_iterations) {
            r = 0.0f; g = 0.0f; b = 0.1f; // Deep blue for set points
            return;
        }
        
        float t = (float)iterations / (float)max_iterations;
        t = sin(t * 3.14159f * 0.5f); // Smoother distribution
        
        r = 0.1f + t * (0.3f + 0.7f * sin(t * 6.28f));
        g = 0.2f + t * (0.6f + 0.4f * cos(t * 4.0f));
        b = 0.4f + t * (0.6f + 0.4f * sin(t * 8.0f));
    }
    
    void mapColor_Psychedelic(int iterations, int max_iterations, float& r, float& g, float& b) {
        if (iterations == max_iterations) {
            r = g = b = 0.0f;
            return;
        }
        
        float t = (float)iterations / (float)max_iterations;
        t = fmod(t * 3.0f, 1.0f); // Triple the frequency for more color bands
        
        r = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 0.0f);
        g = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 2.09f);
        b = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 4.19f);
        
        // Enhance saturation
        float max_val = std::max({r, g, b});
        if (max_val > 0.5f) {
            r = std::min(r * 1.2f, 1.0f);
            g = std::min(g * 1.2f, 1.0f);
            b = std::min(b * 1.2f, 1.0f);
        }
    }
    
    void mapColor(int iterations, int max_iterations, float& r, float& g, float& b) {
        switch (color_scheme) {
            case 0: mapColor_UltraFractal(iterations, max_iterations, r, g, b); break;
            case 1: mapColor_Fire(iterations, max_iterations, r, g, b); break;
            case 2: mapColor_Ocean(iterations, max_iterations, r, g, b); break;
            case 3: mapColor_Psychedelic(iterations, max_iterations, r, g, b); break;
            default: mapColor_UltraFractal(iterations, max_iterations, r, g, b); break;
        }
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
        // CPU computation
        double scale = 4.0 / zoom;
        double aspect_ratio = (double)window_width / (double)window_height;
        
        for (int y = 0; y < window_height; ++y) {
            for (int x = 0; x < window_width; ++x) {
                double real = center_x + scale * aspect_ratio * ((double)x / (double)window_width - 0.5);
                double imag = center_y + scale * ((double)y / (double)window_height - 0.5);
                
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
                float r, g, b;
                mapColor(iterations, max_iterations, r, g, b);
                
                // Write to buffer
                int index = (y * window_width + x) * 4;
                cpu_buffer[index + 0] = r;     // R
                cpu_buffer[index + 1] = g;     // G
                cpu_buffer[index + 2] = b;     // B
                cpu_buffer[index + 3] = 1.0f;  // A
            }
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
        auto* renderer = static_cast<CPUMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        renderer->window_width = width;
        renderer->window_height = height;
        renderer->cpu_buffer.resize(width * height * 4);
        glViewport(0, 0, width, height);
        
        // Recreate texture with new size
        glBindTexture(GL_TEXTURE_2D, renderer->texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        (void)xoffset; // Suppress unused warning
        auto* renderer = static_cast<CPUMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        double zoom_factor = (yoffset > 0) ? 1.2 : 0.8;
        renderer->zoom *= zoom_factor;
    }
    
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        (void)mods; // Suppress unused warning
        auto* renderer = static_cast<CPUMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
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
        auto* renderer = static_cast<CPUMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
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
        auto* renderer = static_cast<CPUMandelbrotRenderer*>(glfwGetWindowUserPointer(window));
        
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
                        case 0: std::cout << "Ultra Fractal (deep blues to oranges)\n"; break;
                        case 1: std::cout << "Fire (heat gradient)\n"; break;
                        case 2: std::cout << "Ocean (turquoise variations)\n"; break;
                        case 3: std::cout << "Psychedelic (enhanced rainbow)\n"; break;
                    }
                    break;
            }
        }
    }
};

int main() {
    CPUMandelbrotRenderer renderer;
    
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer\n";
        return -1;
    }
    
    std::cout << "CPU Mandelbrot Renderer\n";
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
