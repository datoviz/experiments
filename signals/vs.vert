#version 450

// layout(std140, binding = USER_BINDING) uniform BasicParams { float size; }
// params;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 out_color;

void main()
{
    gl_Position = vec4(pos, 1);
    out_color = color;
}
