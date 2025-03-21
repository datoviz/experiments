#version 450

/*in vec3 fragmentColor;*/

layout(location = 0) in vec2 UV; // varying
// in vec2 UV;

layout(location = 0) out vec4 color;
// out vec4 color;


// Uniform struct shared by both shaders (a current limitation of Datoviz, that's why we put
// fragment shader stuff here).
layout(std140, binding = 0) uniform Uniform
{
    // WARNING: the variables should be sorted by decreasing size to avoid alignment issues.
    mat4 view;
    mat4 model;
    mat4 projection;
    vec4 maxColor;
    vec4 minColor;
    vec2 texOffset; /* offset the texture, degrees */
    vec2 texSize;   /* size of the texture, degrees */
    float texAngle; /* rotate the texture, degrees */

    // For fragment shader.
    /* float viewAngle;*/ /* rotation of view, degrees */
    /* vec2 pos;*/        /* position of layer [azimuth, altitude], degrees */
}
ubo;
layout(binding = 1) uniform sampler2D myTextureSampler;
// uniform vec4 maxColor;
// uniform vec4 minColor;
// uniform sampler2D myTextureSampler;



void main()
{
    /*color = vec4(1.0f, 1.0f, 1.0f, 1.0f);*/
    /*color = fragmentColor;*/
    /*color = vec4(1.0f, 1.0f, 1.0f, 1.0f);*/
    /*vec2 scale;
    scale.x = 360/size.x;
    scale.y = 180/size.y;*/

    color = texture(myTextureSampler, UV).rgba;
    color = color * (ubo.maxColor - ubo.minColor) + ubo.minColor;

    // DEBUG
    // color = vec4(uv, 1, 1);
}
