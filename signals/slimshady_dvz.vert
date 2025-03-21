#version 450

const float pi = 3.1415926535897932384626433832795;
const vec3 xax = vec3(1.0f, 0.0f, 0.0f);
const vec3 yax = vec3(0.0f, 1.0f, 0.0f);
const vec3 zax = vec3(0.0f, 0.0f, 1.0f);

mat3 trans2(vec2 v);
mat3 scale2(vec2 v);
mat3 rot2(float angle);
mat4 rot3(vec3 axis, float angle);
mat3 uvScaleOffsetMatrix(vec2 scale, vec2 off);


layout(location = 0) in vec3 vertexPos; // attributes
layout(location = 1) in vec2 vertexUV;

layout(location = 0) out vec2 UV; // varying, with "out"


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


void main()
{
    /*vec2 posRad = pos*pi/180.0;*/
    /*float viewRad = viewAngle*pi/180.0;*/
    /*mat4 view = rot3(zax, posRad.y)*rot3(yax, posRad.x)*rot3(xax, viewRad);*/
    /*mat4 view = rot3(yax, posRad.x)*rot3(zax, posRad.y)*rot3(xax, viewRad);*/

    float texAngle = ubo.texAngle;
    vec2 texOffset = ubo.texOffset;
    vec2 texSize = ubo.texSize;

    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vertexPos.xyz, 1.0f);

    // Vulkan conversion.
    gl_Position.y *= -1.0;
    gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;

    // DEBUG
    // gl_Position = vec4(vertexPos.xyz, 1.0f);
    // gl_PointSize = 2;

    vec2 safeTexSize =
        vec2(texSize.x != 0.0f ? texSize.x : 1e-10, texSize.y != 0.0f ? texSize.y : 1e-10);
    vec2 texScale = vec2(180.0 / safeTexSize.x, 180.0 / safeTexSize.y);
    vec2 texTrans = vec2(-texOffset.x / safeTexSize.x, -texOffset.y / safeTexSize.y);
    mat3 uvTrans = trans2(vec2(0.5) + texTrans) * scale2(texScale) * rot2(texAngle * pi / 180) *
                   scale2(vec2(2.0, 1.0)) * trans2(vec2(-0.5));
    UV = (uvTrans * vec3(vertexUV.xy, 1.0f)).xy;

    // DEBUG
    // UV = vertexUV;
}



mat3 uvScaleOffsetMatrix(vec2 scale, vec2 off)
{
    return mat3(
        scale.x, 0.0, 0.0,                                              /*U scaling column*/
        0.0, scale.y, 0.0,                                              /*V scaling column*/
        0.5 * (1 - scale.x) + off.x, 0.5 * (1 - scale.y) + off.y, 1.0); /*translate column*/
}

mat3 scale2(vec2 s) { return mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, 1.0); }

mat3 trans2(vec2 v) { return mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, v.x, v.y, 1.0); }

mat3 rot2(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, s, 0, -s, c, 0, 0, 0, 1);
}

mat4 rot3(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(
        oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s, 0.0, oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.0,
        oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s,
        oc * axis.z * axis.z + c, 0.0, 0.0, 0.0, 0.0, 1.0);
}
