
/*************************************************************************************************/
/*  Imports                                                                                      */
/*************************************************************************************************/

#include <stddef.h>

#include <cglm/cglm.h>

#include <datoviz_protocol.h>
#include <datoviz_types.h>



/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

#define WIDTH  960
#define HEIGHT 400



/*************************************************************************************************/
/*  Structs                                                                                      */
/*************************************************************************************************/

struct SquareVertex
{
    vec3 pos;
    cvec4 color;
};



struct SphereVertex
{
    vec3 vertexPos;
    vec2 vertexUV;
};



struct SphereUniform
{
    mat4 view;
    mat4 model;
    mat4 projection;
    /* float viewAngle;*/ /* rotation of view, degrees */
    /* vec2 pos;*/        /* position of layer [azimuth, altitude], degrees */
    float texAngle;       /* rotate the texture, degrees */
    vec2 texOffset;       /* offset the texture, degrees */
    vec2 texSize;         /* size of the texture, degrees */

    // For fragment shader.
    vec4 maxColor;
    vec4 minColor;
};



struct Context
{
    DvzBatch* batch;
    DvzId square_vertex;
    float w;
    float h;
    float x;
    float y;
};



/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

static void* read_file(const char* filename, DvzSize* size)
{
    /* The returned pointer must be freed by the caller. */

    void* buffer = NULL;
    DvzSize length = 0;
    FILE* f = fopen(filename, "rb");

    if (!f)
    {
        printf("Could not find %s.\n", filename);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    length = (DvzSize)ftell(f);
    if (size != NULL)
        *size = length;
    fseek(f, 0, SEEK_SET);
    buffer = (void*)calloc(1, (size_t)length);
    fread(buffer, 1, (size_t)length, f);
    fclose(f);

    return buffer;
}



static void set_shaders(
    DvzBatch* batch, DvzId graphics_id, const char* vertex_filename, const char* fragment_filename)
{
    // Vertex shader.
    char* vertex_glsl = read_file(vertex_filename, NULL);
    DvzRequest req = dvz_create_glsl(batch, DVZ_SHADER_VERTEX, vertex_glsl);
    FREE(vertex_glsl);

    // Assign the shader to the graphics pipe.
    DvzId square_vertex = req.id;
    dvz_set_shader(batch, graphics_id, square_vertex);

    // Fragment shader.
    char* fragment_glsl = read_file(fragment_filename, NULL);
    req = dvz_create_glsl(batch, DVZ_SHADER_FRAGMENT, fragment_glsl);
    FREE(fragment_glsl);

    // Assign the shader to the graphics pipe.
    DvzId fragment_id = req.id;
    dvz_set_shader(batch, graphics_id, fragment_id);
}



/*************************************************************************************************/
/*  Helpers                                                                                      */
/*************************************************************************************************/

static DvzId create_square_pipeline(DvzBatch* batch)
{
    // Create a custom graphics.
    DvzRequest req = dvz_create_graphics(batch, DVZ_GRAPHICS_CUSTOM, 0);
    DvzId graphics_id = req.id;

    set_shaders(batch, graphics_id, "vs.vert", "fs.frag");

    // Primitive topology.
    dvz_set_primitive(batch, graphics_id, DVZ_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // Polygon mode.
    dvz_set_polygon(batch, graphics_id, DVZ_POLYGON_MODE_FILL);

    // Vertex binding.
    dvz_set_vertex(
        batch, graphics_id, 0, sizeof(struct SquareVertex), DVZ_VERTEX_INPUT_RATE_VERTEX);

    // Vertex attrs.
    dvz_set_attr(
        batch, graphics_id, 0, 0, DVZ_FORMAT_R32G32B32_SFLOAT, offsetof(struct SquareVertex, pos));
    dvz_set_attr(
        batch, graphics_id, 0, 1, DVZ_FORMAT_R8G8B8A8_UNORM, offsetof(struct SquareVertex, color));

    return graphics_id;
}



static DvzId create_sphere_pipeline(DvzBatch* batch)
{
    // Create a custom graphics.
    DvzRequest req = dvz_create_graphics(batch, DVZ_GRAPHICS_CUSTOM, 0);
    DvzId graphics_id = req.id;

    set_shaders(batch, graphics_id, "slimshady_dvz.vert", "slimshady_dvz.frag");

    // Primitive topology.
    // dvz_set_primitive(batch, graphics_id, DVZ_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    dvz_set_primitive(batch, graphics_id, DVZ_PRIMITIVE_TOPOLOGY_POINT_LIST);
    dvz_set_blend(batch, graphics_id, DVZ_BLEND_STANDARD);
    dvz_set_cull(batch, graphics_id, DVZ_CULL_MODE_BACK);
    dvz_set_front(batch, graphics_id, DVZ_FRONT_FACE_CLOCKWISE);

    // Polygon mode.
    dvz_set_polygon(batch, graphics_id, DVZ_POLYGON_MODE_FILL);

    // Vertex binding.
    dvz_set_vertex(
        batch, graphics_id, 0, sizeof(struct SphereVertex), DVZ_VERTEX_INPUT_RATE_VERTEX);

    // Vertex attrs.
    dvz_set_attr(
        batch, graphics_id, 0, 0, //
        DVZ_FORMAT_R32G32B32_SFLOAT, offsetof(struct SphereVertex, vertexPos));

    dvz_set_attr(
        batch, graphics_id, 0, 1, //
        DVZ_FORMAT_R32G32_SFLOAT, offsetof(struct SphereVertex, vertexUV));

    // Slots.
    dvz_set_slot(batch, graphics_id, 0, DVZ_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    dvz_set_slot(batch, graphics_id, 1, DVZ_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    return graphics_id;
}



// In normalized device coordinates (whole window = [-1..+1]).
static void
upload_rectangle(DvzBatch* batch, DvzId square_vertex, vec2 offset, vec2 shape, cvec4 color)
{
    float x = offset[0];
    float y = offset[1];
    float w = shape[0];
    float h = shape[1];

    uint8_t r = color[0];
    uint8_t g = color[1];
    uint8_t b = color[2];
    uint8_t a = color[3];

    struct SquareVertex data[] = {

        // lower triangle
        {{x, y, 0}, {r, g, b, a}},
        {{x + w, y, 0}, {r, g, b, a}},
        {{x, y + h, 0}, {r, g, b, a}}, //

        // upper triangle
        {{x + w, y + h, 0}, {r, g, b, a}},
        {{x, y + h, 0}, {r, g, b, a}},
        {{x + w, y, 0}, {r, g, b, a}},

    };
    DvzRequest req = dvz_upload_dat(batch, square_vertex, 0, sizeof(data), data, 0);
}



/*************************************************************************************************/
/*  Callbacks                                                                                    */
/*************************************************************************************************/

static void _on_timer(DvzApp* app, DvzId window_id, DvzTimerEvent ev)
{
    struct Context* ctx = (struct Context*)ev.user_data;
    assert(ctx != NULL);

    // Reupload the vertex data with a different color at every timer tick.
    upload_rectangle(
        ctx->batch, ctx->square_vertex, (vec2){ctx->x, ctx->y}, (vec2){ctx->w, ctx->h},
        (ev.step_idx % 2) == 0 ? (cvec4){255, 0, 0, 255} : (cvec4){0, 255, 0, 255});

    // Submit the latest batch request(s).
    dvz_app_submit(app);
}



/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

int main(int argc, char** argv)
{
    // App.
    // --------------------------------------------------------------------------------------------

    DvzApp* app = dvz_app(0);
    DvzBatch* batch = dvz_app_batch(app);
    DvzRequest req = {0};



    // Canvas.
    // --------------------------------------------------------------------------------------------

    req = dvz_create_canvas(batch, WIDTH, HEIGHT, DVZ_DEFAULT_CLEAR_COLOR, 0);
    DvzId canvas_id = req.id;



    // Square.
    // --------------------------------------------------------------------------------------------

    // Create the graphics pipelines.
    DvzId square_graphics = create_square_pipeline(batch);
    uint32_t square_vertex_count = 6; // 2 triangles for the rectangle

    // Create the vertex buffer dat for the square.
    req = dvz_create_dat(
        batch, DVZ_BUFFER_TYPE_VERTEX, square_vertex_count * sizeof(struct SquareVertex),
        DVZ_DAT_FLAGS_PERSISTENT_STAGING);
    DvzId square_vertex = req.id;
    req = dvz_bind_vertex(batch, square_graphics, 0, square_vertex, 0);

    // Upload a rectangle to the vertex buffer.
    float w = 100.0;
    float h = 100.0;
    float x = 1.0 - 2 * w / WIDTH;
    float y = 1.0 - 2 * h / HEIGHT;

    upload_rectangle(batch, square_vertex, (vec2){x, y}, (vec2){w, h}, (cvec4){0, 255, 255, 255});



    // Sphere.
    //
    // --------------------------------------------------------------------------------------------

    DvzId sphere_graphics = create_sphere_pipeline(batch);


    // Create the vertex buffer dat for the sphere.
    uint32_t sphere_vertex_count = 20706;

    req = dvz_create_dat(
        batch, DVZ_BUFFER_TYPE_VERTEX, sphere_vertex_count * sizeof(struct SphereVertex), 0);
    DvzId sphere_vertex = req.id;
    req = dvz_bind_vertex(batch, sphere_graphics, 0, sphere_vertex, 0);

    // Upload vertex data.
    DvzSize vertex_buffer_size = 0;
    char* vertex_data = read_file("vertex", &vertex_buffer_size);
    assert(vertex_buffer_size > 0);
    assert(vertex_buffer_size == sphere_vertex_count * sizeof(struct SphereVertex));
    req = dvz_upload_dat(batch, sphere_vertex, 0, vertex_buffer_size, vertex_data, 0);
    FREE(vertex_data);


    // Create the index buffer dat for the sphere.
    uint32_t sphere_index_count = 124236;

    req = dvz_create_dat(batch, DVZ_BUFFER_TYPE_INDEX, sphere_index_count * sizeof(DvzIndex), 0);
    DvzId sphere_index = req.id;
    req = dvz_bind_index(batch, sphere_graphics, sphere_index, 0);

    // Upload index data.
    DvzSize index_buffer_size = 0;
    DvzIndex* index_data = (DvzIndex*)read_file("index", &index_buffer_size);
    assert(index_buffer_size > 0);
    assert(index_buffer_size == sphere_index_count * sizeof(DvzIndex));
    req = dvz_upload_dat(batch, sphere_index, 0, index_buffer_size, index_data, 0);
    FREE(index_data);



    // UBO.
    req = dvz_create_dat(
        batch, DVZ_BUFFER_TYPE_UNIFORM, sizeof(struct SphereUniform),
        DVZ_DAT_FLAGS_PERSISTENT_STAGING);
    DvzId ubo = req.id;
    req = dvz_bind_dat(batch, sphere_graphics, 0, ubo, 0);

    char* model_data = read_file("model", NULL);
    char* view_data = read_file("view", NULL);
    char* projection_data = read_file("projection", NULL);

    struct SphereUniform ubo_data = {
        .texAngle = 0,
        .texOffset = {0, 0},
        .texSize = {30, 30},

        .maxColor = {1},
        .minColor = {0},
    };
    memcpy(ubo_data.model, model_data, sizeof(mat4));
    memcpy(ubo_data.view, view_data, sizeof(mat4));
    memcpy(ubo_data.projection, projection_data, sizeof(mat4));
    // glm_mat4_print(ubo_data.projection, stdout);

    req = dvz_upload_dat(batch, ubo, 0, sizeof(struct SphereUniform), &ubo_data, 0);

    FREE(model_data);
    FREE(view_data);
    FREE(projection_data);



    // Texture.
    req = dvz_create_tex(batch, 2, DVZ_FORMAT_R8G8B8A8_UINT, (uvec3){3, 3, 1}, 0);
    DvzId tex = req.id;

    req = dvz_create_sampler(batch, DVZ_FILTER_LINEAR, DVZ_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    DvzId sampler = req.id;

    dvz_bind_tex(batch, sphere_graphics, 1, tex, sampler, (uvec3){0, 0, 0});

    DvzSize tex_size = 0;
    char* img = read_file("img", &tex_size);
    assert(tex_size == 3 * 3 * 4 * 1);
    req = dvz_upload_tex(batch, tex, (uvec3){0, 0, 0}, (uvec3){3, 3, 0}, tex_size, img, 0);
    FREE(img);



    // Commands.
    // --------------------------------------------------------------------------------------------

    dvz_record_begin(batch, canvas_id);

    // Viewport.
    dvz_record_viewport(batch, canvas_id, DVZ_DEFAULT_VIEWPORT, DVZ_DEFAULT_VIEWPORT);

    // Square.
    dvz_record_draw(batch, canvas_id, square_graphics, 0, square_vertex_count, 0, 1);

    // Sphere.
    dvz_record_draw_indexed(batch, canvas_id, sphere_graphics, 0, 0, sphere_index_count, 0, 1);
    // dvz_record_draw(batch, canvas_id, sphere_graphics, 0, sphere_vertex_count, 0, 1);

    dvz_record_end(batch, canvas_id);



    // Timer.
    // --------------------------------------------------------------------------------------------

    // struct Context ctx = {
    //     .batch = batch,
    //     .square_vertex = square_vertex, //
    //     .w = w,
    //     .h = h,
    //     .x = x,
    //     .y = y};
    // float dt = 1. / 10; // 10 Hz update
    // dvz_app_timer(app, 0, dt, 0);
    // dvz_app_ontimer(app, _on_timer, &ctx);



    // Run and close.
    // --------------------------------------------------------------------------------------------

    // Run the application.
    dvz_app_run(app, 0);

    // Cleanup.
    dvz_app_destroy(app);

    return 0;
}
