// Imports.
#include <datoviz_protocol.h>
#include <datoviz_types.h>
#include <stddef.h>


#define WIDTH  960
#define HEIGHT 400



struct Vertex
{
    vec3 pos;
    cvec4 color;
};



struct Context
{
    DvzBatch* batch;
    DvzId vertex_id;
    float w;
    float h;
    float x;
    float y;
};



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



static void set_shaders(DvzBatch* batch, DvzId graphics_id)
{

    // Vertex shader.
    char* vertex_glsl = read_file("vs.vert", NULL);
    DvzRequest req = dvz_create_glsl(batch, DVZ_SHADER_VERTEX, vertex_glsl);
    FREE(vertex_glsl);

    // Assign the shader to the graphics pipe.
    DvzId vertex_id = req.id;
    dvz_set_shader(batch, graphics_id, vertex_id);

    // Fragment shader.
    char* fragment_glsl = read_file("fs.frag", NULL);
    req = dvz_create_glsl(batch, DVZ_SHADER_FRAGMENT, fragment_glsl);
    FREE(fragment_glsl);

    // Assign the shader to the graphics pipe.
    DvzId fragment_id = req.id;
    dvz_set_shader(batch, graphics_id, fragment_id);
}



static DvzId create_pipeline(DvzBatch* batch)
{
    // Create a custom graphics.
    DvzRequest req = dvz_create_graphics(batch, DVZ_GRAPHICS_CUSTOM, 0);
    DvzId graphics_id = req.id;

    set_shaders(batch, graphics_id);

    // Primitive topology.
    dvz_set_primitive(batch, graphics_id, DVZ_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // Polygon mode.
    dvz_set_polygon(batch, graphics_id, DVZ_POLYGON_MODE_FILL);

    // Vertex binding.
    dvz_set_vertex(batch, graphics_id, 0, sizeof(struct Vertex), DVZ_VERTEX_INPUT_RATE_VERTEX);

    // Vertex attrs.
    dvz_set_attr(
        batch, graphics_id, 0, 0, DVZ_FORMAT_R32G32B32_SFLOAT, offsetof(struct Vertex, pos));
    dvz_set_attr(
        batch, graphics_id, 0, 1, DVZ_FORMAT_R8G8B8A8_UNORM, offsetof(struct Vertex, color));

    return graphics_id;
}



// In normalized device coordinates (whole window = [-1..+1]).
static void
upload_rectangle(DvzBatch* batch, DvzId vertex_id, vec2 offset, vec2 shape, cvec4 color)
{
    float x = offset[0];
    float y = offset[1];
    float w = shape[0];
    float h = shape[1];

    uint8_t r = color[0];
    uint8_t g = color[1];
    uint8_t b = color[2];
    uint8_t a = color[3];

    struct Vertex data[] = {

        // lower triangle
        {{x, y, 0}, {r, g, b, a}},
        {{x + w, y, 0}, {r, g, b, a}},
        {{x, y + h, 0}, {r, g, b, a}}, //

        // upper triangle
        {{x + w, y + h, 0}, {r, g, b, a}},
        {{x, y + h, 0}, {r, g, b, a}},
        {{x + w, y, 0}, {r, g, b, a}},

    };
    DvzRequest req = dvz_upload_dat(batch, vertex_id, 0, sizeof(data), data, 0);
}



static void _on_timer(DvzApp* app, DvzId window_id, DvzTimerEvent ev)
{
    struct Context* ctx = (struct Context*)ev.user_data;
    assert(ctx != NULL);

    // Reupload the vertex data with a different color at every timer tick.
    upload_rectangle(
        ctx->batch, ctx->vertex_id, (vec2){ctx->x, ctx->y}, (vec2){ctx->w, ctx->h},
        (ev.step_idx % 2) == 0 ? (cvec4){255, 0, 0, 255} : (cvec4){0, 255, 0, 255});

    // Submit the latest batch request(s).
    dvz_app_submit(app);
}



int main(int argc, char** argv)
{
    // Create app object.
    DvzApp* app = dvz_app(0);
    DvzBatch* batch = dvz_app_batch(app);
    DvzRequest req = {0};



    // Create a canvas.
    req = dvz_create_canvas(batch, WIDTH, HEIGHT, DVZ_DEFAULT_CLEAR_COLOR, 0);
    DvzId canvas_id = req.id;



    // Create the graphics pipeline.
    DvzId graphics_id = create_pipeline(batch);



    // Create the vertex buffer dat.
    req = dvz_create_dat(
        batch, DVZ_BUFFER_TYPE_VERTEX, 3 * sizeof(struct Vertex),
        DVZ_DAT_FLAGS_PERSISTENT_STAGING);
    DvzId vertex_id = req.id;
    req = dvz_bind_vertex(batch, graphics_id, 0, vertex_id, 0);

    // Upload a rectangle to the vertex buffer.
    float w = 100.0;
    float h = 100.0;
    float x = 1.0 - 2 * w / WIDTH;
    float y = 1.0 - 2 * h / HEIGHT;

    upload_rectangle(batch, vertex_id, (vec2){x, y}, (vec2){w, h}, (cvec4){0, 255, 255, 255});


    // Commands.
    uint32_t n_vertices = 6; // 2 triangles for the rectangle
    dvz_record_begin(batch, canvas_id);
    dvz_record_viewport(batch, canvas_id, DVZ_DEFAULT_VIEWPORT, DVZ_DEFAULT_VIEWPORT);
    dvz_record_draw(batch, canvas_id, graphics_id, 0, n_vertices, 0, 1);
    dvz_record_end(batch, canvas_id);


    struct Context ctx = {
        .batch = batch,
        .vertex_id = vertex_id, //
        .w = w,
        .h = h,
        .x = x,
        .y = y};
    float dt = 1. / 10; // 10 Hz update
    dvz_app_timer(app, 0, dt, 0);
    dvz_app_ontimer(app, _on_timer, &ctx);



    // Run the application.
    dvz_app_run(app, 0);

    // Cleanup.
    dvz_app_destroy(app);

    return 0;
}
