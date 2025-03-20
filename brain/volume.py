import gzip
from pathlib import Path
import numpy as np
import datoviz as dvz
from datoviz import vec2, vec3, vec4, S_, A_


MOUSE_W = 320
MOUSE_H = 456
MOUSE_D = 528


def load_volume(batch):
    path = '../../datoviz/data/volumes/allen_mouse_brain_rgba.npy.gz'
    with gzip.open(path, 'rb') as f:
        volume = np.load(f)
    format = dvz.FORMAT_R8G8B8A8_UNORM
    tex = dvz.tex_volume(batch, format, MOUSE_W, MOUSE_H, MOUSE_D, A_(volume))
    return tex


def add_volume(batch, panel):
    visual = dvz.volume(batch, dvz.VOLUME_FLAGS_RGBA)

    scaling = 1. / MOUSE_D
    x, y, z = MOUSE_W * scaling, MOUSE_H * scaling, 1
    dvz.volume_bounds(visual, vec2(-x, +x), vec2(-y, +y), vec2(-z, +z))
    dvz.panel_visual(panel, visual, 0)

    return visual


# Boilerplate.
app = dvz.app(0)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)


# Load the volume texture.
tex = load_volume(batch)


# Create a figure 800x600.
figure = dvz.figure(scene, 800, 600, 0)

# Panel spanning the entire window.
panel = dvz.panel_default(figure)

# Arcball interactivity.
arcball = dvz.panel_arcball(panel)

# Add the volume.
visual = add_volume(batch, panel)
dvz.volume_texture(visual, tex, dvz.FILTER_LINEAR, dvz.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)



# Initial arcball angles.
dvz.arcball_initial(arcball, vec3(-2.4, +.7, +1.5))
camera = dvz.panel_camera(panel, 0)
dvz.camera_initial(camera, vec3(0, 0, 1.5), vec3(0, 0, 0), vec3(0, 1, 0))
dvz.panel_update(panel)

# Run the application.
dvz.scene_run(scene, app, 0)

# Cleanup.
dvz.scene_destroy(scene)
dvz.app_destroy(app)
