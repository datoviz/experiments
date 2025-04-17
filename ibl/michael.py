import math
import numpy as np
import datoviz as dvz
from datoviz import Out


def make_texture(batch, image):
    assert image.ndim == 2
    format = dvz.FORMAT_R8_UNORM
    image *= .3
    normalized = dvz.to_byte(image, 0, 1)
    return dvz.tex_image(batch, format, image.shape[1], image.shape[0], normalized, 0)


def add_image(x, y, w, h, image, batch=None, panel=None):
    pos = np.array([[x, y, 0]], dtype=np.float32)
    size = np.array([[w, h]], dtype=np.float32)
    anchor = np.array([[0, 0]], dtype=np.float32)
    texcoords = np.array([[0, 0, 1, 1]], dtype=np.float32)
    tex = make_texture(batch, image)
    address_mode = dvz.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
    filter = dvz.FILTER_NEAREST

    visual = dvz.image(
        batch, dvz.IMAGE_FLAGS_SIZE_NDC | dvz.IMAGE_FLAGS_RESCALE | dvz.IMAGE_FLAGS_MODE_COLORMAP)
    dvz.image_alloc(visual, 1)
    dvz.image_position(visual, 0, 1, pos, 0)
    dvz.image_size(visual, 0, 1, size, 0)
    dvz.image_colormap(visual, dvz.CMAP_BINARY)
    dvz.image_anchor(visual, 0, 1, anchor, 0)
    dvz.image_texcoords(visual, 0, 1, texcoords, 0)
    dvz.image_texture(visual, tex, filter, address_mode)
    dvz.panel_visual(panel, visual, 0)


def get_extent():
    xmin = Out(0.0, 'double')
    xmax = Out(0.0, 'double')
    ymin = Out(0.0, 'double')
    ymax = Out(0.0, 'double')
    dvz.panzoom_bounds(panzoom, ref, xmin, xmax, ymin, ymax)


path = 'concat_ephysFalse.npy'
r = np.load(path, allow_pickle=True).flat[0]
data = r['concat_z'][r['isort']]
print(f"Loaded data with shape {data.shape}")

cols_beryl = np.load('cols_beryl.npy')
cols_beryl = cols_beryl[r['isort']]

FIG_SIZE = 800
app = dvz.app(dvz.APP_FLAGS_WHITE_BACKGROUND)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, FIG_SIZE, FIG_SIZE, 0)
panel = dvz.panel_default(figure)
panzoom = dvz.panel_panzoom(panel, 0)

x = -1
w = 2
y = 1
row = 4096  # maximum texture size
h = 2 * row / float(data.shape[0])
n = math.floor(data.shape[0] / float(row))
for i in range(n):
    add_image(x, y - i * h, w, h,
              data[i*row:(i+1)*row, :], batch=batch, panel=panel)

ref = dvz.ref(0)
dvz.ref_set(ref, dvz.DIM_X, 0, 1)
dvz.ref_set(ref, dvz.DIM_Y, 0, 1)

# @dvz.frame
# def on_frame(app, window_id, ev):
#     get_extent()
# dvz.app_onframe(app, on_frame, None)

dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
dvz.ref_destroy(ref)
