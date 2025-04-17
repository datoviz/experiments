from pathlib import Path
import numpy as np
import datoviz as dvz
from datoviz import Out, uvec3, ivec2


n_channels = 384
sample_rate = 2500
dtype = np.float16
vmin = -5e-4
vmax = +5e-4
tex_size = 2048
max_res = 11


# bounds = {}
files = {}


def path(i):
    return Path(f"pyramid/res_{i:02d}.bin")


def load_file(res):
    fp = path(res)
    if not fp.exists():
        return
    size = fp.stat().st_size
    assert size % (n_channels * 2) == 0
    n_samples = int(round(size / (n_channels * 2)))
    out = np.memmap(fp, shape=(n_samples, n_channels), dtype=dtype, mode='r')
    return out


def safe_slice(data, i0, i1, fill_value=0):
    n = i1 - i0
    shape = (n,) + data.shape[1:]
    result = np.full(shape, fill_value, dtype=data.dtype)

    s0 = max(i0, 0)
    s1 = min(i1, data.shape[0])
    d0 = s0 - i0
    d1 = d0 + (s1 - s0)

    result[d0:d1] = data[s0:s1]
    return result


def load_data(res, i0, i1):
    assert i0 < i1
    if res not in files:
        files[res] = load_file(res)
    data = files[res]
    if data is None or data.size == 0:
        return
    # out = data[i0:i1, :]
    out = safe_slice(data, i0, i1)
    if out.size == 0:
        return
    vmin, vmax = (-5e-4, +5e-4)
    return dvz.to_byte(out, vmin, vmax)


def find_indices(res, t0, t1):
    assert res >= 0
    assert res <= max_res
    res, t0, t1
    assert t0 < t1
    i0 = int(round(t0 * sample_rate / 2.0 ** res))
    i1 = int(round(t1 * sample_rate / 2.0 ** res))
    return i0, i1


def make_texture(batch, width, height):
    format = dvz.FORMAT_R8_UNORM
    tex = dvz.create_tex(batch, dvz.TEX_2D, format, uvec3(width, height, 1), 0).id
    return tex


def make_visual(x, y, w, h, tex, batch=None, panel=None):
    pos = np.array([[x, y, 0]], dtype=np.float32)
    size = np.array([[w, h]], dtype=np.float32)
    anchor = np.array([[0, 0]], dtype=np.float32)

    address_mode = dvz.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
    filter = dvz.FILTER_NEAREST

    visual = dvz.image(
        batch, dvz.IMAGE_FLAGS_SIZE_NDC | dvz.IMAGE_FLAGS_RESCALE | dvz.IMAGE_FLAGS_MODE_COLORMAP)
    dvz.image_alloc(visual, 1)
    dvz.image_position(visual, 0, 1, pos, 0)
    dvz.image_size(visual, 0, 1, size, 0)
    dvz.image_colormap(visual, dvz.CMAP_BINARY)
    dvz.image_permutation(visual, ivec2(1, 0))
    dvz.image_anchor(visual, 0, 1, anchor, 0)
    dvz.image_texture(visual, tex, filter, address_mode)

    return visual


def update_image(res, i0, i1):
    if res < 0 or res > max_res:
        return None
    image = load_data(res, i0, i1)
    if image is None:
        print("Error loading data")
        return
    height, width = image.shape
    if (height > tex_size):
        print("Texture too big")
        return None
    assert image.dtype == np.uint8
    assert width == n_channels
    assert height <= tex_size
    dvz.upload_tex(batch, tex, uvec3(0, 0, 0), uvec3(width, height, 1), image.size, image, 0)

    t = (i1 - i0) / float(tex_size)
    texcoords = np.array([[0, 0, t, 1]], dtype=np.float32)
    dvz.image_texcoords(visual, 0, 1, texcoords, 0)

    return True


def get_extent(ref, panzoom=None):
    xmin = Out(0.0, 'double')
    xmax = Out(0.0, 'double')
    ymin = Out(0.0, 'double')
    ymax = Out(0.0, 'double')
    dvz.panzoom_bounds(panzoom, ref, xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = xmin.value, xmax.value, ymin.value, ymax.value
    w = xmax - xmin
    k = .5
    xmin -= k * w
    xmax += k * w
    return (xmin, xmax, ymin, ymax)


def update_image_position(visual, ref_ndc, panzoom=None):
    xmin, xmax, _, _ = get_extent(ref_ndc, panzoom=panzoom)
    x = xmin
    w = xmax - xmin

    pos = np.array([[x, 1, 0]], dtype=np.float32)
    size = np.array([[w, 2]], dtype=np.float32)

    dvz.image_position(visual, 0, 1, pos, 0)
    dvz.image_size(visual, 0, 1, size, 0)


tmin, tmax = 1000, 1500
res = 11

app = dvz.app(dvz.APP_FLAGS_WHITE_BACKGROUND)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
flags = dvz.CANVAS_FLAGS_IMGUI
figure = dvz.figure(scene, 1200, 600, flags)
panel = dvz.panel_default(figure)
panzoom = dvz.panel_panzoom(panel)
dvz.panzoom_flags(panzoom, dvz.PANZOOM_FLAGS_FIXED_Y)

ref = dvz.ref(0)
dvz.ref_set(ref, dvz.DIM_X, tmin, tmax)
dvz.ref_set(ref, dvz.DIM_Y, 0, n_channels)

ref_ndc = dvz.ref(0)
dvz.ref_set(ref_ndc, dvz.DIM_X, -1, 1)
dvz.ref_set(ref_ndc, dvz.DIM_Y, -1, 1)

tex = make_texture(batch, n_channels, tex_size)
visual = make_visual(-1, 1, 2, 2, tex, batch=batch, panel=panel)

i0, i1 = find_indices(res, tmin, tmax)
assert i1 - i0 <= tex_size
update_image(res, i0, i1)
dvz.panel_visual(panel, visual, 0)


@dvz.frame
def onframe(app, fid, ev):
    global res, tmin, tmax
    new_tmin, new_tmax, _, _ = get_extent(ref, panzoom=panzoom)

    # zoom = dvz.panzoom_level(panzoom, dvz.DIM_X)
    # new_res = int(np.clip(round(max_res - np.log2(max(1, zoom))) - 3, 0, max_res))

    for new_res in range(0, max_res + 1):
        i0, i1 = find_indices(new_res, new_tmin, new_tmax)
        if i1 - i0 <= tex_size:
            break

    if new_res != res or np.abs((new_tmin - tmin) / (tmax - tmin)) > .25:
        i0, i1 = find_indices(new_res, new_tmin, new_tmax)
        if update_image(new_res, i0, i1) is None:
            return
        print(f"Update image, res {new_res}")
        update_image_position(visual, ref_ndc, panzoom=panzoom)
        dvz.visual_update(visual)
        res = new_res
        tmin, tmax = new_tmin, new_tmax


dvz.app_onframe(app, onframe, None)

dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
dvz.ref_destroy(ref)
