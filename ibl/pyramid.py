from pathlib import Path
import numpy as np
import datoviz as dvz


CUR_DIR = Path(__file__).resolve().parent
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
    return Path(CUR_DIR / f"pyramid/res_{i:02d}.bin")


def load_file(res):
    fp = path(res)
    if not fp.exists():
        return
    size = fp.stat().st_size
    assert size % (n_channels * 2) == 0
    n_samples = int(round(size / (n_channels * 2)))
    out = np.memmap(fp, shape=(n_samples, n_channels), dtype=dtype, mode="r")
    print(f"Memmap file with shape {out.shape}")
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
    i0 = int(round(t0 * sample_rate / 2.0**res))
    i1 = int(round(t1 * sample_rate / 2.0**res))
    return i0, i1


# -------------------------------------------------------------------------------------------------


def make_texture(app, width, height):
    texture = app.texture(ndim=2, shape=(width, height), n_channels=1, dtype=np.uint8)
    return texture


def make_visual(x, y, w, h, texture, app=None):
    # x, y in data coordinates
    # w, h in NDC
    x = np.atleast_2d([x])
    y = np.atleast_2d([y])
    size = np.array([[w, h]])
    anchor = np.array([[-1.0, +1.0]])
    position = axes.normalize(x, y)
    # position = np.array([[-1, +1, 0]])
    visual = app.image(
        position=position,
        size=size,
        anchor=anchor,
        texture=texture,
        permutation=(1, 0),
        rescale=True,
        unit="ndc",
        mode="colormap",
        colormap="binary",
    )
    return visual


def update_image(res, i0, i1):
    if res < 0 or res > max_res:
        return None
    image = load_data(res, i0, i1)
    if image is None:
        print("Error loading data")
        return
    height, width = image.shape
    if height > tex_size:
        print("Texture too big")
        return None
    assert image.dtype == np.uint8
    assert width == n_channels
    assert height <= tex_size
    texture.data(image)

    t = (i1 - i0) / float(tex_size)
    texcoords = np.array([[0, 0, t, 1]], dtype=np.float32)
    visual.set_texcoords(texcoords)

    return True


def get_extent(axes):
    (xmin, xmax), (ymin, ymax) = axes.bounds()
    w = xmax - xmin
    k = 0.5
    xmin -= k * w
    xmax += k * w
    return (xmin, xmax, ymin, ymax)


def update_image_position(visual, axes):
    xmin, xmax, ymin, ymax = get_extent(axes)

    p = axes.normalize(np.array([[xmin], [xmax]]), np.array([[ymin], [ymax]]))
    w_ndc = p[1][0] - p[0][0]
    h_ndc = p[1][1] - p[0][1]

    x = np.array([[xmin]], dtype=np.float64)
    y = np.array([[n_channels]], dtype=np.float64)
    size = np.array([[w_ndc, h_ndc]], dtype=np.float32)

    visual.set_data(position=axes.normalize(x, y), size=size)


tmin, tmax = 1000.0, 1500.0
res = 11

app = dvz.App(background="white")
figure = app.figure()
panel = figure.panel()
panzoom = panel.panzoom(fixed="y")
axes = panel.axes((tmin, tmax), (0, n_channels))

texture = make_texture(app, n_channels, tex_size)
visual = make_visual(tmin, n_channels, 2.0, 2.0, texture, app=app)

i0, i1 = find_indices(res, tmin, tmax)
assert i1 - i0 <= tex_size
update_image(res, i0, i1)
panel.add(visual)


@app.connect(figure)
def on_frame(ev):
    global res, tmin, tmax
    new_tmin, new_tmax, _, _ = get_extent(axes)

    for new_res in range(0, max_res + 1):
        i0, i1 = find_indices(new_res, new_tmin, new_tmax)
        if i1 - i0 <= tex_size:
            break

    if new_res != res or np.abs((new_tmin - tmin) / (tmax - tmin)) > 0.25:
        i0, i1 = find_indices(new_res, new_tmin, new_tmax)
        if update_image(new_res, i0, i1) is None:
            return
        print(f"Update image, res {new_res}")
        update_image_position(visual, axes)
        visual.update()
        res = new_res
        tmin, tmax = new_tmin, new_tmax


app.run()
app.destroy()
