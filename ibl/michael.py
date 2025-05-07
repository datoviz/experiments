import math
import numpy as np
import datoviz as dvz


def make_texture(app, image):
    assert image.ndim == 2
    image *= 0.3
    normalized = dvz.to_byte(image, 0, 1)
    return app.texture_2D(normalized)


def add_image(x, y, w, h, image, app=None, panel=None):
    pos = np.array([[x, y, 0]], dtype=np.float32)
    size = np.array([[w, h]], dtype=np.float32)
    anchor = np.array([[-1, 0]], dtype=np.float32)
    texcoords = np.array([[0, 0, 1, 1]], dtype=np.float32)
    texture = make_texture(app, image)

    visual = app.image(
        rescale=True,
        position=pos,
        size=size,
        anchor=anchor,
        texcoords=texcoords,
        unit="ndc",
        colormap="binary",
        texture=texture,
    )
    panel.add(visual)


path = "concat_ephysFalse.npy"
r = np.load(path, allow_pickle=True).flat[0]
data = r["concat_z"][r["isort"]]
print(f"Loaded data with shape {data.shape}")

# cols_beryl = np.load("cols_beryl.npy")
# cols_beryl = cols_beryl[r["isort"]]


app = dvz.App(background="white")
figure = app.figure()
panel = figure.panel()
panzoom = panel.panzoom()

x = -1
w = 2
y = 1
row = 4096  # maximum texture size
h = 2 * row / float(data.shape[0])
n = math.floor(data.shape[0] / float(row))
for i in range(n):
    add_image(x, y - i * h, w, h, data[i * row : (i + 1) * row, :], app=app, panel=panel)


app.run()
app.destroy()
