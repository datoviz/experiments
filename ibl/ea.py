from datoviz import Out, vec2, cvec4
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import datoviz as dvz


mapping = 'Allen'
label = 'latest'
hemisphere = 'left'
point_alpha = 255
cmap = dvz.CMAP_MAGMA
quantile = .01
point_size = 3
font_size = 64


def lateralize_features(df):
    for c in df.columns:
        if c.startswith('atlas_id'):
            df[c] = -df[c].abs()
    return df


def dl(atlas_id):
    mesh_url = CCF_URL + str(atlas_id) + '.obj'
    fn = f"{atlas_id}.obj"
    try:
        urllib.request.urlretrieve(mesh_url, fn)
    except Exception as e:
        print(f"Error: {str(e)}")


def get_color(atlas_id, alpha=255, br=None):
    _, idx = br.id2index(atlas_id)
    color = br.rgb[idx[0][0], :]
    return np.hstack((color, [alpha])).astype(np.uint8)


def load_obj(path):
    obj = Wavefront(path, collect_faces=True, create_materials=True)
    pos = np.array(obj.vertices)
    idx = np.array(obj.mesh_list[0].faces)
    return pos, idx


def load_region(atlas_id, br=None):
    pos, idx = load_obj(f"obj/{atlas_id}.obj")
    color = get_color(atlas_id, br=br)
    return pos, idx, color


def load_mesh(region_idx=315):
    fn = f"mesh/mesh-{region_idx:03d}.npz"
    try:
        m = np.load(fn)
        # print(f"Loaded mesh {fn}")
    except IOError:
        from iblatlas import regions
        br = regions.BrainRegions()
        pos, idx, color = load_region(region_idx, br=br)

        m = dict(pos=pos, idx=idx, color=color)
        np.savez(fn, **m)
        print(f"Saved {fn}")
    return m


def set_feature(idx):
    feature = features[idx]
    values = df_voltage[feature]
    a, b = np.quantile(values, quantile), np.quantile(values, 1-quantile)
    channel_color = dvz.cmap(cmap, values, a, b)
    channel_color[:, 3] = point_alpha
    point.set_color(channel_color)


def set_label(text):
    # TODO
    return
    dvz.glyph_strings(
        glyph, 1, [text],
        np.array([[0, 1, 0]], dtype=np.float32), np.array([1], np.float32),
        cvec4(16, 16, 16, 255), vec2(0, -50), vec2(.5, .5),
    )


to_save = ['features', 'channel_pos', 'channel_size', 'mesh_pos', 'mesh_idx', 'mesh_color']

if not Path("ea.npz").exists():

    from ephys_atlas.features import voltage_features_set
    from ephys_atlas.data import load_voltage_features
    from iblatlas.regions import BrainRegions
    import urllib.request
    from iblatlas import atlas
    from pywavefront import Wavefront

    CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
    a = atlas.AllenAtlas()
    br = BrainRegions()
    local_data_path = Path('/home/cyrille/GIT/IBL/paper-ephys-atlas/data')
    df_voltage, df_channels, df_channels, df_probes = \
        load_voltage_features(local_data_path.joinpath(label), mapping=mapping)
    df_voltage = lateralize_features(df_voltage)
    df_voltage.drop(
        df_voltage[df_voltage[mapping + "_acronym"].isin(["void", "root"])].index, inplace=True)
    channel_pos = np.ascontiguousarray(df_voltage[['x', 'y', 'z']], dtype=np.float32)
    n = channel_pos.shape[0]
    print(f"Loaded {n} channels")

    features = sorted(voltage_features_set())
    channel_size = np.full(n, point_size, dtype=np.float32)

    # Load mesh.
    region_idx = 997
    m = load_mesh(region_idx=region_idx)
    mesh_pos = m['pos']
    mesh_idx = m['idx']
    mesh_color = m['color']
    mesh_pos = a.ccf2xyz(mesh_pos, ccf_order='apdvml')
    mesh_pos = np.ascontiguousarray(mesh_pos)
    print(f"Loaded mesh with {mesh_pos.shape[0]} vertices and {mesh_idx.shape[0] // 3} faces")

    # Normalization.
    center = mesh_pos.mean(axis=0)
    mesh_pos -= center
    channel_pos -= center
    channel_pos *= 200
    mesh_pos *= 200

    df_voltage.to_pickle("ea.pkl")
    np.savez("ea.npz", **{name: globals()[name] for name in to_save})


# Load data.
df_voltage = pd.read_pickle('ea.pkl')
data = np.load("ea.npz", allow_pickle=True)
for name in to_save:
    globals()[name] = data[name]


selected = Out(0)


app = dvz.App(background='white')
figure = app.figure(gui=True)
panel = figure.panel()
arcball = panel.arcball()


# Points.
n = channel_pos.shape[0]
channel_pos = np.ascontiguousarray(channel_pos, dtype=np.float32)

point = app.point(
    depth_test=True,
    position=channel_pos,
    size=channel_size,
)
panel.add(point)


# Mesh.
nv = mesh_pos.shape[0]
ni = mesh_idx.size

mesh_pos = np.ascontiguousarray(mesh_pos, dtype=np.float32)
mesh_color = np.tile(mesh_color, (nv, 1)).astype(np.uint8)
mesh_color[:, 3] = 32
mesh_idx = mesh_idx.astype(np.uint32).ravel()

mesh = app.mesh(indexed=True, lighting=True, cull='back')
mesh.set_data(
    position=mesh_pos,
    color=mesh_color,
    index=mesh_idx,
    compute_normals=True,
)
panel.add(mesh)


set_feature(0)


@app.connect(figure)
def on_gui(ev):
    dvz.gui_size(vec2(300, 100))
    dvz.gui_begin("GUI", 0)
    if dvz.gui_dropdown("Feature", len(features), list(map(str, features)), selected, 0):
        set_feature(selected.value)
        set_label(features[selected.value])
    dvz.gui_end()


@app.connect(figure)
def on_keyboard(ev):
    if ev.key_event() in ('press', 'repeat'):
        if ev.key_name() == 'left':
            selected.value = len(features) - 1 if selected.value == 0 else selected.value - 1
            set_feature(selected.value)
            set_label(features[selected.value])
        elif ev.key_name() == 'right':
            selected.value = 0 if selected.value == len(features) - 1 else selected.value + 1
            set_feature(selected.value)
            set_label(features[selected.value])


app.run()
app.destroy()


# TODO
# glyph = dvz.glyph(batch, 0)
# dvz.visual_fixed(glyph, True, True, True)
# af = dvz.AtlasFont()
# dvz.atlas_font(font_size, af)
# dvz.glyph_atlas_font(glyph, af)
# set_label(features[0])
# dvz.panel_visual(panel, glyph, 0)
