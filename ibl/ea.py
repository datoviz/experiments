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


def add_mesh(batch, panel, pos, idx, color, alpha=255):
    nv = pos.shape[0]
    ni = idx.size

    pos = np.ascontiguousarray(pos, dtype=np.float32)
    color = np.tile(color, (nv, 1)).astype(np.uint8)
    color[:, 3] = alpha
    idx = idx.astype(np.uint32).ravel()

    normals = np.zeros((nv, 3), dtype=np.float32)
    dvz.compute_normals(nv, ni, pos, idx, normals)

    flags = dvz.VISUAL_FLAGS_INDEXED | dvz.MESH_FLAGS_LIGHTING

    visual = dvz.mesh(batch, flags)
    dvz.mesh_alloc(visual, nv, ni)
    dvz.mesh_position(visual, 0, nv, pos, 0)
    dvz.mesh_color(visual, 0, nv, color, 0)
    dvz.mesh_normal(visual, 0, nv, normals, 0)
    dvz.mesh_index(visual, 0, ni, idx, 0)

    # dvz.visual_depth(visual, dvz.DEPTH_TEST_DISABLE)
    dvz.visual_cull(visual, dvz.CULL_MODE_BACK)
    # dvz.visual_blend(visual, dvz.BLEND_OIT)
    # dvz.mesh_light_params(visual, 0, vec4(.75, .1, .1, 16))

    dvz.panel_visual(panel, visual, 0)

    return visual


def add_points(batch, panel, pos, size):
    n = pos.shape[0]
    pos = np.ascontiguousarray(pos, dtype=np.float32)

    visual = dvz.point(batch, 0)
    dvz.visual_depth(visual, dvz.DEPTH_TEST_ENABLE)

    dvz.point_alloc(visual, n)
    dvz.point_position(visual, 0, n, pos.astype(np.float32), 0)
    dvz.point_size(visual, 0, n, size.astype(np.float32), 0)
    dvz.panel_visual(panel, visual, 0)

    return visual


def set_feature(idx):
    feature = features[idx]
    values = df_voltage[feature]
    a, b = np.quantile(values, quantile), np.quantile(values, 1-quantile)
    channel_color = dvz.cmap(cmap, values, a, b)
    channel_color[:, 3] = point_alpha
    dvz.point_color(points, 0, channel_color.shape[0], channel_color.astype(np.uint8), 0)


def set_label(text):
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


@dvz.on_gui
def on_gui(app, fid, ev):
    dvz.gui_size(vec2(300, 100))
    dvz.gui_begin("GUI", 0)
    if dvz.gui_dropdown("Feature", len(features), list(map(str, features)), selected, 0):
        set_feature(selected.value)
        set_label(features[selected.value])
    dvz.gui_end()


@dvz.on_keyboard
def on_keyboard(app, window_id, ev):
    ev = ev.contents
    if ev.type in (dvz.KEYBOARD_EVENT_PRESS, dvz.KEYBOARD_EVENT_REPEAT):
        if ev.key == dvz.KEY_LEFT:
            selected.value = len(features) - 1 if selected.value == 0 else selected.value - 1
            set_feature(selected.value)
            set_label(features[selected.value])
        elif ev.key == dvz.KEY_RIGHT:
            selected.value = 0 if selected.value == len(features) - 1 else selected.value + 1
            set_feature(selected.value)
            set_label(features[selected.value])


# Application.
app = dvz.app(dvz.APP_FLAGS_WHITE_BACKGROUND)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, 1920, 1080, dvz.CANVAS_FLAGS_IMGUI)
panel = dvz.panel_default(figure)
arcball = dvz.panel_arcball(panel)

points = add_points(batch, panel, channel_pos, channel_size)
mesh = add_mesh(batch, panel, mesh_pos, mesh_idx, mesh_color, alpha=32)
set_feature(0)

glyph = dvz.glyph(batch, 0)
dvz.visual_fixed(glyph, True, True, True)
af = dvz.atlas_font(font_size)
dvz.glyph_atlas_font(glyph, af)
set_label(features[0])
dvz.panel_visual(panel, glyph, 0)


dvz.app_gui(app, dvz.figure_id(figure), on_gui, None)
dvz.app_on_keyboard(app, on_keyboard, None)

dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
