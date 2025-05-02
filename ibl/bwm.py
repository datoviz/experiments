from pathlib import Path

import numpy as np

import datoviz as dvz


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


to_save = ['cluster_pos', 'cluster_color', 'mesh_pos', 'mesh_idx', 'mesh_color']
if not Path("bwm.npz").exists():

    import urllib.request
    from iblatlas import atlas

    from pywavefront import Wavefront

    CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
    CUR_DIR = Path(__file__).resolve().parent
    bwm_path = CUR_DIR / 'bwm.npz'
    a = atlas.AllenAtlas()

    # Load BWM data.
    bwm = np.load(bwm_path, allow_pickle=True)
    cluster_color = bwm['color']
    cluster_pos = np.ascontiguousarray(bwm['pos'], dtype=np.float64)
    idx = bwm['atlas_id'] > 0
    cluster_pos = cluster_pos[idx]
    cluster_color = cluster_color[idx]
    n = cluster_pos.shape[0]
    print(f"Loaded {n} clusters")

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
    cluster_pos -= center
    cluster_pos *= 200
    mesh_pos *= 200

    np.savez("bwm.npz", **{name: globals()[name] for name in to_save})

# Load data.
data = np.load("bwm.npz")
for name in to_save:
    globals()[name] = data[name]


app = dvz.App(background='white')
figure = app.figure()
panel = figure.panel()
arcball = panel.arcball()


# Points.
n = cluster_pos.shape[0]
cluster_pos = np.ascontiguousarray(cluster_pos, dtype=np.float32)
size = np.full(n, 5, dtype=np.float32)

visual = app.point(
    depth_test=True,
    position=cluster_pos,
    color=cluster_color,
    size=size,
)
panel.add(visual)


# Mesh.
nv = mesh_pos.shape[0]
ni = mesh_idx.size

mesh_pos = np.ascontiguousarray(mesh_pos, dtype=np.float32)
mesh_color = np.tile(mesh_color, (nv, 1)).astype(np.uint8)
mesh_color[:, 3] = 32
mesh_idx = mesh_idx.astype(np.uint32).ravel()

visual = app.mesh(indexed=True, lighting=True, cull='back')
visual.set_data(
    position=mesh_pos,
    color=mesh_color,
    index=mesh_idx,
    compute_normals=True,
)
panel.add(visual)


app.run()
app.destroy()
