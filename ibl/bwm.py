from pathlib import Path
import urllib.request
from iblatlas import atlas
import numpy as np
from pywavefront import Wavefront
import datoviz as dvz
from datoviz import vec4


# def hex_to_rgba(arr, alpha=255):
#     arr = np.char.lstrip(arr.astype(str), '#')
#     rgba = np.zeros((arr.size, 4), dtype=np.uint8)
#     for i, hex_str in enumerate(arr):
#         if len(hex_str) == 6:
#             rgba[i, :3] = [int(hex_str[j:j+2], 16) for j in (0, 2, 4)]
#             rgba[i, 3] = alpha
#         elif len(hex_str) == 8:
#             rgba[i] = [int(hex_str[j:j+2], 16) for j in (0, 2, 4, 6)]
#         else:
#             raise ValueError(f"Invalid hex color: {hex_str}")
#     return rgba


CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'
CUR_DIR = Path(__file__).resolve().parent
bwm_path = CUR_DIR / 'bwm.npz'
a = atlas.AllenAtlas()


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


def add_points(batch, panel, pos, color, usize=5):
    n = pos.shape[0]
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    size = np.full(n, usize, dtype=np.float32)

    visual = dvz.point(batch, 0)
    dvz.visual_depth(visual, dvz.DEPTH_TEST_ENABLE)

    dvz.point_alloc(visual, n)
    dvz.point_position(visual, 0, n, pos.astype(np.float32), 0)
    dvz.point_color(visual, 0, n, color.astype(np.uint8), 0)
    dvz.point_size(visual, 0, n, size, 0)
    dvz.panel_visual(panel, visual, 0)

    return visual


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


# Application.
app = dvz.app(dvz.APP_FLAGS_WHITE_BACKGROUND)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, 1920, 1080, 0)
panel = dvz.panel_default(figure)
arcball = dvz.panel_arcball(panel)

add_points(batch, panel, cluster_pos, cluster_color)
add_mesh(batch, panel, mesh_pos, mesh_idx, mesh_color, alpha=32)

dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
