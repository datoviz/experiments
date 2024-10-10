from pathlib import Path
import urllib.request

import numpy as np
from pywavefront import Wavefront
import datoviz as dvz
from datoviz import vec3, vec4, S_, A_


CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'

MOUSE_W = 320
MOUSE_H = 456
MOUSE_D = 528


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

    pos -= pos.mean(axis=0)
    pos /= np.abs(pos).max()

    return pos, idx


def load_region(atlas_id, br=None):
    pos, idx = load_obj(f"obj/{atlas_id}.obj")
    color = get_color(atlas_id, br=br)
    return pos, idx, color


def start():
    app = dvz.app(0)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    figure = dvz.figure(scene, 1200, 1024, 0)
    panel = dvz.panel_default(figure)
    arcball = dvz.panel_arcball(panel)

    return app, batch, scene, figure, panel, arcball


def run(app, scene):
    dvz.scene_run(scene, app, 0)
    dvz.scene_destroy(scene)
    dvz.app_destroy(app)


def add_mesh(batch, panel, pos, idx, color):
    nv = pos.shape[0]
    ni = idx.size

    pos = pos.astype(np.float32)
    color = np.tile(color, (nv, 1)).astype(np.uint8)
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

    dvz.mesh_light_pos(visual, vec3(-1, +1, +10))
    dvz.mesh_light_params(visual, vec4(.5, .5, .5, 16))

    dvz.panel_visual(panel, visual, 0)

    return visual


def load_mesh(region_idx=315):
    try:
        m = np.load("mesh.npz")
        print("Loaded mesh.npz")
    except IOError:
        from iblatlas import atlas, regions
        a = atlas.AllenAtlas()
        br = regions.BrainRegions()
        ids = a.regions.id
        pos, idx, color = load_region(region_idx, br=br)

        m = dict(pos=pos, idx=idx, color=color)
        np.savez("mesh.npz", **m)
        print("Saved mesh.npz")
    return m


def load_volume(batch):
    path = '../../datoviz/data/volumes/allen_mouse_brain_rgba.npy'
    volume = np.load(path)
    format = dvz.FORMAT_R8G8B8A8_UNORM
    tex = dvz.tex_volume(batch, format, MOUSE_W, MOUSE_H, MOUSE_D, A_(volume))
    return tex


def add_volume(batch, panel):
    visual = dvz.volume(batch, dvz.VOLUME_FLAGS_RGBA)
    dvz.volume_alloc(visual, 1)

    scaling = 1. / MOUSE_D
    dvz.volume_size(visual, MOUSE_W * scaling, MOUSE_H * scaling, MOUSE_D * scaling)
    dvz.panel_visual(panel, visual, 0)

    return visual


if __name__ == '__main__':
    m = load_mesh()
    pos = m['pos']
    idx = m['idx']
    color = m['color']

    # import matplotlib.pyplot as plt
    # plt.plot(pos[:, 2], -pos[:, 1], ',')
    # plt.axis('equal')
    # plt.show()
    # exit()

    pos *= .39
    pos[:, 0], pos[:, 1], pos[:, 2] = pos[:, 1].copy(), pos[:, 2].copy(), pos[:, 0].copy()
    pos[:, 0] -= .1

    # Start the app.
    app, batch, scene, figure, panel, arcball = start()

    # Volume.
    tex = load_volume(batch)
    visual = add_volume(batch, panel)
    dvz.volume_texture(visual, tex, dvz.FILTER_LINEAR, dvz.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)

    # Mesh.
    add_mesh(batch, panel, pos, idx, color)

    # Initial arcball angles.
    dvz.arcball_initial(arcball, vec3(-2.4, +.7, +1.5))
    camera = dvz.panel_camera(panel, 0)
    dvz.camera_initial(camera, vec3(0, 0, 1.5), vec3(0, 0, 0), vec3(0, 1, 0))
    dvz.panel_update(panel)

    # Run and exit the app.
    run(app, scene)
