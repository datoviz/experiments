import ctypes
import gzip
from pathlib import Path
import urllib.request

import numpy as np
from pywavefront import Wavefront
import datoviz as dvz
from datoviz import vec2, ivec3, vec3, vec4, S_, A_, V_


CCF_URL = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'

# MOUSE_W = 320
# MOUSE_H = 456
# MOUSE_D = 528

from iblatlas import atlas
a = atlas.AllenAtlas()

xmin, xmax = a.bc.xlim  # ml
ymin, ymax = a.bc.ylim  # ap
zmin, zmax = a.bc.zlim  # dv

nx, ny, nz = a.bc.nxyz  # 456, 528, 320

xc = .5 * (xmin + xmax)
yc = .5 * (ymin + ymax)
zc = .5 * (zmin + zmax)

dx = xmax - xmin
dy = ymax - ymin
dz = zmax - zmin


def norm_x(u):
    return (u - xc) * 50

def norm_y(u):
    return (u - yc) * 50

def norm_z(u):
    return (u - zc) * 50


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


def start():
    app = dvz.app(0)
    batch = dvz.app_batch(app)
    scene = dvz.scene(batch)
    figure = dvz.figure(scene, 1200, 1024, dvz.CANVAS_FLAGS_IMGUI)
    panel = dvz.panel_default(figure)
    arcball = dvz.panel_arcball(panel)

    return app, batch, scene, figure, panel, arcball


def run(app, scene):
    dvz.scene_run(scene, app, 0)
    dvz.scene_destroy(scene)
    dvz.app_destroy(app)


def load_mesh(region_idx=315):
    fn = f"mesh/mesh-{region_idx:03d}.npz"
    try:
        m = np.load(fn)
        print(f"Loaded mesh {fn}")
    except IOError:
        from iblatlas import atlas, regions
        a = atlas.AllenAtlas()
        br = regions.BrainRegions()
        pos, idx, color = load_region(region_idx, br=br)

        m = dict(pos=pos, idx=idx, color=color)
        np.savez(fn, **m)
        print(f"Saved {fn}")
    return m


def load_volume(batch):
    path = '../../datoviz/data/volumes/allen_mouse_brain_rgba.npy.gz'
    with gzip.open(path, 'rb') as f:
        volume = np.load(f)
    # volume shape (528, 456, 320, 4) ap=y ml=x dv=z
    # nx, ny, nz = (456, 528, 320)
    format = dvz.FORMAT_R8G8B8A8_UNORM
    tex = dvz.tex_volume(batch, format, nz, nx, ny, A_(volume))
    return tex


def add_volume(batch, panel):
    global visual
    visual = dvz.volume(batch, dvz.VOLUME_FLAGS_RGBA)

    dvz.volume_bounds(
        visual,
        vec2(norm_x(xmin), norm_x(xmax)),
        vec2(norm_y(ymin), norm_y(ymax)),
        vec2(norm_z(zmin), norm_z(zmax)),
        )

    dvz.volume_texcoords(visual, vec3(0, 1, 0), vec3(1, 0, 1))
    dvz.volume_transfer(visual, vec4(.5, 0, 0, 0))
    dvz.volume_permutation(visual, ivec3(2, 0, 1))
    dvz.panel_visual(panel, visual, 0)

    return visual


def add_mesh(batch, panel, pos, idx, color):
    nv = pos.shape[0]
    ni = idx.size

    pos = pos.astype(np.float32)
    color = np.tile(color, (nv, 1)).astype(np.uint8)
    # color[:, 3] = 127
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

    dvz.panel_visual(panel, visual, 0)

    return visual


visual = None
slider = V_(0, ctypes.c_float)


@dvz.gui
def ongui(app, fid, ev):
    dvz.gui_size(vec2(170, 110))
    dvz.gui_begin(S_("Slider GUI"), 0)
    with slider:
        if dvz.gui_slider(S_("Slider"), 0, 1, slider.P_):
            s = slider.value

            dvz.volume_bounds(
                visual,
                vec2(norm_x(xmin), norm_x(xmax)),
                vec2(norm_y(ymin + s * dy), norm_y(ymax)),
                vec2(norm_z(zmin), norm_z(zmax)),
            )
            dvz.volume_texcoords(visual, vec3(0, 1, 0), vec3(1, s, 1))

            # NOTE: this is sort of working (displaying the opaque slice) but we'll need
            # to implement multipass rendering to enable depth testing and proper transparency
            # so this works fully.
            # if s > 0:
            #     dvz.volume_slice(visual, 4)
    dvz.gui_end()


if __name__ == '__main__':
    region_idx = 128  # 315
    m = load_mesh(region_idx=region_idx)

    pos = m['pos']
    idx = m['idx']
    color = m['color']

    # pos is in CCF, p in xyz
    p = a.ccf2xyz(pos, ccf_order='apdvml')
    # Datoviz normalization into normalized device coordinates, and transposition to match the
    # volume.
    x, y, z = p.T
    xx = norm_x(x)
    yy = norm_y(y)
    zz = norm_z(z)
    p2 = np.c_[xx, yy, zz].copy()


    # Start the app.
    app, batch, scene, figure, panel, arcball = start()

    # Volume.
    tex = load_volume(batch)
    visual = add_volume(batch, panel)
    dvz.volume_texture(visual, tex, dvz.FILTER_LINEAR, dvz.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)

    # Mesh.
    add_mesh(batch, panel, p2, idx, color)

    # Initial arcball angles.
    dvz.arcball_initial(arcball, vec3(-.93, .01, 2.36))
    camera = dvz.panel_camera(panel, 0)
    dvz.arcball_gui(arcball, app, dvz.figure_id(figure), panel)
    dvz.camera_initial(camera, vec3(0, 0, 1.5), vec3(0, 0, 0), vec3(0, 1, 0))
    dvz.panel_update(panel)

    dvz.app_gui(app, dvz.figure_id(figure), ongui, None)

    # Run and exit the app.
    run(app, scene)
