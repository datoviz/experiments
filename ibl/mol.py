import glob
import numpy as np
import datoviz as dvz
from datoviz import vec3, vec4


def load_data():
    # comes from: https://www.rcsb.org/structure/6QZP
    import MDAnalysis as mda
    files = sorted(glob.glob("mol/6qzp-pdb-bundle*.pdb"))
    universes = [mda.Universe(f) for f in files]
    print(set(atom.element for u in universes for atom in u.atoms))

    positions = np.concatenate([u.atoms.positions for u in universes], axis=0)

    element_colors_u8 = {
        "H":  (255, 255, 255),
        "C":  (200, 200, 200),
        "N":  (143, 143, 255),
        "O":  (240, 0, 0),
        "S":  (255, 200, 50),
        "P":  (255, 165, 0),
        "Mg": (42, 128, 42),
        "Zn": (165, 42, 42),
    }

    colors = np.concatenate([np.array([
        element_colors_u8[atom.element] for atom in u.atoms], dtype=np.uint8) for u in universes],
        axis=0)

    colors = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)

    vdw_radii = {
        "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
        "Mg": 1.73, "Zn": 1.39,
    }

    atomic_radii = np.concatenate([
        np.array([vdw_radii[atom.element] for atom in u.atoms])
        for u in universes
    ])

    sizes = 5 + 10 * (atomic_radii - atomic_radii.min()) / \
        (atomic_radii.max() - atomic_radii.min())
    sizes = sizes.astype(np.float32)

    positions -= positions.mean(axis=0)
    positions /= np.max(np.linalg.norm(positions, axis=1))

    return positions, colors, sizes


# positions, colors, sizes = load_data()
# np.savez("mol/mol.npz", positions=positions, colors=colors, sizes=sizes)

data = np.load("mol/mol.npz")
positions = data['positions']
colors = data['colors']
sizes = data['sizes']
N = len(positions)
print(f"Loaded {N} atoms")


app = dvz.app(0)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, 1920, 1080, 0)
panel = dvz.panel_default(figure)
arcball = dvz.panel_arcball(panel, 0)
camera = dvz.panel_camera(panel, 0)

visual = dvz.sphere(batch, 0)
dvz.sphere_alloc(visual, N)
dvz.sphere_position(visual, 0, N, positions.astype(np.float32), 0)
dvz.sphere_color(visual, 0, N, colors.astype(np.uint8), 0)
dvz.sphere_size(visual, 0, N, sizes.astype(np.float32), 0)
dvz.sphere_light_pos(visual, vec3(-5, +5, +100))
dvz.sphere_light_params(visual, vec4(.4, .8, 2, 32))
dvz.panel_visual(panel, visual, 0)


@dvz.on_timer
def _on_timer(app, window_id, ev):
    t = ev.contents.time
    z = 3 * np.exp(-.1 * np.mod(t, 10))
    angle = .025 * np.pi * 2 * t

    dvz.camera_position(camera, vec3(0, 0, z))
    dvz.arcball_set(arcball, vec3(.25 * angle, angle, 0))
    dvz.panel_update(panel)


# dvz.app_timer(app, 0, 1.0 / 60.0, 0)

dvz.app_on_timer(app, _on_timer, None)


dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
