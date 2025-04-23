from datoviz import vec2, vec3, Out
import datoviz as dvz
import numpy as np
from iblatlas import atlas, regions

a = atlas.AllenAtlas()
br = regions.BrainRegions()

n = br.n_lr
count = n - 1  # we remove void, the first item
levels = br.level[1:n].astype(np.uint32)
names = br.name[1:n].tolist()
acronyms = br.acronym[1:n].tolist()
colors = br.rgba[1:n].astype(np.uint8)
folded = np.zeros(n - 1, dtype=np.bool)
selected = np.zeros(n - 1, dtype=np.bool)
visible = np.ones(n - 1, dtype=np.bool)
haystack = np.array([f"{a} {n}".lower() for a, n in zip(acronyms, names)])
folded[levels >= 5] = True


app = dvz.app(0)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, 800, 600, dvz.CANVAS_FLAGS_IMGUI)

search = dvz.CStringBuffer("")


@dvz.on_gui
def on_gui(app, fid, ev):
    dvz.gui_pos(vec2(25, 25), vec2(0, 0))
    dvz.gui_size(vec2(500, 500))
    dvz.gui_begin("My GUI", 0)

    if dvz.gui_textbox("Search", 64, search, 0):
        s = search.value.lower()
        if s:
            visible[:] = np.char.find(haystack, s) != -1
        else:
            visible[:] = 1

    dvz.gui_tree(count, acronyms, names, levels, colors, folded, selected, visible)
    dvz.gui_end()


# panel1 = dvz.panel(figure, 50, 50, 300, 300)
# dvz.demo_panel3D(panel1)
# dvz.panel_gui(panel1, "Panel 1", 0)

# panel2 = dvz.panel(figure, 400, 100, 300, 300)
# dvz.demo_panel2D(panel2)
# dvz.panel_gui(panel2, "Panel 2", 0)

dvz.app_gui(app, dvz.figure_id(figure), on_gui, None)
dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
