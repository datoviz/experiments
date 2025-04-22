from datoviz import vec2, vec3, Out
import datoviz as dvz
import numpy as np
from iblatlas import atlas, regions
a = atlas.AllenAtlas()
br = regions.BrainRegions()


def propagate_visibility(hidden: np.ndarray, levels: np.ndarray) -> None:
    """
    In‑place visibility propagation.

    Parameters
    ----------
    hidden : np.ndarray, bool
        `True`  → item is hidden
        `False` → item is shown / matched

        On return every ancestor of a shown entry is also set to
        `False`.  The array is *modified in place*.

    levels : np.ndarray, int
        Indent level of each row (root = 0).  Must have the same
        length as `hidden`.

    Notes
    -----
    *Assumes `levels` is the classical “pre‑order, depth‑first”
    ordering produced by e.g. an ImGui tree: a parent always appears
    **before** any of its children.*
    """

    # -----------------------------------------------------------------
    # `stack[level]` will hold the index of the **latest** row
    # encountered at that level (i.e. the current parent for deeper
    # rows).  It grows / shrinks as we walk through the flat list.
    # -----------------------------------------------------------------
    stack: list[int] = []

    for idx, lvl in enumerate(levels):
        # Keep stack length == current depth
        # (pop when we climb back up in the tree).
        while len(stack) > lvl:
            stack.pop()

        # If this row is visible, mark every ancestor visible too.
        if not hidden[idx]:
            for anc in stack:          # ancestors are indices in stack
                if hidden[anc]:
                    hidden[anc] = False

        # Push *this* row so it becomes the parent for the next level
        stack.append(idx)


n = br.n_lr
count = n - 1  # we remove void, the first item
levels = br.level[1:n].astype(np.uint32)
names = br.name[1:n].tolist()
acronyms = br.acronym[1:n].tolist()
colors = br.rgba[1:n].astype(np.uint8)
folded = np.zeros(n - 1, dtype=np.bool)
selected = np.zeros(n - 1, dtype=np.bool)
hidden = np.zeros(n - 1, dtype=np.bool)
haystack = np.array([f"{a} {n}".lower() for a, n in zip(acronyms, names)])
folded[levels >= 5] = True


app = dvz.app(0)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)
figure = dvz.figure(scene, 800, 600, dvz.CANVAS_FLAGS_IMGUI)

search = dvz.CStringBuffer("")


@dvz.gui
def ongui(app, fid, ev):
    dvz.gui_pos(vec2(25, 25), vec2(0, 0))
    dvz.gui_size(vec2(500, 500))
    dvz.gui_begin("My GUI", 0)

    # hidden[:] = 0
    if dvz.gui_textbox("Search", 64, search, 0):
        s = search.value.lower()
        if s:
            hidden[:] = np.char.find(haystack, s) == -1
        else:
            hidden[:] = 0

    propagate_visibility(hidden, levels)

    dvz.gui_tree(count, acronyms, names, levels, colors, folded, selected, hidden)
    dvz.gui_end()


panel1 = dvz.panel(figure, 50, 50, 300, 300)
dvz.demo_panel3D(panel1)
dvz.panel_gui(panel1, "Panel 1", 0)

panel2 = dvz.panel(figure, 400, 100, 300, 300)
dvz.demo_panel2D(panel2)
dvz.panel_gui(panel2, "Panel 2", 0)

dvz.app_gui(app, dvz.figure_id(figure), ongui, None)
dvz.scene_run(scene, app, 0)
dvz.scene_destroy(scene)
dvz.app_destroy(app)
