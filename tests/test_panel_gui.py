"""Capture the FoamCraft sidebar, for reviewing the panel layout by eye.

    python tools/launch_blender.py --port 9886 --gui
    python tools/blender_bridge.py --port 9886 < tests/test_panel_gui.py

Layout is a judgement, not a number (CLAUDE.md), so the deliverable here is
the screenshot. The checks only pin what has to be true for the picture to
mean anything: a windowed Blender, a UV editor with its sidebar open on the
FoamCraft tab, an unwrapped mesh in edit mode, and a craftability result so
the findings list is populated rather than reading "Not checked yet".

Written for the 2026-11-17 panel pass (export first, compact craftability,
stroke bake last). In a background Blender it skips rather than fails.
"""

import sys
import traceback

import bpy

_TOOLS = r"O:\Work\BlenderPlugin\FoamCraft\tools"
sys.path.insert(0, _TOOLS)
import sandbox_guard  # noqa: E402,F401  (this suite only runs in its own Blender)


def _scene():
    sandbox_guard.clear_scene()
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.object
    obj.name = "Suzanne"
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    return obj


def _uv_area_with_sidebar():
    screen = bpy.context.screen
    biggest = max((a for a in screen.areas
                   if a.type not in {'PROPERTIES', 'OUTLINER'}),
                  key=lambda a: a.width * a.height)
    biggest.type = 'IMAGE_EDITOR'
    biggest.ui_type = 'UV'
    space = biggest.spaces.active
    space.show_region_ui = True
    space.show_region_toolbar = False
    ui = next(r for r in biggest.regions if r.type == 'UI')
    window = next(r for r in biggest.regions if r.type == 'WINDOW')
    return biggest, ui, window


def test_sidebar_capture():
    if bpy.app.background:
        return {"skipped": "background Blender has no window to capture",
                "checks": {}}
    obj = _scene()
    area, ui, window = _uv_area_with_sidebar()
    # The sidebar only knows its tabs once it has been drawn: setting the
    # category on a sidebar opened this instant is silently refused (measured:
    # it stays on "Image"). Draw once, then switch.
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    category_set = True
    try:
        ui.active_panel_category = "FoamCraft"
    except (AttributeError, TypeError):
        category_set = False
    with bpy.context.temp_override(area=area, region=window,
                                   edit_object=obj, active_object=obj):
        bpy.ops.foamcraft_drachis.check_craftability()
    # Two redraws: the first lays the panels out, the second paints them.
    for _ in range(2):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    from shot import next_path
    path = next_path("panel_sidebar")
    with bpy.context.temp_override(area=area, region=ui):
        bpy.ops.screen.screenshot_area(filepath=path)
    import os
    return {
        "path": path,
        "checks": {
            "category_is_foamcraft": category_set
            and ui.active_panel_category == "FoamCraft",
            "sidebar_has_width": ui.width > 100,
            "capture_written": os.path.getsize(path) > 1000,
        },
    }


TESTS = (test_sidebar_capture,)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-6:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
