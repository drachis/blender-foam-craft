"""Checks that need a window: does the overlay actually draw?

    python tools/run_tests.py --gui --only overlay

Everything else in this suite reads numbers out of Blender. None of it can tell
you whether a single pixel reached the screen -- and this project has already
shipped an overlay that could never have drawn at all, because it asked for a
shader name removed in 4.0 and the failure only surfaced when someone looked.

So these compare pictures. Capture the UV editor with the overlay off, capture
it again with the overlay on, and count the pixels that changed. A draw handler
that throws, draws nothing, or draws off-screen all leave the two frames
identical, and none of those can be caught any other way.

Headless runs skip, reporting why rather than passing silently: a check that
cannot fail is worse than no check.
"""

import math
import sys
import traceback

import bpy
import bmesh

# The add-on is re-enabled first, and this file must not reload the module
# afterwards.
#
# The draw handler and the cache belong to whichever module instance was live
# when the add-on registered. The other test files purge the whole package from
# sys.modules to pick up edits, which orphans that registration: the handler
# still runs from a module object nothing can reach any more, so this test would
# inspect a fresh empty copy and report "nothing to draw" about an overlay that
# is working. Measured that way once -- draw handle unset, timer unregistered,
# cache empty -- with the add-on apparently loaded.
_ADDON = "bl_ext.user_default.foamcraft_drachis"
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
try:
    bpy.ops.preferences.addon_disable(module=_ADDON)
except Exception:                                        # noqa: BLE001
    pass
for _name in [n for n in list(sys.modules) if n.startswith(_ADDON)]:
    del sys.modules[_name]
bpy.ops.preferences.addon_enable(module=_ADDON)

import bl_ext.user_default.foamcraft_drachis.foam_overlay as overlay   # noqa: E402

_TOOLS = "O:/Work/BlenderPlugin/FoamCraft/tools"
# Every Nth pixel. Comparing several million floats in Python is slow and adds
# nothing -- a mark that changes only one pixel in a hundred is invisible anyway.
_SAMPLE = 37


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.objects, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _uv_area():
    """An area switched to the UV editor, with its window region."""
    screen = bpy.context.screen
    if screen is None:
        return None, None
    biggest = None
    for area in screen.areas:
        if area.type in {'PROPERTIES', 'OUTLINER'}:
            continue
        if biggest is None or area.width * area.height > \
                biggest.width * biggest.height:
            biggest = area
    if biggest is None:
        return None, None
    biggest.type = 'IMAGE_EDITOR'
    biggest.ui_type = 'UV'
    # Hide the sidebar. Toggling the overlay also changes what the panel shows,
    # so with it visible the two captures differ by repainted UI whether or not
    # a single mark was drawn -- 85 changed pixels, measured, from chrome alone.
    space = biggest.spaces.active
    if hasattr(space, "show_region_ui"):
        space.show_region_ui = False
    if hasattr(space, "show_region_toolbar"):
        space.show_region_toolbar = False
    region = next((r for r in biggest.regions if r.type == 'WINDOW'), None)
    return biggest, region


def _capture(area, region, label):
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    from shot import next_path
    path = next_path("gui_%s" % label)
    area.tag_redraw()
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path)
    return path


def _changed_pixels(path_a, path_b):
    """How many sampled pixels differ between two captures."""
    img_a = bpy.data.images.load(path_a)
    img_b = bpy.data.images.load(path_b)
    try:
        a = img_a.pixels[:]
        b = img_b.pixels[:]
        if len(a) != len(b):
            return None
        changed = 0
        total = 0
        for i in range(0, len(a), 4 * _SAMPLE):
            total += 1
            if abs(a[i] - b[i]) > 0.02 or abs(a[i + 1] - b[i + 1]) > 0.02 \
                    or abs(a[i + 2] - b[i + 2]) > 0.02:
                changed += 1
        return changed, total
    finally:
        bpy.data.images.remove(img_a)
        bpy.data.images.remove(img_b)


def test_overlay_draws_into_the_uv_editor():
    """The overlay must change what is on screen. Off, then on, then compare.

    The failure this exists for: an overlay that raises inside its draw handler
    is silent -- Blender prints to the console and carries on rendering the
    frame without it -- so every data check passes while nothing is drawn.
    """
    if bpy.app.background:
        return {"skipped": "needs a window; run with --gui",
                "checks": {"ran_with_a_window": False}}

    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.04)

    area, region = _uv_area()
    if area is None or region is None:
        return {"skipped": "no usable area", "checks": {"found_uv_area": False}}

    settings = bpy.context.scene.foamcraft_drachis
    settings.show_overlay = False
    off = _capture(area, region, "overlay_off")

    settings.show_overlay = True
    # The overlay rebuilds on a timer and refuses to draw while dirty, so a
    # capture taken immediately can legitimately show nothing. The operator
    # forces the rebuild that the timer would otherwise do.
    # Force the rebuild rather than waiting for it. The overlay rebuilds on a
    # timer and refuses to draw while dirty, and a timer never fires part-way
    # through a script -- so a capture taken here would legitimately be blank
    # and the test would blame the draw handler for it.
    try:
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.foamcraft_drachis.refresh_overlay()
        if hasattr(overlay, "_timer_tick"):
            overlay._timer_tick()
    except RuntimeError:
        pass
    on = _capture(area, region, "overlay_on")

    result = _changed_pixels(off, on)
    handler_live = overlay._draw_handle is not None
    runs = len(overlay._cache.runs)
    bpy.ops.object.mode_set(mode='OBJECT')
    if result is None:
        return {"error": "captures differ in size",
                "checks": {"comparable": False}}
    changed, total = result

    return {
        "off": off,
        "on": on,
        "sampled_pixels": total,
        "changed_pixels": changed,
        "cached_runs": runs,
        "draw_handler_registered": handler_live,
        "checks": {
            # If this is false the rest is meaningless: nothing was ever going
            # to draw, and a blank frame says nothing about the overlay.
            "draw_handler_registered": handler_live,
            # Something reached the screen.
            "overlay_changed_the_frame": changed > 0,
            # ...and it is marks, not the whole view. A changed background or a
            # different zoom would repaint nearly everything, which would mean
            # the two captures are not comparable rather than that it drew.
            "changed_a_minority_of_the_frame": 0 < changed < total * 0.5,
            # The data behind it exists, so a blank frame would be a draw fault
            # rather than an empty mesh.
            "there_was_something_to_draw": runs > 0,
        },
    }


TESTS = (test_overlay_draws_into_the_uv_editor,)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-5:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
