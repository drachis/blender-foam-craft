"""Every overlay setting makes the overlay rebuild when it changes.

    python tools/launch_blender.py --port 9925 --gui --offscreen
    python tools/blender_bridge.py --port 9925 < tests/test_settings_refresh.py

THE BUG (maker, 2026-11-17): toggling Convex/Concave Colour
(`show_dart_colors`) in the panel did not recolour the cut-angle labels.

Measured in a live windowed Blender, across separate bridge calls so the
event loop ran between them: with a counting `depsgraph_update_post`
handler installed, setting `show_dart_colors` through RNA fired it 0 times,
`foam_overlay._dirty` stayed False, and the cached bevel labels kept tone
'convex' while the setting read False. Moving the object, as a control,
fired it once and the labels became 'neutral' on the next tick. So the
setting *was* in `_signature` -- the signature was simply never re-checked,
because the only thing that raised the dirty flag was a depsgraph update,
and a scene's custom property group is not depsgraph data.

The fix is Blender's idiom for it: an `update=` callback on every property
of the settings group (and on the per-object piece prefix) that calls
`foam_overlay.mark_dirty()`. And the signature is now walked from RNA
rather than listed by hand, so a setting cannot be left out of it. These
cases pin both, for every property, so the next setting added cannot
silently miss either.

What one bridge call cannot do is let the event loop run: a timer never
fires part-way through a script. So the GUI case does what the timer does,
`_timer_tick()`, after the RNA assignment -- which is the step the bug
skipped. The update callback runs synchronously inside the assignment, so
whether it raised the flag is observable here directly.
"""

import sys
import traceback

import bpy
import bmesh

_ADDON = "bl_ext.user_default.foamcraft_drachis"
# Re-enabled rather than purged and imported: the GUI case needs the module
# instance whose timer and draw handler are live (see test_overlay_gui.py).
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
try:
    bpy.ops.preferences.addon_disable(module=_ADDON)
except Exception:                                        # noqa: BLE001
    pass
for _name in [n for n in list(sys.modules) if n.startswith(_ADDON)]:
    del sys.modules[_name]
bpy.ops.preferences.addon_enable(module=_ADDON)

import bl_ext.user_default.foamcraft_drachis.foam_overlay as ov  # noqa: E402

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")
import sandbox_guard  # noqa: E402


def _cube_in_edit_mode():
    sandbox_guard.clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    return obj


def _other_value(prop, value):
    """A different legal value for `prop`, or None when there is none."""
    kind = prop.type
    if kind == 'BOOLEAN' and not getattr(prop, "array_length", 0):
        return not value
    if kind == 'INT' and not getattr(prop, "array_length", 0):
        return value + 1 if value + 1 <= prop.hard_max else value - 1
    if kind == 'FLOAT':
        lo, hi = prop.hard_min, prop.hard_max
        if getattr(prop, "array_length", 0):
            first = value[0]
            new = first - 0.25 if first - 0.25 >= lo else first + 0.25
            return (new,) + tuple(value[1:])
        lo = max(lo, -1e6)
        hi = min(hi, 1e6)
        new = value * 1.5 if value * 1.5 <= hi and abs(value) > 1e-4 \
            else value + (hi - lo) * 0.1
        if new > hi:
            new = (value + lo) / 2.0
        return new
    if kind == 'ENUM':
        items = [i.identifier for i in prop.enum_items]
        others = [i for i in items if i != value]
        return others[0] if others else None
    if kind == 'STRING':
        return value + "x"
    return None


def test_every_display_setting_is_in_the_signature_and_marks_dirty():
    """For each property of the settings group: changing it changes
    `_signature`, and raises the dirty flag through its update callback.

    Both bounds: the exempt list names only properties that exist, and it
    is short -- every other property is checked, and the count checked is
    the count the group has, so a property skipped for having no "other
    value" is reported rather than silently passed.
    """
    _cube_in_edit_mode()
    s = bpy.context.scene.foamcraft_drachis
    pairs = ov._edit_meshes(bpy.context)
    props = [p for p in s.bl_rna.properties
             if p.identifier not in ("rna_type", "name")]
    names = {p.identifier for p in props}

    missing_sig, missing_dirty, skipped = [], [], []
    checked = 0
    for prop in props:
        name = prop.identifier
        if prop.type == 'POINTER':
            skipped.append(name)
            continue
        before = getattr(s, name)
        before = tuple(before) if getattr(prop, "array_length", 0) else before
        new = _other_value(prop, before)
        if new is None:
            skipped.append(name)
            continue
        base = ov._signature(pairs, s)
        ov._dirty = False
        try:
            setattr(s, name, new)
            if name not in ov._SIGNATURE_EXEMPT and \
                    ov._signature(pairs, s) == base:
                missing_sig.append(name)
            if not ov._dirty:
                missing_dirty.append(name)
            checked += 1
        finally:
            setattr(s, name, before)
    bpy.ops.object.mode_set(mode='OBJECT')
    pointer_names = {p.identifier for p in props if p.type == 'POINTER'}
    return {
        "checked": checked, "skipped": skipped,
        "missing_from_signature": missing_sig,
        "not_marking_dirty": missing_dirty,
        "checks": {
            "every_setting_in_signature": missing_sig == [],
            "every_setting_marks_dirty": missing_dirty == [],
            # The ones the bug was about, by name, so a renamed walk cannot
            # pass by checking nothing.
            "colour_toggle_checked": "show_dart_colors" in names
            and "show_dart_colors" not in skipped,
            "colours_checked": {"convex_color", "concave_color"} <= names,
            # Only pointers are skipped; everything else got a new value.
            "only_pointers_skipped": set(skipped) == pointer_names,
            "exempt_names_exist": ov._SIGNATURE_EXEMPT <= names,
            "most_settings_checked": checked >= len(props) - len(skipped)
            and checked > 40,
        },
    }


def test_the_piece_prefix_rebuilds_the_overlay():
    """The per-object abbreviation is printed on every piece, so it must
    change the signature and raise the flag like a scene setting."""
    obj = _cube_in_edit_mode()
    s = bpy.context.scene.foamcraft_drachis
    pairs = ov._edit_meshes(bpy.context)
    base = ov._signature(pairs, s)
    ov._dirty = False
    obj.foamcraft_drachis_prefix = "ZZ"
    dirty = ov._dirty
    changed = ov._signature(pairs, s) != base
    obj.foamcraft_drachis_prefix = ""
    bpy.ops.object.mode_set(mode='OBJECT')
    return {"checks": {"marks_dirty": dirty, "changes_signature": changed}}


def test_toggling_convex_concave_colour_recolours_the_labels():
    """The maker's bug, in a window: on -> every cached bevel label on the
    cube is 'convex' (all 24 runs are +90 folds); off -> every one is
    'neutral'; on again -> 'convex' again. Each change is made through RNA,
    then the timer's own tick is run, as the event loop would."""
    if bpy.app.background:
        return {"skipped": "needs a window; run with --gui",
                "checks": {"ran_with_a_window": False}}
    _cube_in_edit_mode()
    s = bpy.context.scene.foamcraft_drachis
    saved = (s.show_overlay, s.show_dart_colors, s.show_brackets)
    seen = []
    try:
        s.show_overlay = True
        s.show_brackets = True
        for value in (True, False, True):
            ov._dirty = False
            s.show_dart_colors = value
            raised = ov._dirty
            ov._timer_tick()
            seen.append({"value": value, "dirty_raised": raised,
                         "tones": sorted({m.tone for m in ov._cache.bevel_marks}),
                         "labels": len(ov._cache.bevel_marks)})
    finally:
        s.show_overlay, s.show_dart_colors, s.show_brackets = saved
        bpy.ops.object.mode_set(mode='OBJECT')
    return {
        "seen": seen,
        "checks": {
            "flag_raised_each_time": all(x["dirty_raised"] for x in seen),
            "on_is_convex": seen[0]["tones"] == ["convex"],
            "off_is_neutral": seen[1]["tones"] == ["neutral"],
            "on_again_is_convex": seen[2]["tones"] == ["convex"],
            "24_labels_each_time": all(x["labels"] == 24 for x in seen),
        },
    }


TESTS = (
    test_every_display_setting_is_in_the_signature_and_marks_dirty,
    test_the_piece_prefix_rebuilds_the_overlay,
    test_toggling_convex_concave_colour_recolours_the_labels,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-8:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
