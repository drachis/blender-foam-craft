"""The Print / Export panel draws every export setting, and only the right ones.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_print_panel.py

Before this panel, show_align_ticks, tick_mode, ticks_per_run, tick_spacing,
tick_length, tick_numbers, island_fill and show_scale_ruler existed on the
settings group and were drawn by nothing, so a user could not reach them.

Panel draw code is exercised with a fake layout that records .prop() calls,
which needs no window and reads no mesh (a panel must not). Expectations come
from the panel's own contract: PER_RUN shows ticks_per_run and not
tick_spacing, BY_DISTANCE the reverse, and with ticks off the detail
properties are either absent or drawn disabled.
"""

import sys
import traceback

import bpy

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")
import sandbox_guard  # noqa: E402,F401  (this suite only runs in its own Blender)

PANEL_ID = "IMAGE_PT_foamcraft_drachis_export"

EXPORT_PROPS = (
    "show_align_ticks", "tick_mode", "ticks_per_run", "tick_spacing",
    "tick_length", "tick_numbers", "island_fill", "show_scale_ruler",
    # The bevel labels (2026-11-17): what each one says, and which face of
    # the foam the pattern is traced on, which decides the arrows.
    "show_seam_numbers", "pattern_surface",
)
TICK_DETAIL = ("tick_mode", "ticks_per_run", "tick_spacing",
               "tick_length", "tick_numbers")


class FakeLayout:
    """Records prop() calls with the effective enabled state; nests freely."""

    def __init__(self, log, parent_enabled=True):
        self._log = log
        self._enabled = True
        self._parent_enabled = parent_enabled
        self.scale_y = 1.0
        self.alert = False

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)

    def _eff(self):
        return self._parent_enabled and self._enabled

    def _child(self, *a, **k):
        return FakeLayout(self._log, self._eff())

    box = row = column = split = column_flow = grid_flow = _child

    def prop(self, data, name, **kw):
        self._log["props"].append((name, self._eff()))

    def operator(self, idname, **kw):
        self._log["ops"].append((idname, kw.get("text"), self._eff()))

    def label(self, **kw):
        self._log["labels"].append((kw.get("text"), self._eff()))

    def separator(self, **kw):
        pass


def _draw(context):
    cls = getattr(bpy.types, PANEL_ID)
    log = {"props": [], "ops": [], "labels": []}

    class Self:
        layout = FakeLayout(log)
    cls.draw(Self(), context)
    return log


def _enabled_names(log):
    return {n for n, en in log["props"] if en}


def _all_names(log):
    return {n for n, _ in log["props"]}


def _with_settings(**values):
    s = bpy.context.scene.foamcraft_drachis
    before = {k: getattr(s, k) for k in values}
    for k, v in values.items():
        setattr(s, k, v)
    return s, before


def _restore(s, before):
    for k, v in before.items():
        setattr(s, k, v)


def test_panel_registered_and_props_exist():
    s = bpy.context.scene.foamcraft_drachis
    log = _draw(bpy.context)
    rna = set(s.bl_rna.properties.keys())
    drawn = _all_names(log)
    cls = getattr(bpy.types, PANEL_ID, None)
    return {
        "drawn": sorted(drawn),
        "checks": {
            "panel_registered": hasattr(bpy.types, PANEL_ID),
            "title": cls is not None and cls.bl_label == "Print / Export",
            "same_tab_as_siblings":
                cls.bl_category == bpy.types.IMAGE_PT_foamcraft_drachis.bl_category,
            # Both bounds: nothing drawn is missing from RNA, and the drawn
            # set is not empty.
            "every_drawn_prop_exists": bool(drawn) and drawn <= rna,
            "something_was_drawn": len(drawn) > 0,
        },
    }


def test_per_run_mode():
    s, before = _with_settings(show_align_ticks=True, tick_mode='PER_RUN')
    try:
        names = _enabled_names(_draw(bpy.context))
    finally:
        _restore(s, before)
    return {
        "enabled": sorted(names),
        "checks": {
            "has_ticks_per_run": "ticks_per_run" in names,
            "no_tick_spacing": "tick_spacing" not in names,
            "all_other_export_props":
                {"show_align_ticks", "tick_mode", "tick_length", "tick_numbers",
                 "island_fill", "show_scale_ruler", "show_seam_numbers",
                 "pattern_surface"} <= names,
        },
    }


def test_by_distance_mode():
    s, before = _with_settings(show_align_ticks=True, tick_mode='BY_DISTANCE')
    try:
        names = _enabled_names(_draw(bpy.context))
    finally:
        _restore(s, before)
    return {
        "enabled": sorted(names),
        "checks": {
            "has_tick_spacing": "tick_spacing" in names,
            "no_ticks_per_run": "ticks_per_run" not in names,
        },
    }


def test_ticks_off_disables_detail():
    s, before = _with_settings(show_align_ticks=False, tick_mode='PER_RUN')
    try:
        log = _draw(bpy.context)
    finally:
        _restore(s, before)
    enabled = _enabled_names(log)
    return {
        "enabled": sorted(enabled),
        "checks": {
            "toggle_itself_stays_enabled": "show_align_ticks" in enabled,
            "no_tick_detail_enabled": not (enabled & set(TICK_DETAIL)),
            "sheet_props_still_enabled":
                {"island_fill", "show_scale_ruler"} <= enabled,
            # Bevel labels do not depend on the ticks.
            "bevel_props_still_enabled":
                {"show_seam_numbers", "pattern_surface"} <= enabled,
        },
    }


def test_all_export_props_are_reachable():
    """Union over both modes covers every orphan the panel was written for."""
    union = set()
    s, before = _with_settings(show_align_ticks=True)
    try:
        for mode in ('PER_RUN', 'BY_DISTANCE'):
            s.tick_mode = mode
            union |= _all_names(_draw(bpy.context))
    finally:
        _restore(s, before)
    missing = [p for p in EXPORT_PROPS if p not in union]
    return {"missing": missing,
            "checks": {"none_missing": not missing,
                       "show_darts_not_duplicated": "show_darts" not in union}}


def test_export_buttons_and_missing_operator_guard():
    log = _draw(bpy.context)
    ops = {o: (t, en) for o, t, en in log["ops"]}
    labels = [t for t, _ in log["labels"] if t]
    svg = "foamcraft_drachis.export_panel_pattern"
    img = "foamcraft_drachis.export_pattern_image"
    import bl_ext.user_default.foamcraft_drachis.foam_overlay as fo
    img_there = fo._operator_registered(img)
    return {
        "image_registered": img_there,
        "ops": ops, "labels": labels,
        "checks": {
            "guard_true_for_a_real_operator": fo._operator_registered(svg),
            "guard_false_for_a_bogus_one":
                not fo._operator_registered("foamcraft_drachis.no_such_op_xyz"),
            "guard_false_for_bogus_category":
                not fo._operator_registered("nosuchcat.nothing"),
            "svg_button_drawn": ops.get(svg) == ("Export SVG", True),
            # Either the button, or a disabled label -- never both, never neither.
            "image_button_xor_placeholder":
                (img in ops) != any("Export Image" in t for t in labels),
            "image_button_matches_registration": (img in ops) == img_there,
        },
    }


TESTS = (
    test_panel_registered_and_props_exist,
    test_per_run_mode,
    test_by_distance_mode,
    test_ticks_off_disables_detail,
    test_all_export_props_are_reachable,
    test_export_buttons_and_missing_operator_guard,
)

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
