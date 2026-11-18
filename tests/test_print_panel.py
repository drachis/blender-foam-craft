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
    # The smallest printed label (2026-11-17): the floor foam_layout
    # shrinks a crowded bevel label to before it moves it onto a leader.
    "min_label_mm",
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

    def panel(self, idname, default_closed=False):
        # Blender's collapsible section (2026-11-17 panel pass). Drawn open
        # here, whatever its default: the test is about what the panel can
        # reach, and a closed section is still reachable with one click.
        return self._child(), self._child()

    def prop(self, data, name, **kw):
        self._log["props"].append((name, self._eff()))
        # Which struct it was drawn from (2026-11-17): the piece prefix is
        # the active object's, not the scene settings', so "exists" has to
        # be asked of the owner rather than of the settings group.
        self._log.setdefault("owned", []).append(
            (name, name in data.bl_rna.properties.keys(),
             type(data).__name__))

    def operator(self, idname, **kw):
        self._log["ops"].append((idname, kw.get("text"), self._eff()))

    def label(self, **kw):
        self._log["labels"].append((kw.get("text"), self._eff()))

    def separator(self, **kw):
        pass


PANEL_IDS = (
    # Top to bottom, as the sidebar shows them (bl_order 0..4).
    "IMAGE_PT_foamcraft_drachis_export",     # Foam Pattern (+ export)
    "IMAGE_PT_foamcraft_drachis_advice",     # Craftability
    "IMAGE_PT_foamcraft_drachis",            # Cut Runs
    "IMAGE_PT_foamcraft_drachis_display",    # On Screen Only
    "IMAGE_PT_foamcraft_drachis_pattern",    # Pattern from Strokes
)

# Settings no code reads, so there is nothing for a panel to offer
# (checked 2026-11-17 by grepping the package): drawing them would be a
# control that does nothing.
UNUSED_SETTINGS = {"kerf", "dissolve_angle_limit_deg"}


def _draw(context, panel_id=PANEL_ID):
    cls = getattr(bpy.types, panel_id)
    log = {"props": [], "ops": [], "labels": []}

    class Self:
        layout = FakeLayout(log)
    # The header too: the overlay switch sits in the Foam Pattern header,
    # Blender's usual place for a panel's on/off checkbox.
    if hasattr(cls, "draw_header"):
        cls.draw_header(Self(), context)
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
            # Renamed in the 2026-11-17 panel pass: it holds every setting
            # that changes the printout, not only the export buttons.
            "title": cls is not None and cls.bl_label == "Foam Pattern",
            "first_in_the_tab": getattr(cls, "bl_order", None) == 0,
            "same_tab_as_siblings":
                cls.bl_category == bpy.types.IMAGE_PT_foamcraft_drachis.bl_category,
            # Both bounds: nothing drawn is missing from RNA, and the drawn
            # set is not empty.
            # Asked of whichever struct the panel drew it from (the prefix
            # lives on the Object); every settings-group one still in `rna`.
            "every_drawn_prop_exists": bool(drawn)
            and all(ok for _n, ok, _t in log.get("owned", ()))
            and {n for n, _ok, t in log.get("owned", ())
                 if t != "Object"} <= rna,
            "something_was_drawn": len(drawn) > 0,
            # 2.5 mm: the smallest text the maker can print and read. A
            # physical limit (CLAUDE.md), so its default is pinned here.
            "min_label_default_2_5_mm":
                "min_label_mm" in rna and abs(
                    s.bl_rna.properties["min_label_mm"].default - 2.5) < 1e-6,
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
                 "pattern_surface", "min_label_mm"} <= names,
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
                {"show_seam_numbers", "pattern_surface",
                 "min_label_mm"} <= enabled,
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
                       # Darts moved here in the panel pass: they print.
                       "darts_on_the_print_panel": "show_darts" in union}}


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


def test_piece_prefix_is_drawn_for_the_active_mesh():
    """The active mesh's piece prefix and the label style are on the panel;
    with no active object there is no prefix field (nothing to edit), and
    the style still is. An empty prefix also says what it falls back to."""
    import sandbox_guard
    sandbox_guard.clear_scene()
    without = _all_names(_draw(bpy.context))
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "horn"
    obj.foamcraft_drachis_prefix = ""
    log = _draw(bpy.context)
    with_ = _all_names(log)
    labels = [t for t, _ in log["labels"] if t]
    return {
        "labels": labels,
        "checks": {
            "style_drawn": "piece_label_style" in with_
            and "piece_label_style" in without,
            "prefix_drawn_with_a_mesh": "foamcraft_drachis_prefix" in with_,
            "no_prefix_without_one": "foamcraft_drachis_prefix" not in without,
            "says_the_default": "Uses HO" in labels,
        },
    }


def test_every_setting_has_exactly_one_home():
    """Each scene setting is drawn by one FoamCraft panel, and only one.

    The panel pass (2026-11-17) was prompted by settings that were not
    exposed at all (notch_shape, notch_depth_ratio, origin_object) and by
    others that turned up in two places. Both bounds: nothing missing, and
    nothing drawn twice. Modes that swap props are unioned over.
    """
    s = bpy.context.scene.foamcraft_drachis
    rna = {k for k, p in s.bl_rna.properties.items()
           if k not in ("rna_type", "name") and not p.is_hidden}
    homes = {}
    s, before = _with_settings(tick_mode='PER_RUN', scale_text_with_view=True)
    try:
        for tick_mode, by_view in (('PER_RUN', True), ('BY_DISTANCE', False)):
            s.tick_mode = tick_mode
            s.scale_text_with_view = by_view
            for pid in PANEL_IDS:
                for name, _en in _draw(bpy.context, pid)["props"]:
                    homes.setdefault(name, set()).add(pid)
    finally:
        _restore(s, before)
    missing = sorted(rna - UNUSED_SETTINGS - set(homes))
    twice = {n: sorted(p) for n, p in homes.items() if len(p) > 1}
    return {"missing": missing, "twice": twice,
            "checks": {"none_missing": not missing,
                       "none_drawn_twice": not twice,
                       "something_drawn": len(homes) > 20}}


def test_panels_in_workflow_order():
    """Export first, the stroke bake last (the user's order, 2026-11-17)."""
    orders = [getattr(getattr(bpy.types, pid), "bl_order", None)
              for pid in PANEL_IDS]
    # One tab only (2026-11-17): the old paper overlay registered its own
    # "Fold Angles" tab and read as a second plugin. Every sidebar panel this
    # add-on registers sits in FoamCraft, and there are exactly five.
    ours = [cls for cls in bpy.types.Panel.__subclasses__()
            if "foamcraft_drachis" in cls.__module__
            and getattr(cls, "bl_region_type", "") == 'UI']
    categories = sorted({getattr(cls, "bl_category", "") for cls in ours})
    return {"orders": orders, "categories": categories,
            "checks": {"strictly_increasing": orders == sorted(orders)
                       and len(set(orders)) == len(orders),
                       "one_tab": categories == ["FoamCraft"],
                       "five_panels": len(ours) == len(PANEL_IDS),
                       "no_fold_angles_panel":
                           not hasattr(bpy.types, "IMAGE_PT_fold_angle_overlay")}}


# Hard maxima that stand for something real (2026-11-17). Everything else
# gets a soft_max only -- Blender's convention: the slider has a range, the
# field accepts what you type. Each entry says what breaks past it.
HARD_MAX_ALLOWED = {
    "max_tilt_deg": "bevel depth is thickness * tan(tilt); diverges at 90",
    "max_angle_variation_deg": "an angle between faces cannot pass 180",
    "dissolve_angle_limit_deg": "an angle between faces cannot pass 180",
    "dart_wedge_deg": "an angle cannot pass 180",
    "corner_angle_deg": "a turn along an outline cannot pass 180",
    "min_angle": "a dihedral cannot pass 180",
    "notch_depth_ratio": "a fraction of the foam's thickness",
    "merge_fraction": "a fraction",
    "decimals": "Blender's own display precision stops at 6",
}


def _hard_maxima(rna_props):
    """Names whose hard max is not the type's own ceiling."""
    found = {}
    for name, prop in rna_props.items():
        if prop.type not in {'FLOAT', 'INT'} or getattr(prop, "array_length", 0):
            continue
        ceiling = 2 ** 31 - 1 if prop.type == 'INT' else 3.0e38
        if prop.hard_max < ceiling:
            found[name] = prop.hard_max
    return found


def test_no_arbitrary_hard_limits():
    """A typed value is accepted unless it is meaningless (user, 2026-11-17:
    "keep the blender philosophy ... no arb limits"). Tick Length refusing
    anything over 5 cm prompted it. Both bounds: the allowed list is not
    empty of real hits, and nothing outside it has a hard max."""
    s = bpy.context.scene.foamcraft_drachis
    found = _hard_maxima(s.bl_rna.properties)
    for idname in ("foamcraft_drachis.export_pattern_image",
                   "foamcraft_drachis.bake_pattern"):
        category, _, name = idname.partition(".")
        rna = getattr(getattr(bpy.ops, category), name).get_rna_type()
        for k, v in _hard_maxima(rna.properties).items():
            found["%s.%s" % (name, k)] = v
    unexplained = {k: v for k, v in found.items()
                   if k.split(".")[-1] not in HARD_MAX_ALLOWED}
    tick = s.bl_rna.properties["tick_length"]
    return {"hard_maxima": found, "unexplained": unexplained,
            "checks": {"none_unexplained": not unexplained,
                       "tilt_still_bounded": "max_tilt_deg" in found,
                       "tick_length_unbounded": "tick_length" not in found,
                       # float32 in RNA: 0.05 reads back as 0.0500000007.
                       "tick_length_keeps_its_slider":
                           abs(tick.soft_max - 0.05) < 1e-6}}


TESTS = (
    test_no_arbitrary_hard_limits,
    test_every_setting_has_exactly_one_home,
    test_panels_in_workflow_order,
    test_panel_registered_and_props_exist,
    test_piece_prefix_is_drawn_for_the_active_mesh,
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
