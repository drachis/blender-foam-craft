"""FoamCraft -- fold angles, bevel/dart planning and cut patterns for panel fabrication.

Two related workflows share one geometry core:

* Paper  -- ``uv_fold_angle_overlay`` (live UV overlay) and ``paper_model_export``
  (companion SVG for Blender's bundled Export Paper Model add-on).
* Foam   -- ``foam_panel_tabs``: saw bevel angles, relief darts and cut patterns
  for panels with real material thickness.

Installs as a Blender extension (blender_manifest.toml). ``bl_info`` is kept so
the same folder still works as a legacy add-on on pre-4.2 installs.
"""

bl_info = {
    "name": "FoamCraft",
    "author": "FoamCraft",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "UV Editor > Sidebar (N); File > Export",
    "description": "Fold/bevel angles, foam darts and cut patterns for panel fabrication.",
    "category": "Mesh",
}

from . import foam_angle_core  # noqa: F401  (pure helpers, nothing to register)
from . import uv_fold_angle_overlay
from . import paper_model_export

# Submodules exposing register()/unregister(), in registration order.
_modules = (
    uv_fold_angle_overlay,
    paper_model_export,
)


def register():
    for mod in _modules:
        mod.register()


def unregister():
    for mod in reversed(_modules):
        try:
            mod.unregister()
        except Exception as exc:  # keep unwinding so a partial failure can't wedge the add-on
            print(f"FoamCraft: error unregistering {mod.__name__}: {exc}")
