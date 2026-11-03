"""The guard that stops this suite wiping somebody's Blender session.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_sandbox_guard.py

On 2026-11-03 this suite destroyed a user's unsaved sculpt. `launch_blender.py`
probed port 9885, found the user's own Blender listening there (it had the MCP
add-on enabled on the same port), printed "already serving; reusing it", and
every test piped to that port afterwards ran inside their session -- opening,
as nearly every file here does, by deleting every object in the scene.

tools/sandbox_guard.py is the fix. This file is the proof that it works, and
most of it is deliberately about making the guard *fail*: CLAUDE.md is explicit
that a check which has never failed should be distrusted, and a safety net
nobody has fallen into is a guess. Both directions are asserted here --
refusing in an unmarked session, and still permitting in a marked one, because
a guard that blocks everything would pass a one-sided test while making the
suite unrunnable.
"""

import os
import sys
import importlib
import traceback

import bpy

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import sandbox_guard                                       # noqa: E402
importlib.reload(sandbox_guard)


def _with_marker(value):
    """Set or clear the marker, returning what it was."""
    before = os.environ.get(sandbox_guard.ENV_MARKER)
    if value is None:
        os.environ.pop(sandbox_guard.ENV_MARKER, None)
    else:
        os.environ[sandbox_guard.ENV_MARKER] = value
    return before


def _restore(before):
    if before is None:
        os.environ.pop(sandbox_guard.ENV_MARKER, None)
    else:
        os.environ[sandbox_guard.ENV_MARKER] = before


def test_this_instance_is_marked_as_disposable():
    """The suite's own Blender must carry the marker.

    If this fails, the launcher is not stamping the environment and every
    other check below is testing a mechanism that is not actually armed in
    the configuration the tests run in.
    """
    return {
        "marker": sandbox_guard.marker(),
        "is_disposable": sandbox_guard.is_disposable(),
        "checks": {
            "marker_is_present": sandbox_guard.is_disposable(),
            # It carries the port, so a stray marker from somewhere else is
            # visible rather than merely truthy.
            "marker_looks_like_a_port":
                (sandbox_guard.marker() or "").isdigit(),
        },
    }


def test_it_refuses_to_clear_an_unmarked_session():
    """Make the guard fail on purpose, and check nothing was deleted.

    This is the case that matters: an unmarked session is somebody's work.
    The scene is populated first, the marker removed, and clear_scene() called
    -- the exact call the real tests make. It must raise, and crucially the
    objects must still be there afterwards, because an exception raised
    *after* the delete would be no protection at all.
    """
    before = _with_marker(None)
    try:
        # Build a scene worth protecting. Named so the failure message is
        # recognisable if this ever fires for real.
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_monkey_add()
        bpy.context.object.name = "precious_sculpt"
        bpy.ops.mesh.primitive_cube_add(location=(3, 0, 0))

        before_names = sorted(o.name for o in bpy.data.objects)

        raised = None
        try:
            sandbox_guard.clear_scene()
        except sandbox_guard.NotADisposableSession as exc:
            raised = str(exc)

        after_names = sorted(o.name for o in bpy.data.objects)
        return {
            "objects_before": before_names,
            "objects_after": after_names,
            "raised": raised is not None,
            "message_mentions_the_scene":
                bool(raised) and "precious_sculpt" in raised,
            "message_head": (raised or "").splitlines()[0] if raised else None,
            "checks": {
                "it_refused": raised is not None,
                # The whole point: the refusal happens before the delete.
                "nothing_was_deleted": before_names == after_names,
                "scene_was_not_empty_to_begin_with": len(before_names) == 2,
                # The message has to name what it protected, or whoever sees
                # it cannot tell whether they nearly lost something.
                "message_names_the_objects":
                    bool(raised) and "precious_sculpt" in raised,
            },
        }
    finally:
        _restore(before)


def test_it_still_clears_a_marked_session():
    """The other bound. A guard that refuses everything is not a guard.

    Without this, a clear_scene() that always raised would pass the case above
    and quietly break every other test file in the suite.
    """
    before = _with_marker("9885")
    try:
        bpy.ops.mesh.primitive_monkey_add()
        bpy.ops.mesh.primitive_cube_add(location=(3, 0, 0))
        populated = len(bpy.data.objects)

        sandbox_guard.clear_scene()
        remaining = len(bpy.data.objects)
        return {
            "objects_before": populated,
            "objects_after": remaining,
            "checks": {
                "the_scene_was_populated": populated >= 2,
                "it_cleared": remaining == 0,
            },
        }
    finally:
        _restore(before)


def test_an_unsaved_user_scene_is_described_usefully():
    """What the refusal message can say about what it is protecting.

    The real session that was lost was unsaved, so bpy.data.filepath was empty
    and the filename could not identify it. The object names and vertex count
    are what remain, and they were in fact enough to recognise the loss
    afterwards -- a 114-vertex sphere where a 128k-vertex sculpt had been.
    """
    before = _with_marker("9885")
    try:
        sandbox_guard.clear_scene()
        bpy.ops.mesh.primitive_monkey_add()
        bpy.context.object.name = "beast_stand_in"
        info = sandbox_guard.describe_session()
        return {
            "described": info,
            "checks": {
                "names_the_objects": "beast_stand_in" in info["object_names"],
                "counts_the_vertices": info["total_verts"] > 0,
                "says_unsaved_rather_than_blank":
                    info["blend_file"] == "(unsaved)",
            },
        }
    finally:
        _restore(before)


TESTS = (
    test_this_instance_is_marked_as_disposable,
    test_it_refuses_to_clear_an_unmarked_session,
    test_it_still_clears_a_marked_session,
    test_an_unsaved_user_scene_is_described_usefully,
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
