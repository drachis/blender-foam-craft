"""Refuse to destroy a Blender session this suite did not create.

WHY THIS EXISTS (2026-11-03)
----------------------------

A session destroyed a user's unsaved work -- a 128k-vertex sculpt, several
hours old -- and the mechanism deserves writing down because nothing in it
looked wrong at the time.

`launch_blender.py --port 9885` probed the port, found something listening,
printed "port 9885 already serving; reusing it" and exited 0. That message was
read as "my earlier test instance is still up". It was not: it was the user's
own Blender, which had the MCP add-on enabled on the same port. Every script
piped to 9885 afterwards ran inside their session, and most of this suite's
files open with

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

because CLAUDE.md requires each measurement to rebuild scene state. That is
correct for a throwaway instance and catastrophic in a session someone is
working in. Their scene was replaced by a 114-vertex test sphere, and the next
autosave overwrote the only copy of their work.

The port probe is a bare TCP connect, so it cannot tell one Blender from
another. This module supplies the thing it was missing: a way for an instance
to prove it was started by the test suite.

HOW IT PROVES IT
----------------

`launch_blender.py` spawns Blender with FOAMCRAFT_TEST_INSTANCE set in the
child's environment. An environment variable is used rather than a scene
property or a driver-namespace entry because it belongs to the *process*: it
survives `read_homefile`, file loads, add-on reloads and anything a test does
to the scene, and a user's own Blender cannot acquire it by accident.

Fails closed, always. An instance that cannot prove it is disposable is
treated as somebody's work, because the cost of the two mistakes is not
symmetric: refusing to run a test wastes a minute, and the other way round
costs a day's modelling.
"""

import os

ENV_MARKER = "FOAMCRAFT_TEST_INSTANCE"


class NotADisposableSession(RuntimeError):
    """Raised instead of deleting anything, when ownership is not proven."""


def marker():
    """The port this instance was started to serve, or None."""
    return os.environ.get(ENV_MARKER)


def is_disposable():
    return bool(marker())


def describe_session():
    """What is in the current scene, for an error message worth reading."""
    import bpy
    objs = list(bpy.data.objects)
    verts = sum(len(o.data.vertices) for o in objs if o.type == 'MESH')
    return {
        "blend_file": bpy.data.filepath or "(unsaved)",
        "objects": len(objs),
        "object_names": [o.name for o in objs][:10],
        "total_verts": verts,
    }


def assert_disposable(action="clear the scene"):
    """Raise unless this Blender was started by the test suite.

    Call before anything destructive. The message names what would have been
    destroyed, because the one time this fires in earnest, the person reading
    it needs to know immediately whether they nearly lost something.
    """
    if is_disposable():
        return
    import bpy
    info = describe_session()
    raise NotADisposableSession(
        "REFUSING to %s.\n"
        "  This Blender was not started by tools/launch_blender.py: the\n"
        "  %s environment variable is absent, so it is somebody's\n"
        "  working session and not a throwaway one.\n"
        "  File    : %s\n"
        "  Contains: %d object(s), %d vertices -- %s\n"
        "  Start a test instance instead:\n"
        "      python tools/launch_blender.py --port <free port>\n"
        "  and pipe the test to that port. If this really is a scratch\n"
        "  Blender you want to wipe, set %s=manual in its environment\n"
        "  before launching it."
        % (action, ENV_MARKER, info["blend_file"], info["objects"],
           info["total_verts"], ", ".join(info["object_names"]) or "nothing",
           ENV_MARKER))


def clear_scene():
    """Delete every object -- but only in a session we started.

    The guarded replacement for the select-all-and-delete that every test file
    opens with. Same behaviour in a test instance; an exception instead of
    data loss anywhere else.
    """
    assert_disposable("delete every object in the scene")
    import bpy
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
