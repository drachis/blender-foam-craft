"""Logging for FoamCraft.

Every number this add-on computes -- run counts, flagged edges, island sizes,
export totals -- ends up here, so a result can be read out of a file instead of
counted off a screenshot. Counting marks in a picture is slow, it cannot be
diffed between runs, and it silently gets the answer wrong when two marks
overlap.

The log lives at::

    <Blender user data>/foam_craft/foam_craft.log

which is stable across sessions, unlike ``bpy.app.tempdir``. The exact path is
shown in the FoamCraft sidebar so it can be found without guessing.

Levels, and what belongs at each:

    INFO   what the user asked for and what came back -- an export wrote N
           pieces, a craftability check found N blockers. One line per action.
    DEBUG  the overlay's per-rebuild detail. The rebuild runs on a timer, so
           this is noisy by design and off unless switched on.
    ERROR  an operator failed, with the traceback.
"""

import logging
import logging.handlers
import os

import bpy

_LOGGER_NAME = "foam_craft"
_FILENAME = "foam_craft.log"

# Rotate rather than truncate: a crash is usually explained by what happened
# just before it, and truncating on load throws exactly that away.
_MAX_BYTES = 1 << 20
_BACKUPS = 3

_configured = False


def log_dir():
    """Directory holding the log file, created if missing."""
    return bpy.utils.user_resource('DATAFILES', path="foam_craft", create=True)


def log_path():
    return os.path.join(log_dir(), _FILENAME)


def get_logger():
    """The add-on's logger, configured on first use.

    Handlers are attached once and reused. Re-running register() during
    development would otherwise stack a second copy of every handler onto the
    same logger and duplicate every line.
    """
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    # Blender's own logging is not ours to configure, and propagating would
    # print our lines twice once it has a root handler.
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    try:
        handler = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
            encoding="utf-8")
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    except OSError:
        # A read-only or missing data directory must not stop the add-on
        # loading. The console handler below still gets the lines.
        pass

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    _configured = True
    return logger


def set_level(level):
    """Raise or lower what reaches the handlers.

    Set from the sidebar. DEBUG turns on the per-rebuild overlay detail, which
    is far too noisy to leave on while working.
    """
    logging.getLogger(_LOGGER_NAME).setLevel(
        getattr(logging, level, logging.INFO))


def summary(event, **fields):
    """Log one line of key=value pairs.

    Written this way so a run can be grepped and diffed against another run
    without parsing prose::

        rebuild runs=770 flagged_short=6 min_len=0.017 max_len=1.438

    Floats are rounded on the way in; six decimals is well past the precision
    any of these measurements actually carry, and full repr makes the columns
    impossible to scan.
    """
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append("%s=%.6g" % (key, value))
        else:
            parts.append("%s=%s" % (key, value))
    return "%s %s" % (event, " ".join(parts))
