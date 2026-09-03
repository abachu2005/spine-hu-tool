"""Application-wide file logging.

Every GUI/CLI session writes to a rotating log file in the user data dir, so
"it failed on the PI's machine" is diagnosable from a file they can send us
(and that failed runs auto-archive to the cloud) instead of a screenshot of a
dialog. Error dialogs print the log path for exactly that reason.
"""
from __future__ import annotations
import logging
import logging.handlers
import os
import platform
import sys
import traceback

_LOGGER_NAME = "spine_hu"
_initialized = False


def log_dir() -> str:
    from ..segmentation.local_setup import _user_data_root
    return os.path.join(_user_data_root(), "logs")


def log_path() -> str:
    return os.path.join(log_dir(), "spine_hu.log")


def get_logger(name: str = "") -> logging.Logger:
    base = logging.getLogger(_LOGGER_NAME)
    return base.getChild(name) if name else base


def init_logging(verbose_console: bool = False) -> logging.Logger:
    """Set up the rotating file log (idempotent) and log the startup banner."""
    global _initialized
    log = logging.getLogger(_LOGGER_NAME)
    if _initialized:
        return log
    log.setLevel(logging.DEBUG)

    try:
        os.makedirs(log_dir(), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)
    except OSError:
        # An unwritable data dir must never stop the app from starting.
        pass
    if verbose_console:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        log.addHandler(sh)

    _initialized = True
    _log_startup(log)
    _install_excepthook(log)
    return log


def _log_startup(log: logging.Logger) -> None:
    """One diagnostic block per session: version, platform, resolved runtimes."""
    from .. import __version__
    log.info("---- session start: Spine HU Tool v%s ----", __version__)
    log.info("platform: %s | python %s | frozen=%s | exe=%s",
             platform.platform(), platform.python_version(),
             bool(getattr(sys, "frozen", False)), sys.executable)
    try:
        from ..segmentation.backends import resolve_seg_url
        log.info("segmentation URL: %s", resolve_seg_url(None))
    except Exception as e:
        log.warning("could not resolve segmentation URL: %s", e)
    try:
        from ..segmentation import local_setup as ls
        cands = ls.runtime_candidates()
        if cands:
            for source, py in cands:
                log.info("local-seg runtime candidate [%s]: %s", source, py)
        else:
            log.info("no local-seg runtime installed (cloud-only)")
        overrides = ls._active_overrides()
        if overrides:
            log.warning("active SPINE_HU overrides: %s", ", ".join(overrides))
    except Exception as e:
        log.warning("could not probe local-seg runtimes: %s", e)


def _install_excepthook(log: logging.Logger) -> None:
    """Log any uncaught exception (with traceback) before the default handling."""
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        try:
            log.critical("UNCAUGHT EXCEPTION:\n%s",
                         "".join(traceback.format_exception(exc_type, exc, tb)))
        finally:
            prev(exc_type, exc, tb)

    sys.excepthook = hook


def log_exception(context: str, exc: BaseException) -> None:
    """Log a handled exception with its full traceback under `context`."""
    get_logger().error("%s: %s\n%s", context, exc,
                       "".join(traceback.format_exception(exc)))
