"""Notebook/session helpers for ZShooter ScopeSim workflows."""

from __future__ import annotations

import importlib.resources
import inspect
import io
import pathlib
import warnings
from contextlib import contextmanager
from typing import Any

import scopesim as sim
import scopesim.optics.optical_train as optical_train

####### Path abstractions ########

def configure_irdb_path(irdb_path: str | pathlib.Path | None = None) -> str:
    """Resolve the local IRDB checkout or installed editable package root.
     Then Configure ScopeSim to find the local IRDB package once."""
    try:
        resolved = pathlib.Path(importlib.resources.files("irdb")).parent.resolve()
    except ModuleNotFoundError:
        if irdb_path is None:
            irdb_path = pathlib.Path.home() / "src" / "irdb"
        resolved = pathlib.Path(irdb_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
    resolved = str(resolved)

    sim.rc.__config__["!SIM.file.local_packages_path"] = resolved
    search_path = sim.rc.__config__["!SIM.file.search_path"]
    if resolved not in search_path:
        search_path.append(resolved)
    return resolved

def instrument_package_dir(instrument: str, irdb_path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the instrument package directory inside an IRDB checkout."""
    if irdb_path is None:
        irdb_path = configure_irdb_path()
    return pathlib.Path(irdb_path).expanduser().resolve() / instrument

####### Configuring logging ##########

def set_scopesim_log_level(level: int | str) -> None:
    """Set ScopeSim's logging level for notebook sessions."""
    sim.utils.set_console_log_level(level)

def disable_scopesim_top_level_catch() -> None:
    """Unwrap ScopeSim's bug-report decorator while debugging in notebooks."""
    cls = optical_train.OpticalTrain
    for method_name in ("__init__", "observe"):
        method = getattr(cls, method_name)
        unwrapped_method = inspect.unwrap(method)
        if unwrapped_method is not method:
            setattr(cls, method_name, unwrapped_method)
    sim.OpticalTrain = cls

@contextmanager
def disable_scopesim_progress_bars(disable: bool = True):
    """Temporarily suppress ScopeSim observe-time tqdm progress bars.

    ScopeSim imports ``tqdm`` directly in ``optical_train``. Patching that
    module binding keeps the behavior local to the active notebook cell and
    avoids changing ScopeSim or rebuilding the optical train.
    """
    if not disable:
        yield
        return

    original_tqdm = optical_train.tqdm

    def quiet_tqdm(*args: Any, **kwargs: Any):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    optical_train.tqdm = quiet_tqdm
    try:
        yield
    finally:
        optical_train.tqdm = original_tqdm

def warning_prevent_sync_alt_ra_dec(cmd: Any) -> None:
    """Populate AltAz command keys without letting RA/Dec override airmass."""
    from scopesim.utils import (airmass2zendist, from_currsys)

    alt = from_currsys("!OBS.alt", cmd) if "!OBS.alt" in cmd else None
    az = from_currsys("!OBS.az", cmd) if "!OBS.az" in cmd else 0.0
    airmass = from_currsys("!OBS.airmass", cmd) if "!OBS.airmass" in cmd else None

    if alt is None and airmass is not None:
        alt = 90.0 - airmass2zendist(float(airmass))
    if alt is not None:
        if airmass is not None:
            airmass = float(airmass)
            alt_airmass = 90.0 - airmass2zendist(airmass)
            if not abs(alt_airmass - airmass) < 0.01:
                warnings.warn(f"Both !OBS.alt ({alt}) and !OBS.airmass ({airmass}) are set, but they are inconsistent. "
                          f"Using !OBS.alt and ignoring !OBS.airmass.", UserWarning)
        cmd["!OBS.alt"] = alt
        cmd["!OBS.az"] = az
        cmd["!OBS.ra"] = None
        cmd["!OBS.dec"] = None

def ignore_warnings() -> None:
    """Hide common notebook/debugger deprecation warnings."""
    warning_rules = (
        {
            "message": "co_lnotab is deprecated, use co_lines instead.",
            "category": DeprecationWarning,
            "module": r".*pydevd_collect_try_except_info",
        },
        {
            "message": r"metadata \{'args': \(None,\)\} was set from the constructor.*",
            "category": DeprecationWarning,
            "module": r"bqscales\.traits",
        },
        {
            "message": r"Passing unrecognized arguments to super\(DataGrid\)\.__init__\(display_length=-1\).*",
            "category": DeprecationWarning,
            "module": r"traitlets\.traitlets",
        },
        {
            "message": r"The fov_grid method is deprecated.*",
            "category": DeprecationWarning,
            "module": r"scopesim\.effects\.spectral_trace_list",
        },
    )
    for rule in warning_rules:
        warnings.filterwarnings("ignore", **rule)

def show_yappi_stats(sort: str = "tsub", print_output=True) -> str:
    """Return formatted yappi function stats for notebook display."""
    import yappi

    out = io.StringIO()
    stats = yappi.get_func_stats()
    stats.sort(sort, "desc")
    stats.print_all(out=out, columns={
        0: ("name", 80),
        1: ("ncall", 10),
        2: ("tsub", 10),
        3: ("ttot", 10),
        4: ("tavg", 10),
    })
    formatted = out.getvalue()
    if print_output:
        print(formatted)
    return formatted
