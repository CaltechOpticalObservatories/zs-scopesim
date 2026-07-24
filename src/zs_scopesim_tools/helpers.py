"""Notebook/session helpers for ZShooter ScopeSim workflows."""

from __future__ import annotations

import importlib.resources
import inspect
import io
import pathlib
import warnings
from contextlib import contextmanager
from typing import Any


def repo_root(start: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the zs-scopesim repository root."""
    path = (
        pathlib.Path(start).expanduser().resolve()
        if start
        else pathlib.Path(__file__).resolve()
    )
    candidates = [path] if path.is_dir() else [path.parent]
    candidates.extend(candidates[0].parents)
    for parent in candidates:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return pathlib.Path.cwd().resolve()


def resolve_irdb_path(irdb_path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Resolve the local IRDB checkout or installed editable package root."""
    try:
        return pathlib.Path(importlib.resources.files("irdb")).parent.resolve()
    except ModuleNotFoundError:
        if irdb_path is None:
            irdb_path = pathlib.Path.home() / "src" / "irdb"
        resolved = pathlib.Path(irdb_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved


def configure_path_and_logging(
    irdb_path: str | pathlib.Path | None = None,
    level: str = "debug",
) -> str:
    """Configure ScopeSim to find the local IRDB package once."""
    import scopesim as sim

    resolved = str(resolve_irdb_path(irdb_path))
    sim.rc.__config__["!SIM.file.local_packages_path"] = resolved

    search_path = sim.rc.__config__["!SIM.file.search_path"]
    if resolved not in search_path:
        search_path.append(resolved)

    sim.utils.set_console_log_level(level)
    return resolved


def zshooter_package_dir(irdb_path: str | pathlib.Path) -> pathlib.Path:
    """Return the ZShooter_v2 package directory inside an IRDB checkout."""
    return pathlib.Path(irdb_path).expanduser().resolve() / "ZShooter_v2"


def validation_work_dir(
    irdb_path: str | pathlib.Path | None = None,
    name: str = "validation_outputs",
    base_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Return a repo-local output directory, avoiding accidental writes into IRDB."""
    cwd = (
        pathlib.Path(base_dir).expanduser().resolve()
        if base_dir
        else pathlib.Path.cwd().resolve()
    )
    if irdb_path is not None:
        package_dir = zshooter_package_dir(irdb_path)
        if cwd == package_dir or package_dir in cwd.parents:
            cwd = repo_root()
    output_dir = cwd / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def disable_scopesim_top_level_catch() -> None:
    """Unwrap ScopeSim's bug-report decorator while debugging in notebooks."""
    import scopesim
    import scopesim.optics.optical_train as optical_train

    cls = optical_train.OpticalTrain
    for method_name in ("__init__", "observe"):
        method = getattr(cls, method_name)
        unwrapped_method = inspect.unwrap(method)
        if unwrapped_method is not method:
            setattr(cls, method_name, unwrapped_method)
    scopesim.OpticalTrain = cls


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

    import scopesim.optics.optical_train as optical_train

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
    from astropy import units as u
    from astropy.coordinates import AltAz
    from scopesim.utils import (
        airmass2zendist,
        from_currsys,
        get_observation_info_from_cmds,
    )

    def cmd_value(key: str, default: Any = None) -> Any:
        try:
            return from_currsys(key, cmd)
        except (KeyError, ValueError):
            return default

    def degree_value(value: Any) -> float:
        quantity = u.Quantity(value)
        if quantity.unit == u.dimensionless_unscaled:
            return float(quantity.value)
        return float(quantity.to_value(u.deg))

    alt = cmd_value("!OBS.alt")
    az = cmd_value("!OBS.az", 0.0)
    airmass = cmd_value("!OBS.airmass")
    if alt is None and airmass is not None:
        alt = 90.0 - airmass2zendist(float(airmass))
    if alt is not None:
        cmd["!OBS.alt"] = degree_value(alt)
        cmd["!OBS.az"] = degree_value(az)
    else:
        target, location, time = get_observation_info_from_cmds(cmd)
        if hasattr(target, "alt") and hasattr(target, "az"):
            altaz_target = target
        else:
            altaz_target = target.transform_to(
                AltAz(obstime=time, location=location),
            )
        cmd["!OBS.alt"] = float(altaz_target.alt.to("deg").value)
        cmd["!OBS.az"] = float(altaz_target.az.to("deg").value)
    cmd["!OBS.ra"] = None
    cmd["!OBS.dec"] = None


def ignore_warnings() -> None:
    """Hide PyCharm debugger's Python 3.14 co_lnotab deprecation warning."""
    warnings.filterwarnings(
        "ignore",
        message="co_lnotab is deprecated, use co_lines instead.",
        category=DeprecationWarning,
        module=r".*pydevd_collect_try_except_info",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"metadata \{'args': \(None,\)\} was set from the constructor.*",
        category=DeprecationWarning,
        module=r"bqscales\.traits",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Passing unrecognized arguments to super\(DataGrid\)\.__init__\(display_length=-1\).*",
        category=DeprecationWarning,
        module=r"traitlets\.traitlets",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"The fov_grid method is deprecated.*",
        category=DeprecationWarning,
        module=r"scopesim\.effects\.spectral_trace_list",
    )


def format_yappi_stats(sort: str = "tsub") -> str:
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
    return out.getvalue()


def print_yappi_stats(sort: str = "tsub") -> None:
    """Print formatted yappi function stats in a notebook cell."""
    print(format_yappi_stats(sort=sort))
