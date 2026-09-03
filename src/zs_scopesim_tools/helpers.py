"""Notebook/session helpers for ZShooter ScopeSim workflows."""

from __future__ import annotations

import inspect
import io
from pathlib import Path
import warnings
from contextlib import contextmanager
from typing import Any
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from IPython.display import display

import scopesim as sim
import scopesim.optics.optical_train as optical_train
from scopesim.utils import from_currsys

from zs_scopesim_tools.paths import resolve_irdb_path, make_output_dir

####### Path abstractions ########
def configure_irdb_path(fallback_irdb_path: str | Path | None = None) -> str:
    """Resolve the local IRDB checkout or installed editable package root.
     Then Configure ScopeSim to find the local IRDB package once."""
    resolved = str(resolve_irdb_path(fallback_irdb_path))

    sim.rc.__config__["!SIM.file.local_packages_path"] = resolved
    search_path = sim.rc.__config__["!SIM.file.search_path"]
    if resolved not in search_path:
        search_path.append(resolved)
    return resolved

def instrument_package_dir(instrument: str) -> Path:
    """Return the instrument package directory inside IRDB checkout."""
    inst_path = Path(sim.rc.__config__["!SIM.file.local_packages_path"]).resolve() / instrument
    if not inst_path.exists():
        inst_path = resolve_irdb_path() / instrument
        if inst_path.exists():
            warnings.warn(f"Instrument package {instrument} not found in ScopeSim local packages path: "
                          f"{sim.rc.__config__['!SIM.file.local_packages_path']}, "
                          f"but found in: {inst_path}. Configuring ScopeSim to use this IRDB path instead.")
            configure_irdb_path(inst_path.parent)
        else:
            raise FileNotFoundError(inst_path)
    return inst_path

def make_local_output_dir(dirname: str = "outputs") -> Path:
    """Return a repo-local output directory, avoiding accidental writes into the IRDB checkout."""
    return make_output_dir(dirname=dirname, avoid_path=resolve_irdb_path())

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
    from astropy import units as u
    from astropy.coordinates import AltAz
    from scopesim.utils import airmass2zendist, from_currsys, get_observation_info_from_cmds

    def degree_value(value: Any) -> float:
        quantity = u.Quantity(value)
        if quantity.unit == u.dimensionless_unscaled:
            return float(quantity.value)
        return float(quantity.to_value(u.deg))

    alt = from_currsys("!OBS.alt", cmd) if "!OBS.alt" in cmd else None
    az = from_currsys("!OBS.az", cmd) if "!OBS.az" in cmd else 0.0
    airmass = from_currsys("!OBS.airmass", cmd) if "!OBS.airmass" in cmd else None

    if alt is None and airmass is not None:
        alt = 90.0 - airmass2zendist(float(airmass))
    if alt is not None:
        if airmass is not None:
            airmass = float(airmass)
            alt_airmass = 90.0 - airmass2zendist(airmass)
            if not abs(alt_airmass - degree_value(alt)) < 0.01:
                warnings.warn(f"Both !OBS.alt ({alt}) and !OBS.airmass ({airmass}) are set, but they are inconsistent. "
                          f"Using !OBS.alt and ignoring !OBS.airmass.", UserWarning)
        cmd["!OBS.alt"] = degree_value(alt)
        cmd["!OBS.az"] = degree_value(az)
    else:
        target, location, time = get_observation_info_from_cmds(cmd)
        altaz_target = target if hasattr(target, "alt") and hasattr(target, "az") else target.transform_to(AltAz(obstime=time, location=location))
        cmd["!OBS.alt"] = degree_value(altaz_target.alt)
        cmd["!OBS.az"] = degree_value(altaz_target.az)
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

def display_frame(data, *, columns=None, rename=None, formats=None):
    if isinstance(data, Table):
        view = data[columns] if columns is not None else data
        frame = view.to_pandas()
    else:
        frame = pd.DataFrame(data)
        if columns is not None:
            frame = frame.loc[:, columns]
    if rename:
        frame = frame.rename(columns=rename)
    styler = frame.style.hide(axis="index")
    if formats:
        active_formats = {key: value for key, value in formats.items() if key in frame.columns}
        styler = styler.format(active_formats, na_rep="--")
    display(styler)

###################### Observing helpers ######################
def list_effects(cmds):
    """Return a list of effect names in the optical train for the given commands without initialising the train."""
    data = []
    for yaml_dict in cmds.yaml_dicts:
        yname = yaml_dict["name"]
        alias = yaml_dict["alias"]
        for eff in yaml_dict.get("effects", []):
            kwargs = eff.get("kwargs", {})
            data.append({"included": from_currsys(eff.get("include", True), cmds), "effect": eff["name"], "class": eff["class"],
                         "config": f"{yname} ({alias})", "selector": from_currsys(kwargs.get("selector", ""), cmds),
                         "data file": from_currsys(kwargs.get("filename", ""), cmds)
                         })
    return pd.DataFrame(data)


_BACKGROUND_SELECTORS = {"sky", "sky_continuum", "sky_lines"}

@contextmanager
def disable_background(name_or_names, train):
    """
    Context manager to temporarily disable specific background selectors in the ScopeSim optical train.
    :param name_or_names: list of background selector names to disable, or a single name as a string. Valid options are "sky", "sky_continuum", and "sky_lines".
    :param train: optical train instance (scopesim.OpticalTrain) to modify.
    """
    if name_or_names is None:
        yield
        return

    requested = {name_or_names} if isinstance(name_or_names, str) else set(name_or_names)
    unknown = requested - _BACKGROUND_SELECTORS
    if unknown:
        raise ValueError(f"Unknown background selector(s): {sorted(unknown)}")

    disable_continuum = "sky" in requested or "sky_continuum" in requested
    disable_lines = "sky" in requested or "sky_lines" in requested

    sky_continuum = train["continuum_emission"]
    palace = train["airglow_and_interline_continuum"]

    saved = {
        "sky_continuum_include": sky_continuum.include,
        "palace_include": palace.include,
        "only_line": palace.meta["only_line"],
        "only_continuum": palace.meta["only_continuum"],
    }

    try:
        sky_continuum.include = saved["sky_continuum_include"] and not disable_continuum
        palace.include = saved["palace_include"] and not (disable_continuum and disable_lines)
        palace.meta["only_line"] = disable_continuum and not disable_lines
        palace.meta["only_continuum"] = disable_lines and not disable_continuum
        yield
    finally:
        sky_continuum.include = saved["sky_continuum_include"]
        palace.include = saved["palace_include"]
        palace.meta["only_line"] = saved["only_line"]
        palace.meta["only_continuum"] = saved["only_continuum"]

def simulate(cmds, *,
             source=None,
             disable_effects=None,
             disable_backgrounds=None,
             hide_progress_bars=True):
    """
    Run a ScopeSim observation with optional effect disabling and progress bar suppression.

    Parameters
    ----------
    cmds : sim.UserCommands
        The ScopeSim user commands for the observation.
    source : sim.Source | None, optional
        The source to observe. If None, empty sky is simulated.
    disable_effects : list[str] | None, optional
        A list of effect names to disable during the observation.
    disable_backgrounds : list[str] | None, optional
        A list of background selectors to disable during the observation, options are "sky", "sky_continuum", and "sky_lines".
    hide_progress_bars : bool, optional
        If True, suppress the tqdm progress bars during the observation.
    """
    train = sim.OpticalTrain(cmds)

    effect_names = list(disable_effects or [])
    for effect in effect_names:
        if effect in train.effects['name']:
            train[effect].include = False
        else:
            warnings.warn(f"Effect with name {effect} not present in the train.")

    with disable_background(disable_backgrounds, train), disable_scopesim_progress_bars(hide_progress_bars):
        if source is None:
            train.observe()
        else:
            train.observe(source)
        hdul = train.readout()
    return train, hdul

def add_cmds_to_readout_header(hdul: fits.HDUList, cmds: sim.UserCommands, train: sim.OpticalTrain):
    """Add the command dictionary to the readout FITS header using the SimulationConfigFitsKeywords effect.
    New headers get added to the primary HDU of the readout FITS file.
    """
    hdreff = sim.effects.fits_headers.SimulationConfigFitsKeywords(cmds=cmds)
    if isinstance(hdul, fits.HDUList):
        return hdreff.apply_to(hdul, optical_train=train)
    else:
        raise TypeError(f"Expected hdul to be an instance of astropy.io.fits.HDUList, got {type(hdul)} instead.")

def save_readout_to_fits(hdul: fits.HDUList, filename: str):
    """Save the readout HDUList to a FITS file."""
    if isinstance(hdul, fits.HDUList):
        hdul.writeto(filename, overwrite=True)
    else:
        raise TypeError(f"Expected hdul to be an instance of astropy.io.fits.HDUList, got {type(hdul)} instead.")

def save_zshooter_readout(list_of_hdul: list[fits.HDUList], output_dir: str | Path,
                          *, filename_prefix: str = 'sim', imagetype: str = 'OBJECT',
                          cmds: sim.UserCommands | None = None, train: sim.OpticalTrain | None = None):
    """
    Save a list of readout HDULists to FITS files from ZShooter spectral channels.
    :param list_of_hdul: List of HDULists corresponding to the ZShooter channels (blue, green, red, yj, h, k).
    :param output_dir: Directory to save the FITS files.
    :param filename_prefix: Prefix for the output filenames (default: 'sim').
    :param imagetype: Value for the HIERARCH IMAGETYPE header keyword (default: 'OBJECT').
    :param cmds: Optional UserCommands object to add to the FITS headers.
    :param train: Optional OpticalTrain object to use for header information.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    channels = ["blue", "green", "red", "yj", "h", "k"]
    outfiles = []

    for i, hdul in enumerate(list_of_hdul):
        if cmds is not None and train is not None:
            hdul = add_cmds_to_readout_header(hdul, cmds, train)
            hdul[1].header['EXPTIME'] = hdul[0].header[f"HIERARCH SIM CONFIG OBS dit_{channels[i]}"]
            hdul[1].header['OBJECT'] = filename_prefix.upper()
            hdul[1].header['HIERARCH IMAGETYPE'] = imagetype.upper()

        filename = output_dir / f"{filename_prefix}_{channels[i].upper()}.fits"
        save_readout_to_fits(hdul, str(filename))
        outfiles.append(filename)
    return outfiles
