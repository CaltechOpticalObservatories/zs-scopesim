"""Notebook/session helpers for ZShooter ScopeSim workflows."""

from __future__ import annotations

import inspect
import io
import gzip
from pathlib import Path
import warnings
from contextlib import contextmanager
from typing import Any
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from IPython.display import display
from copy import deepcopy
import numpy as np
import time
from functools import wraps

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


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}({kwargs.get('object_name', '')}): {time.perf_counter() - t0:.1f} s")
        return result
    return wrapper


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

def zshooter_commands(detector_mode, exptimes=None, ndit=1, *, airmass=None, seeing=None, brightness=None, vis_slit=None, nir_slit=None, pixel_scale=None, pupil_angle=None, ao_on=False, diffuse_emissivity_on=False, params=None):
    """Configure DIT (seconds) and NDIT in BLUE/GREEN/RED/YJ/H/K order."""
    config = sim.UserCommands(use_instrument="ZShooter_v2", set_modes=[detector_mode])
    overrides = {"!OBS.airmass": airmass, "!OBS.seeing": seeing, "!OBS.brightness": brightness, "!OBS.pupil_angle": pupil_angle, "!INST.vis_curr_slit": vis_slit, "!INST.nir_curr_slit": nir_slit, "!INST.pixel_scale": pixel_scale}
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["!INST.include_diffuse_emissivity"] = diffuse_emissivity_on
    config["!ATMO.ao_enabled"] = ao_on
    dits = [0.]*6 if exptimes is None else exptimes if isinstance(exptimes, list) else [exptimes]*6
    ndits = ndit if isinstance(ndit, list) else [ndit]*6
    config["!OBS.dit"], config["!OBS.ndit"] = dits[0], ndits[0]
    for dit, count, arm in zip(dits, ndits, ("blue", "green", "red", "yj", "h", "k"), strict=True):
        config[f"!OBS.dit_{arm}"] = dit
        config[f"!OBS.ndit_{arm}"] = count
    if params is not None:
        for key, value in params.items():
            config[key] = value
    warning_prevent_sync_alt_ra_dec(config)
    return config


def zshooter_channel_configuration(header, channel):
    """Physical simulation settings in a saved ScopeSim header for one arm."""
    arm_names = {"BLUE": ("b", "blue"), "GREEN": ("g", "green"), "RED": ("r", "red"), "YJ": ("yj",), "H": ("h",), "K": ("k",)}
    other_arms = tuple("_" + name for arm, names in arm_names.items() if arm != channel for name in names)
    configuration = {}
    for key in header:
        if not key.startswith("SIM CONFIG "):
            continue
        if key.startswith(("SIM CONFIG SIM file search_path", "SIM CONFIG SIM logging ", "SIM CONFIG SIM reports ")):
            continue
        setting = key.split()[3]
        if key in ("SIM CONFIG OBS modes", "SIM CONFIG DET detector_mode", "SIM CONFIG OBS dit", "SIM CONFIG OBS ndit", "SIM CONFIG OBS exptime", "SIM CONFIG OBS roid"):
            continue
        if key.startswith(("SIM CONFIG DET ", "SIM CONFIG OBS ")) and setting.endswith(other_arms):
            continue
        configuration[key] = header[key]
    return configuration


def simulate(cmds, *,
             source=None,
             disable_effects=None,
             disable_backgrounds=None,
             hide_progress_bars=True,
             image_plane_ids=None):
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
    image_plane_ids : sequence of int | None
        Configured image planes to observe. Their physical detector IDs are retained.
    """
    train = sim.OpticalTrain(cmds)

    effect_names = list(disable_effects or [])
    for effect in effect_names:
        if effect in train.effects['name']:
            train[effect].include = False
        else:
            warnings.warn(f"Effect with name {effect} not present in the train.")

    with disable_background(disable_backgrounds, train), disable_scopesim_progress_bars(hide_progress_bars):
        if image_plane_ids is not None:
            train.update()
            train.fov_manager._fovs_list = [fov for fov in train.fov_manager.fovs if fov.meta["image_plane_id"] in image_plane_ids]
            train.detector_managers = [manager for manager in train.detector_managers if manager._detector_list.image_plane_id in image_plane_ids]
        train.observe(source, update=image_plane_ids is None)
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
    """Save a readout losslessly, using fast gzip for a .gz destination."""
    if isinstance(hdul, fits.HDUList):
        if Path(filename).suffix == '.gz':
            with gzip.open(filename, 'wb', compresslevel=1) as stream:
                hdul.writeto(stream)
        else:
            hdul.writeto(filename, overwrite=True)
    else:
        raise TypeError(f"Expected hdul to be an instance of astropy.io.fits.HDUList, got {type(hdul)} instead.")


def _detector_dark_rate(train, index):
    from scopesim.effects import SelectorWheel
    from scopesim.effects.electronic import DarkCurrent
    from scopesim.effects.electronic.pixels import BinnedImageBase

    detector, = train.detector_managers[index]
    effects = [effect.get_effect(detector.det_id) if isinstance(effect, SelectorWheel) else effect for effect in train.optics_manager.detector_effects]
    dark, = [effect for effect in effects if isinstance(effect, DarkCurrent)]
    pixel_area = np.prod([np.prod(effect.bin_factors) for effect in effects if isinstance(effect, BinnedImageBase)])
    return dark._get_dark_level(detector.det_id) * pixel_area


def save_zshooter_readout(list_of_hdul: list[fits.HDUList], output_dir: str | Path,
                          *, object_name: str = 'sim', imagetype: str = '', ndit: int = 0,
                          cmds: sim.UserCommands | None = None, train: sim.OpticalTrain | None = None,
                          extra_headers: dict | None = None):
    """
    Save ZShooter channel readouts as .fits.gz and return their paths.
    :param list_of_hdul: List of HDULists corresponding to the ZShooter channels (blue, green, red, yj, h, k).
    :param output_dir: Directory to save the FITS files.
    :param object_name: Name of the object being observed (default: 'sim').
    :param imagetype: Value for the HIERARCH IMAGETYPE header keyword (default: 'OBJECT').
    :param ndit: Number of image in the sequence (default: 1).
    :param cmds: Optional UserCommands object to add to the FITS headers.
    :param train: Optional OpticalTrain object to use for header information.
    :param extra_headers: Optional dictionary of additional header keywords to add to the primary HDU.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    channels = ["blue", "green", "red", "yj", "h", "k"]
    outfiles = []

    for i, hdul in enumerate(list_of_hdul):
        channel = channels[hdul[1].header['ID']].upper()
        hdul[0].header['OBJECT'] = object_name.upper()
        hdul[0].header['HIERARCH IMAGETYPE'] = imagetype.upper()
        hdul[0].header['HIERARCH IMAGENUM'] = ndit
        hdul[0].header['CHANNEL'] = channel

        if train is not None and 'VARIANCE' not in hdul:
            from zs_scopesim_tools.comparison import detector_noise
            exptime, dark, read_variance = detector_noise(train, i)
            dark_rate = _detector_dark_rate(train, i)
            for header in (hdul[0].header, hdul[1].header):
                header['RON'] = (np.sqrt(read_variance), 'Accumulated read noise [electron RMS]')
                header['DARK'] = (dark_rate, 'Dark rate per output pixel [electron/s]')
                header['GAIN'] = (1., 'ZShooter readouts are in electrons')
                header['BUNIT'] = 'electron'
                header['EXPTIME'] = exptime
                header['DIT'] = hdul[1].header['DIT']
                header['NDIT'] = hdul[1].header['NDIT']
                header['COMBINE'] = 'sum'

        if train is not None and 'continuum_emission' in train:
            hdul[0].header['MJD-OBS'] = train['continuum_emission'].time.mjd
        elif 'MJD-OBS' not in hdul[0].header:
            hdul[0].header['MJD-OBS'] = Time.now().mjd
        hdul[0].header['DATE-OBS'] = Time(hdul[0].header['MJD-OBS'], format='mjd').isot

        if cmds is not None and train is not None:
            hdul = add_cmds_to_readout_header(hdul, cmds, train)

        if extra_headers is not None:
            for key, value in extra_headers.items():
                hdul[0].header[key] = value

        filename_prefix = f"{object_name}_{ndit}"
        filename = output_dir / f"{filename_prefix}_{channel}.fits.gz"
        save_readout_to_fits(hdul, str(filename))
        outfiles.append(filename)
    return outfiles

def subtract_hduls(hdul0, hdul1):
    """
    Subtract two HDULists (hdul0 - hdul1) element-wise, returning a new HDUList with the same structure, and header of hdul0.
    Assumes that both HDULists have the same number of HDUs and compatible data shapes.
    """
    if len(hdul0) != len(hdul1):
        raise ValueError("HDULists must have the same number of HDUs to subtract.")

    sub_hdul = []
    for i in range(len(hdul0)):
        hdu = deepcopy(hdul0[i])
        dat0 = hdul0[i][1].data
        dat1 = hdul1[i][1].data
        hdu[1].data = dat0 - dat1
        if ('VARIANCE' in hdul0[i]) != ('VARIANCE' in hdul1[i]):
            raise ValueError('Both images need variance to propagate a subtraction.')
        if 'VARIANCE' in hdul0[i]:
            hdu['VARIANCE'].data = hdul0[i]['VARIANCE'].data + hdul1[i]['VARIANCE'].data
        sub_hdul.append(hdu)
    return sub_hdul


def electron_readouts(train, readouts):
    """Copy a current readout into accumulated electrons with its expected pixel variance.

    Call before observing another scene: the current image planes supply the
    Poisson mean, including source, sky and thermal background. Random draws
    in the science image are retained. Gain and NIR ramp averaging are undone.
    """
    from scopesim.effects import SelectorWheel
    from scopesim.effects.electronic import ADConversion, ExposureOutput
    from zs_scopesim_tools.comparison import detector_noise, detector_rates

    rates = detector_rates(train)
    result = []
    for i, (manager, hdul, rate) in enumerate(zip(train.detector_managers, readouts, rates, strict=True)):
        detector, = manager
        scale = 1.
        for effect in train.optics_manager.detector_effects:
            selected = effect.get_effect(detector.det_id) if isinstance(effect, SelectorWheel) else effect
            if isinstance(selected, ADConversion):
                scale *= selected._get_gain(detector.det_id)
            if isinstance(selected, ExposureOutput) and selected.current_mode == 'average':
                scale *= from_currsys(selected.meta['ndit'], train.cmds)
        exptime, dark, read_variance = detector_noise(train, i)
        copied = fits.HDUList([hdu.copy() for hdu in hdul])
        copied[1].data = hdul[1].data * scale
        factor = detector._native_header['EXNOISE']
        variance = factor * rate * exptime + dark + read_variance
        if np.any(variance < 0):
            raise ValueError('The configured source/background has a negative expected pixel variance.')
        copied.append(fits.ImageHDU(variance, name='VARIANCE'))
        copied['VARIANCE'].header['BUNIT'] = 'electron2'
        dark_rate = _detector_dark_rate(train, i)
        for header in (copied[0].header, copied[1].header):
            header['BUNIT'] = 'electron'
            header['GAIN'] = 1.
            header['COMBINE'] = 'sum'
            header['EXNOISE'] = (factor, 'Photon variance multiplier')
            header['EXPTIME'] = exptime
            header['DIT'] = hdul[1].header['DIT']
            header['NDIT'] = hdul[1].header['NDIT']
            header['RON'] = (np.sqrt(read_variance), 'Accumulated read noise [electron RMS]')
            header['DARK'] = (dark_rate, 'Dark rate per output pixel [electron/s]')
        result.append(copied)
    return result


def combine_readouts(readout_groups, coefficients, *, source_indices):
    """Combine unique raw exposures before extraction, retaining reused-sky variance.

    For A1 + A2 - 2*B, pass the three unique electron-readout groups with
    coefficients [1, 1, -2] and source_indices [0, 1]. B then contributes
    four times its variance. On-source exposure excludes the sky integrations.
    Source coefficients must be +1 so exposure and counts describe a sum.
    Source integrations must share one DIT to retain a truthful DIT/NDIT header.
    """
    if len(readout_groups) != len(coefficients):
        raise ValueError('Each unique exposure needs one combination coefficient.')
    if any(coefficients[j] != 1 for j in source_indices):
        raise ValueError('On-source exposures must enter the accumulated-electron sum with coefficient +1.')
    result = []
    for arm, hdul in enumerate(readout_groups[0]):
        dit = readout_groups[source_indices[0]][arm][1].header['DIT']
        if any(readout_groups[j][arm][1].header['DIT'] != dit for j in source_indices):
            raise ValueError('A combined DIT/NDIT header requires the same source integration time.')
        copied = fits.HDUList([hdu.copy() for hdu in hdul])
        copied[1].data = np.zeros_like(hdul[1].data, dtype=np.float64)
        copied['VARIANCE'].data = np.zeros_like(hdul['VARIANCE'].data, dtype=np.float64)
        for group, coefficient in zip(readout_groups, coefficients, strict=True):
            if group[arm][1].header['BUNIT'] != 'electron':
                raise ValueError('Combine accumulated electron readouts, not ADU or averaged ramps.')
            copied[1].data += coefficient * group[arm][1].data
            copied['VARIANCE'].data += coefficient**2 * group[arm]['VARIANCE'].data
        exptime = sum(readout_groups[j][arm][1].header['EXPTIME'] for j in source_indices)
        for header in (copied[0].header, copied[1].header):
            header['EXPTIME'] = exptime
            header['DIT'] = dit
            header['HIERARCH SUBTRACTED'] = True
            header['NDIT'] = sum(readout_groups[j][arm][1].header['NDIT'] for j in source_indices)
            header['RON'] = np.sqrt(sum(c**2 * group[arm][1].header['RON']**2 for group, c in zip(readout_groups, coefficients, strict=True)))
        result.append(copied)
    return result
