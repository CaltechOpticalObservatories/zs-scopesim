"""Source-construction helpers for ZShooter validation notebooks."""

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from synphot import SourceSpectrum
from synphot.units import PHOTLAM, FLAM, convert_flux
from astropy.constants import c
from astropy.cosmology import Planck18
import h5py

import scopesim.source.source_templates as source_templates
from spextra import Spextrum
from zs_scopesim_tools.helpers import instrument_package_dir


DEFAULT_LAMP_LINE_FILENAMES = ("ThAr_XSHOOTER_UVB_lines.dat", "Ne_IR_MOSFIRE_lines.dat", "Ar_IR_MOSFIRE_lines.dat")

def bin_edges_from_centers(centers: Any) -> np.ndarray:
    """Return wavelength bin edges from monotonically increasing centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a one-dimensional array with at least two values")
    mids = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.r_[first, mids, last]

def _erf(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf
        return erf(values)
    except ModuleNotFoundError:
        from math import erf
        return np.vectorize(erf, otypes=[float])(values)

def gaussian_line_flux_density(wave_centers: Any, line_centers: Any, line_fluxes: Any, line_fwhm: Any) -> np.ndarray:
    """Sample integrated Gaussian line fluxes as flux density on a grid.

    Each line is integrated over the output wavelength bins, then divided by
    bin width. This preserves line flux even when the line is narrow relative
    to the sampling grid and no bin center lands exactly on the peak.
    """
    wave_centers = np.asarray(wave_centers, dtype=float)
    line_centers = np.asarray(line_centers, dtype=float)
    line_fluxes = np.asarray(line_fluxes, dtype=float)
    line_fwhm = np.asarray(line_fwhm, dtype=float)

    edges = bin_edges_from_centers(wave_centers)
    widths = np.diff(edges)
    flux_per_bin = np.zeros_like(wave_centers, dtype=float)

    sigma = line_fwhm / (2 * np.sqrt(2 * np.log(2)))
    for center, flux, sig in zip(line_centers, line_fluxes, sigma):
        if sig <= 0 or not np.isfinite(sig):
            idx = np.searchsorted(edges, center) - 1
            if 0 <= idx < flux_per_bin.size:
                flux_per_bin[idx] += flux
            continue

        lo = np.searchsorted(edges, center - 8 * sig, side="left")
        hi = np.searchsorted(edges, center + 8 * sig, side="right")
        lo = max(lo - 1, 0)
        hi = min(hi + 1, edges.size)
        local_edges = edges[lo:hi]
        cdf = 0.5 * (1 + _erf((local_edges - center) / (np.sqrt(2) * sig)))
        flux_per_bin[lo:hi - 1] += flux * np.diff(cdf)

    return flux_per_bin / widths

@lru_cache(maxsize=None)
def _read_lamp_lines(line_dir: Path, filenames: Sequence[str] = DEFAULT_LAMP_LINE_FILENAMES) -> Table:
    """Read and combine ZShooter lamp-line tables."""
    tables = [Table.read(line_dir / filename, delimiter="|", format="ascii", header_start=0) for filename in filenames]
    return vstack(tables, metadata_conflicts="silent")

def lamp_flat(*, line_dir: str | Path | None = None, filenames: Sequence[str] = DEFAULT_LAMP_LINE_FILENAMES,
              wave_step: u.Quantity = 0.05 * u.AA, resolving_power: float = 8e4, extent: float = 60,
              scale_amplitude: float = 1.0):
    """Build a calibration-line flat source with flux-preserving line sampling."""
    if line_dir is None:
        line_dir = instrument_package_dir("ZShooter_v2") / "lamp_lines"
    else:
        line_dir = Path(line_dir).expanduser().resolve()

    lines = _read_lamp_lines(line_dir, filenames=tuple(filenames))
    centers = np.asarray(lines["wave"], dtype=float)  # Angstrom
    amplitudes = np.asarray(lines["amplitude"], dtype=float) * scale_amplitude
    fwhm = centers / resolving_power # Angstrom

    wave_min = max(100.0, np.nanmin(centers) - 20 * np.nanmax(fwhm))
    wave_max = np.nanmax(centers) + 20 * np.nanmax(fwhm)
    step = u.Quantity(wave_step, u.AA).to_value(u.AA)
    waves = np.arange(wave_min, wave_max + step, step)
    flux_density = gaussian_line_flux_density(waves, centers, amplitudes, fwhm)

    spectrum = empirical_spectrum(waves, flux_density)
    return source_templates.uniform_source(spectrum, extent=extent)

def ab_magnitude_flat(mag: float = 20, extent: float = 60):
    """Return a uniform flat-field source with constant AB magnitude."""
    return source_templates.uniform_source(source_templates.ab_spectrum(mag=mag), extent=extent)

def constant_photon_flux_flat(amplitude: float = 0.001, extent: float = 60,
                              wave_min: float = 100, wave_max: float = 300000, n_samples: int = 50000):
    """Return a uniform source with constant photon flux density."""
    waves = np.geomspace(wave_min, wave_max, n_samples)
    spectrum_model = source_templates.ConstFlux1D(amplitude=amplitude)
    spectrum = empirical_spectrum(waves, spectrum_model(waves))
    return source_templates.uniform_source(spectrum, extent=extent)

def slit_frame_offsets(separation: u.Quantity = 1.0 * u.arcsec, angle_on_slit: u.Quantity = 0.0 * u.deg, *, centered: bool = True) -> tuple[u.Quantity, u.Quantity]:
    """Return two source offsets with x along the slit and y across the slit."""
    sep = u.Quantity(separation).to(u.arcsec)
    angle = u.Quantity(angle_on_slit).to(u.rad)
    dx = sep * np.cos(angle)
    dy = sep * np.sin(angle)
    if centered:
        return u.Quantity([-0.5 * dx.value, 0.5 * dx.value], dx.unit), u.Quantity([-0.5 * dy.value, 0.5 * dy.value], dy.unit)
    return u.Quantity([0.0, dx.value], dx.unit), u.Quantity([0.0, dy.value], dy.unit)

def angle_from_cmds(cmds: Any, key: str = "!OBS.pupil_angle", default: u.Quantity = 0.0 * u.deg) -> u.Quantity:
    """Return an angle setting from a ScopeSim command object."""
    from scopesim.utils import from_currsys

    value = from_currsys(key, cmds) if key in cmds else default
    return u.Quantity(value, u.deg).to(u.deg)

def two_point_source(*, separation: u.Quantity = 1.0 * u.arcsec, angle_on_slit: u.Quantity = 0.0 * u.deg,
                     center: tuple[u.Quantity, u.Quantity] = (0.0 * u.arcsec, 0.0 * u.arcsec),
                     centered: bool = True, spectrum: Any | None = None, mag: float = 20.0,
                     weights: Sequence[float] = (1.0, 1.0)):
    """Build a two-point-source scene for slit-loss and ADC validation."""
    if len(weights) != 2:
        raise ValueError("weights must contain exactly two values")

    x_offsets, y_offsets = slit_frame_offsets(separation=separation, angle_on_slit=angle_on_slit, centered=centered)
    x = u.Quantity(center[0]).to(u.arcsec) + x_offsets
    y = u.Quantity(center[1]).to(u.arcsec) + y_offsets
    spectrum = spectrum if spectrum is not None else source_templates.ab_spectrum(mag=mag)
    table = Table(data=[x.to_value(u.arcsec), y.to_value(u.arcsec), np.asarray(weights, dtype=float),
                        np.zeros(2, dtype=int), ["source_0", "source_1"]],
                  names=["x", "y", "weight", "ref", "label"], units=[u.arcsec, u.arcsec, None, None, None])
    table.meta.update({"angle_on_slit": u.Quantity(angle_on_slit).to_value(u.deg), "angle_on_slit_unit": "deg",
                       "separation": u.Quantity(separation).to_value(u.arcsec), "separation_unit": "arcsec",
                       "centered": centered, "frame": "slit", "x_convention": "along slit", "y_convention": "across slit"})
    source = source_templates.Source(spectra=[spectrum], table=table)
    source.meta.update({"function_call": "two_point_source", "angle_on_slit": table.meta["angle_on_slit"],
                        "angle_on_slit_unit": "deg"})
    return source

def field_angle_demo_sources(*, along_separation: u.Quantity = 5.0 * u.arcsec,
                             across_separation: u.Quantity = 0.9 * u.arcsec,
                             angle_on_slit: u.Quantity = 0.0 * u.deg, along_mag: float = 15.0,
                             across_mag: float = 17.0, spectrum: Any | None = None) -> OrderedDict[str, Any]:
    """Return the two source scenes used for field-angle and slit validation."""
    angle_on_slit = u.Quantity(angle_on_slit, u.deg).to(u.deg)
    scenarios = OrderedDict([
        ("along_slit_centered", two_point_source(separation=along_separation, angle_on_slit=angle_on_slit,
                                                  centered=True, mag=along_mag, spectrum=spectrum)),
        ("across_slit_one_off", two_point_source(separation=across_separation,
                                                  angle_on_slit=angle_on_slit + 90 * u.deg,
                                                  centered=False, mag=across_mag, spectrum=spectrum)),
    ])
    descriptions = {"along_slit_centered": "Pair centered on the slit and separated along the slit.",
                    "across_slit_one_off": "Pair separated across the slit with the second source off slit."}
    for name, source in scenarios.items():
        source.meta.update({"name": name, "scenario": name, "function_call": "field_angle_demo_sources",
                            "description": descriptions[name], "scene_angle_on_slit": angle_on_slit.to_value(u.deg),
                            "scene_angle_on_slit_unit": "deg",
                            "workflow_note": "Set angle_on_slit explicitly in the slit frame, then rebuild this source scene and rerun observe/readout."})
    return scenarios

def empirical_spectrum(wave: u.Quantity, flux: u.Quantity):
    """
    Return a SourceSpectrum object created from an empirical model defined using input wave and flux u.Quantity arrays.
    If not Quantity, wave is assumed to be in Angstrom, and flux in PHOTLAM.
    """
    if not isinstance(wave, u.Quantity):
        wave = wave * u.AA
    if not isinstance(flux, u.Quantity):
        flux = flux * PHOTLAM
    flux = convert_flux(wave, flux, PHOTLAM)
    return source_templates.SourceSpectrum(source_templates.Empirical1D,
                                           points=wave.to_value(u.AA), lookup_table=flux.value, z_type="conserve_flux")

def transient(*, x: float = 0.0, y: float = 0.0,
              spextrum_template: str | SourceSpectrum | None = None,
              wavelength: u.Quantity | None = None,
              flux: u.Quantity | None = None,
              scale_to_redshift: float | None = None,
              scale_to_mag: float | u.Quantity | None = None,
              filter_curve: str = 'g',
              name: str = "AT"):
    """
    Create a transient source at given position with either a spextrum template or wavelength and flux arrays.
    :param x: float
        x position in arcsec
    :param y: float
        y position in arcsec
    :param spextrum_template: str
        Name of spextrum template, e.g. sne/sn1a (find templates here: https://scopesim.univie.ac.at/spextra/database/libraries/)
        or a SourceSpectrum object
    :param wavelength: u.Quantity
        Wavelength array (if spextrum_template is None)
    :param flux: u.Quantity
        Flux density array (if spextrum_template is None)
    :param scale_to_redshift: float | None
        If not None, scale spectrum to given redshift
    :param scale_to_mag: float | u.Quantity | None
        If not None, scale spectrum to given magnitude in given filter (default: g)
    :param filter_curve: str
        Filter curve to use for scaling to magnitude (default: g)
    :return: Source object
    """
    if spextrum_template is not None:
        if isinstance(spextrum_template, str):
            sed = Spextrum(template_name=spextrum_template)
        elif isinstance(spextrum_template, SourceSpectrum):
            sed = Spextrum(modelclass=spextrum_template)
        else:
            raise TypeError("spextrum_template must be a string or a SourceSpectrum object.")
    elif wavelength is not None and flux is not None:
        sed = empirical_spectrum(wavelength, flux)
        sed = Spextrum(modelclass=sed)  # convert to Spextrum for consistency
    else:
        raise ValueError("Either spextrum_template or both wavelength and flux must be provided.")

    sed = sed.redshift(scale_to_redshift) if scale_to_redshift is not None else sed
    if scale_to_mag is not None:
        if not isinstance(scale_to_mag, u.Quantity):
            scale_to_mag = scale_to_mag * u.ABmag
        sed = sed.scale_to_magnitude(scale_to_mag, filter_curve)
    spectra = [sed]
    # create transient source
    return source_templates.Source(x=[x], y=[y], ref=[0.], weight=[1.0], spectra=spectra, name=name)

def kilonova_spectra(model_files: Sequence[str | Path], phases: Sequence[float] = (1.5, 7.5), *,
                      redshift: float, combine_models: bool = True):
    """Read Kasen-model spectra and interpolate them to requested rest-frame phases in days."""
    if redshift <= 0:
        raise ValueError("redshift must be positive so the model luminosity has a finite flux scale")

    distance = Planck18.luminosity_distance(redshift).to_value(u.cm)
    speed_of_light = c.to_value(u.cm / u.s)
    spectra = {phase: [] for phase in phases}

    for model_file in model_files:
        with h5py.File(model_file, "r") as model:
            nu = np.array(model["nu"], dtype=float)
            times = np.array(model["time"], dtype=float) / 86400
            lnu_all = np.array(model["Lnu"], dtype=float)

        for phase in phases:
            if phase < times[0] or phase > times[-1]:
                raise ValueError(f"phase {phase} d is outside the model range {times[0]}--{times[-1]} d")
            upper = np.searchsorted(times, phase)
            if upper == 0 or times[upper] == phase:
                lnu = lnu_all[upper]
            else:
                lower = upper - 1
                fraction = (phase - times[lower]) / (times[upper] - times[lower])
                lnu = lnu_all[lower] + fraction * (lnu_all[upper] - lnu_all[lower])

            wavelength = speed_of_light / nu * 1e8
            llam = lnu * nu ** 2 / speed_of_light / 1e8
            flux = llam / (4 * np.pi * distance ** 2)
            spectra[phase].append(empirical_spectrum(wavelength * u.AA, flux * FLAM))

    if combine_models:
        for phase, model_spectra in spectra.items():
            combined = model_spectra[0]
            for spectrum in model_spectra[1:]:
                combined += spectrum
            spectra[phase] = combined
    return spectra
