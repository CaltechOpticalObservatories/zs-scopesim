"""Source-construction helpers for ZShooter validation notebooks."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from synphot.units import PHOTLAM, convert_flux

import scopesim.source.source_templates as source_templates
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
              wave_step: u.Quantity = 0.05 * u.AA, resolving_power: float = 8e4, extent: float = 60):
    """Build a calibration-line flat source with flux-preserving line sampling."""
    if line_dir is None:
        line_dir = instrument_package_dir("ZShooter_v2") / "lamp_lines"
    else:
        line_dir = Path(line_dir).expanduser().resolve()

    lines = _read_lamp_lines(line_dir, filenames=filenames)
    centers = np.asarray(lines["wave"], dtype=float)  # Angstrom
    amplitudes = np.asarray(lines["amplitude"], dtype=float)
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
                                           points=wave.to_value(u.AA), lookup_table=flux.value)

def transient(*, x: float = 0.0, y: float = 0.0,
              spextrum_template: str | None = None,
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
        from spextra import Spextrum
        sed = Spextrum(spextrum_template)
        sed = sed.redshift(scale_to_redshift) if scale_to_redshift is not None else sed
        if scale_to_mag is not None:
            if not isinstance(scale_to_mag, u.Quantity):
                scale_to_mag = scale_to_mag * u.ABmag
            sed = sed.scale_to_magnitude(scale_to_mag, filter_curve)
        spectra = [sed]
    elif wavelength is not None and flux is not None:
        spectra = [empirical_spectrum(wavelength, flux)]
    else:
        raise ValueError("Either spextrum_template or both wavelength and flux must be provided.")
    # create transient source
    return source_templates.Source(x=[x], y=[y], ref=[0.], weight=[1.0], spectra=spectra, name=name)

