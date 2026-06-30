"""Source-construction helpers for ZShooter validation notebooks."""

from __future__ import annotations

import math
import pathlib
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack


DEFAULT_LAMP_LINE_FILENAMES = (
    "ThAr_XSHOOTER_UVB_lines.dat",
    "Ne_IR_MOSFIRE_lines.dat",
    "Ar_IR_MOSFIRE_lines.dat",
)


def lamp_line_dir(zshooter_dir: str | pathlib.Path) -> pathlib.Path:
    """Return the ZShooter lamp-line directory."""
    return pathlib.Path(zshooter_dir).expanduser().resolve() / "lamp_lines"


def bin_edges_from_centers(centers: Any) -> np.ndarray:
    """Return wavelength bin edges from monotonically increasing centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError(
            "centers must be a one-dimensional array with at least two values"
        )
    mids = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.r_[first, mids, last]


def _erf(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf

        return erf(values)
    except ModuleNotFoundError:
        return np.vectorize(math.erf, otypes=[float])(values)


def gaussian_line_flux_density(
    wave_centers: Any,
    line_centers: Any,
    line_fluxes: Any,
    line_fwhm: Any,
) -> np.ndarray:
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


def read_lamp_lines(
    line_dir: str | pathlib.Path,
    filenames: Sequence[str] = DEFAULT_LAMP_LINE_FILENAMES,
) -> Table:
    """Read and combine ZShooter lamp-line tables."""
    line_dir = pathlib.Path(line_dir).expanduser().resolve()
    tables = [
        Table.read(
            line_dir / filename,
            delimiter="|",
            format="ascii",
            header_start=0,
        )
        for filename in filenames
    ]
    return vstack(tables, metadata_conflicts="silent")


def lamp_flat(
    *,
    line_dir: str | pathlib.Path | None = None,
    zshooter_dir: str | pathlib.Path | None = None,
    filenames: Sequence[str] = DEFAULT_LAMP_LINE_FILENAMES,
    wave_step: u.Quantity = 0.05 * u.AA,
    resolving_power: float = 8e4,
    extent: float = 60,
):
    """Build a calibration-line flat source with flux-preserving line sampling."""
    if line_dir is None:
        if zshooter_dir is None:
            raise ValueError("Pass either line_dir or zshooter_dir.")
        line_dir = lamp_line_dir(zshooter_dir)

    import scopesim.source.source_templates as source_templates

    lines = read_lamp_lines(line_dir, filenames=filenames)
    centers = np.asarray(lines["wave"], dtype=float)  # Angstrom
    amplitudes = np.asarray(lines["amplitude"], dtype=float)
    fwhm = centers / resolving_power

    wave_min = max(100.0, np.nanmin(centers) - 20 * np.nanmax(fwhm))
    wave_max = np.nanmax(centers) + 20 * np.nanmax(fwhm)
    step = u.Quantity(wave_step, u.AA).to_value(u.AA)
    waves = np.arange(wave_min, wave_max + step, step)
    flux_density = gaussian_line_flux_density(waves, centers, amplitudes, fwhm)

    spectrum = source_templates.SourceSpectrum(
        source_templates.Empirical1D,
        points=waves,
        lookup_table=flux_density,
    )
    return source_templates.uniform_source(spectrum, extent=extent)


def ab_magnitude_flat(mag: float = 20, extent: float = 60):
    """Return a uniform flat-field source with constant AB magnitude."""
    from scopesim.source.source_templates import ab_spectrum, uniform_source

    return uniform_source(ab_spectrum(mag=mag), extent=extent)


def constant_photon_flux_flat(
    amplitude: float = 0.001,
    extent: float = 60,
    wave_min: float = 100,
    wave_max: float = 300000,
    n_samples: int = 50000,
):
    """Return a uniform source with constant photon flux density."""
    from scopesim.source.source_templates import ConstFlux1D, Empirical1D
    from scopesim.source.source_templates import SourceSpectrum, uniform_source

    waves = np.geomspace(wave_min, wave_max, n_samples)
    spectrum_model = ConstFlux1D(amplitude=amplitude)
    spectrum = SourceSpectrum(
        Empirical1D,
        points=waves,
        lookup_table=spectrum_model(waves),
    )
    return uniform_source(spectrum, extent=extent)


def slit_frame_offsets(
    separation: u.Quantity = 1.0 * u.arcsec,
    angle_on_slit: u.Quantity = 0.0 * u.deg,
    *,
    centered: bool = True,
) -> tuple[u.Quantity, u.Quantity]:
    """Return two source offsets in the slit frame.

    The slit-frame convention follows ScopeSim source coordinates and the
    ZShooter slit files: ``x`` is along the slit length and ``y`` is across the
    slit width. ``angle_on_slit=0 deg`` therefore separates the pair along the
    slit; ``angle_on_slit=90 deg`` separates the pair across the slit.

    If ``centered`` is True, the pair is centered on ``(0, 0)``. If False,
    source 0 is placed at ``(0, 0)`` and source 1 receives the full offset.
    """
    sep = u.Quantity(separation).to(u.arcsec)
    angle = u.Quantity(angle_on_slit).to(u.rad)
    dx = sep * np.cos(angle)
    dy = sep * np.sin(angle)
    if centered:
        return (
            u.Quantity([-0.5 * dx.value, 0.5 * dx.value], dx.unit),
            u.Quantity([-0.5 * dy.value, 0.5 * dy.value], dy.unit),
        )
    return (
        u.Quantity([0.0, dx.value], dx.unit),
        u.Quantity([0.0, dy.value], dy.unit),
    )


def angle_from_cmds(
    cmds: Any,
    key: str = "!OBS.pupil_angle",
    default: u.Quantity = 0.0 * u.deg,
) -> u.Quantity:
    """Return an angle setting from a ScopeSim command object."""
    from scopesim.utils import from_currsys

    try:
        value = from_currsys(key, cmds)
    except Exception:
        value = default
    return u.Quantity(value, u.deg).to(u.deg)


def two_point_source(
    *,
    separation: u.Quantity = 1.0 * u.arcsec,
    angle_on_slit: u.Quantity = 0.0 * u.deg,
    center: tuple[u.Quantity, u.Quantity] = (0.0 * u.arcsec, 0.0 * u.arcsec),
    centered: bool = True,
    spectrum: Any | None = None,
    mag: float = 20.0,
    weights: Sequence[float] = (1.0, 1.0),
):
    """Build a two-point-source scene for slit-loss/ADC validation.

    ``angle_on_slit`` is the desired apparent pair angle in the slit frame. The
    helper itself only builds source positions; it does not inspect or apply
    instrument derotation settings.
    """
    from scopesim.source.source import Source
    from scopesim.source.source_templates import ab_spectrum

    if len(weights) != 2:
        raise ValueError("weights must contain exactly two values.")

    x_offsets, y_offsets = slit_frame_offsets(
        separation=separation,
        angle_on_slit=angle_on_slit,
        centered=centered,
    )
    x0 = u.Quantity(center[0]).to(u.arcsec)
    y0 = u.Quantity(center[1]).to(u.arcsec)
    x = x0 + x_offsets
    y = y0 + y_offsets
    spectrum = spectrum if spectrum is not None else ab_spectrum(mag=mag)

    table = Table(
        data=[
            x.to_value(u.arcsec),
            y.to_value(u.arcsec),
            np.asarray(weights, dtype=float),
            np.zeros(2, dtype=int),
            ["source_0", "source_1"],
        ],
        names=["x", "y", "weight", "ref", "label"],
        units=[u.arcsec, u.arcsec, None, None, None],
    )
    table.meta.update({
        "angle_on_slit": u.Quantity(angle_on_slit).to_value(u.deg),
        "angle_on_slit_unit": "deg",
        "separation": u.Quantity(separation).to_value(u.arcsec),
        "separation_unit": "arcsec",
        "centered": centered,
        "frame": "slit",
        "x_convention": "along slit",
        "y_convention": "across slit",
    })
    source = Source(spectra=[spectrum], table=table)
    source.meta.update({
        "function_call": "two_point_source",
        "angle_on_slit": table.meta["angle_on_slit"],
        "angle_on_slit_unit": "deg",
    })
    return source


def field_angle_demo_sources(
    *,
    along_separation: u.Quantity = 5.0 * u.arcsec,
    across_separation: u.Quantity = 0.9 * u.arcsec,
    angle_on_slit: u.Quantity = 0.0 * u.deg,
    along_mag: float = 15.0,
    across_mag: float = 17.0,
    spectrum: Any | None = None,
) -> OrderedDict[str, Any]:
    """Return the two source scenes used for field-angle/slit validation.

    ``angle_on_slit`` is the explicit apparent source-pair angle in the slit
    frame: 0 deg places the main pair along the slit, and nonzero angles move
    the pair across the slit.
    """
    angle_on_slit = u.Quantity(angle_on_slit, u.deg).to(u.deg)
    scenarios = OrderedDict([
        (
            "along_slit_centered",
            two_point_source(
                separation=along_separation,
                angle_on_slit=angle_on_slit,
                centered=True,
                mag=along_mag,
                spectrum=spectrum,
            ),
        ),
        (
            "across_slit_one_off",
            two_point_source(
                separation=across_separation,
                angle_on_slit=angle_on_slit + 90 * u.deg,
                centered=False,
                mag=across_mag,
                spectrum=spectrum,
            ),
        ),
    ])
    descriptions = {
        "along_slit_centered": (
            "Pair centered on the slit and separated along the slit."
        ),
        "across_slit_one_off": (
            "Pair separated across the slit with the second source off slit."
        ),
    }
    for name, source in scenarios.items():
        source.meta.update({
            "name": name,
            "scenario": name,
            "function_call": "field_angle_demo_sources",
            "description": descriptions[name],
            "scene_angle_on_slit": angle_on_slit.to_value(u.deg),
            "scene_angle_on_slit_unit": "deg",
            "workflow_note": (
                "Set angle_on_slit explicitly in the slit frame, then rebuild "
                "this source scene and rerun observe/readout."
            ),
        })
    return scenarios
