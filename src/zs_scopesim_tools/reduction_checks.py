"""Finite-bin checks of flux-calibrated, covariance-aware extracted spectra."""

import numpy as np
import astropy.units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
from pyreduce.spectra import Spectra


def flat_ab_extraction_bins(files, *, intervals, max_gap, magnitude, bin_R):
    """Measure flat-AB recovery and S/N in finite wavelength integration bins.

    ``files`` maps arm names to flux-calibrated FITS files: one spliced spectrum
    per arm, with flux density in erg/s/cm2/Angstrom and full sparse covariance.
    ``intervals`` contains explicitly retained wavelength intervals per arm,
    and ``max_gap`` gives that arm's largest allowed adjacent native-sample gap;
    both require wavelength Quantities. Choose intervals to exclude specified
    telluric bands or poorly calibrated response, rather than selecting on the
    noisy recovered flux. Gaps and masked samples split coverage. Native pixel
    edges are adjacent-center midpoints, with half-step outer edges.

    Width-weighted means use only supported wavelengths. The table reports the
    actual covered width, including partial edge bins. ``bin_R`` sets logarithmic
    integration-bin spacing, not an LSF or a claim of achieved resolving power.
    The input flat-AB flux is evaluated at the same native sample centers and
    averaged with the same weights as the extracted flux.
    """
    if bin_R <= 0:
        raise ValueError('bin_R must be positive')
    flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
    rows = []
    for arm, filename in files.items():
        spectrum, = Spectra.read(filename, raw=True).data
        if spectrum.covariance is None:
            raise ValueError(f'{arm} has no propagated spectral covariance')
        wave = spectrum.wave * u.AA
        expected = (magnitude * u.ABmag).to(flux_unit, equivalencies=u.spectral_density(wave))
        valid = np.isfinite(wave) & np.isfinite(spectrum.spec) & np.isfinite(spectrum.covariance.diagonal())
        indices = np.flatnonzero(valid)
        if np.any(np.diff(wave[indices]) <= 0 * u.AA):
            raise ValueError(f'{arm} wavelength samples must increase strictly')
        gaps = (np.diff(indices) != 1) | (np.diff(wave[indices]) > max_gap[arm])
        segments = np.split(indices, np.flatnonzero(gaps) + 1)
        for index in segments:
            if len(index) < 2:
                raise ValueError(f'{arm} has isolated samples without defined native pixel widths')
            centers = wave[index].to_value(u.AA)
            edges = np.r_[centers[0] - (centers[1]-centers[0])/2, (centers[:-1]+centers[1:])/2, centers[-1] + (centers[-1]-centers[-2])/2]
            for lower, upper in intervals[arm]:
                lower, upper = lower.to_value(u.AA), upper.to_value(u.AA)
                if upper <= lower:
                    raise ValueError('Retained wavelength intervals must have positive width')
                start, stop = max(edges[0], lower), min(edges[-1], upper)
                if stop <= start:
                    continue
                bin_edges = start * np.exp(np.arange(np.ceil(np.log(stop/start)*bin_R)) / bin_R)
                bin_edges = np.r_[bin_edges, stop]
                for low, high in zip(bin_edges[:-1], bin_edges[1:]):
                    overlap = np.maximum(0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))
                    selected = overlap > 0
                    sample = index[selected]
                    width = overlap[selected].sum()
                    weights = overlap[selected] / width
                    covariance = spectrum.covariance[sample][:, sample]
                    flux = weights @ spectrum.spec[sample] * flux_unit
                    uncertainty = np.sqrt(weights @ (covariance @ weights)) * flux_unit
                    input_flux = np.sum(weights * expected[sample])
                    wavelength = np.sum(weights * wave[sample])
                    rows.append((arm, wavelength, low*u.AA, high*u.AA, width*u.AA, flux, uncertainty, input_flux, flux/input_flux, flux/uncertainty))
    table = QTable(rows=rows, names=('arm', 'wavelength', 'wave_low', 'wave_high', 'covered_width', 'flux', 'uncertainty', 'expected_flux', 'ratio', 'snr'))
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for arm in files:
        part = table[table['arm'] == arm]
        # Bin midpoints and horizontal extents show actual supported widths;
        # the table separately retains the flux-weighted native wavelength.
        center = (part['wave_low'] + part['wave_high']) / 2
        half_width = part['covered_width'] / 2
        axes[0].errorbar(center.to_value(u.um), part['ratio'].value, xerr=half_width.to_value(u.um), yerr=(part['uncertainty']/part['expected_flux']).value, fmt='.', ms=2, elinewidth=.4, label=arm)
        axes[1].errorbar(center.to_value(u.um), part['snr'].value, xerr=half_width.to_value(u.um), fmt='.', ms=2, elinewidth=.4)
    axes[0].axhline(1, color='0.4', lw=.7)
    axes[0].set(ylabel='Recovered / input flux', title=rf'Flat AB {magnitude:g}: $\lambda/\Delta\lambda\approx{bin_R:g}$ integration bins')
    axes[0].legend()
    axes[1].set(xlabel=r'Wavelength [$\mu$m]', ylabel='S/N in integration bin')
    return table, figure
