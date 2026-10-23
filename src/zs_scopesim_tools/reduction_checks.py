"""Finite-bin checks of flux-calibrated, covariance-aware extracted spectra."""

import numpy as np
import astropy.units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
from pyreduce.spectra import Spectra


def flat_ab_band_integrals(files, *, intervals, max_gap, magnitude):
    """Measure flux closure in selected comparison windows with full covariance.

    Each arm gets one width-weighted flux-density mean over its explicit,
    non-overlapping wavelength intervals. Missing coverage contributes no
    width, while covariance between separated retained regions is preserved.
    Units and native pixel boundaries follow ``flat_ab_extraction_bins``.
    """
    flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
    rows = []
    for arm, filename in files.items():
        product = Spectra.read(filename, raw=True)
        spectrum, = product.data
        if spectrum.covariance is None:
            raise ValueError(f'{arm} has no propagated spectral covariance')
        flux = (spectrum.spec * u.Unit(product.header['BUNIT'])).to(flux_unit)
        covariance_scale = (1 * u.Unit(product.header['COVUNIT'])).to_value(flux_unit**2)
        wave = spectrum.wave * u.AA
        valid = np.isfinite(wave) & np.isfinite(spectrum.spec) & np.isfinite(spectrum.covariance.diagonal())
        indices = np.flatnonzero(valid)
        if np.any(np.diff(wave[indices]) <= 0 * u.AA):
            raise ValueError(f'{arm} wavelength samples must increase strictly')
        windows = [(lower.to_value(u.AA), upper.to_value(u.AA)) for lower, upper in intervals[arm]]
        for i, (lower, upper) in enumerate(windows):
            if upper <= lower or (i > 0 and lower < windows[i-1][1]):
                raise ValueError('Retained intervals must have positive width and be ordered without overlap')
        widths = np.zeros(len(wave))
        gaps = (np.diff(indices) != 1) | (np.diff(wave[indices]) > max_gap[arm])
        for index in np.split(indices, np.flatnonzero(gaps) + 1):
            if len(index) < 2:
                raise ValueError(f'{arm} has isolated samples without defined native pixel widths')
            centers = wave[index].to_value(u.AA)
            edges = np.r_[centers[0] - (centers[1]-centers[0])/2, (centers[:-1]+centers[1:])/2, centers[-1] + (centers[-1]-centers[-2])/2]
            for lower, upper in windows:
                widths[index] += np.maximum(0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
        selected = np.flatnonzero(widths > 0)
        if not len(selected):
            raise ValueError(f'{arm} has no samples in the retained wavelength intervals')
        covered_width = widths.sum() * u.AA
        weights = widths[selected] / widths.sum()
        covariance = spectrum.covariance[selected][:, selected]
        measured = np.sum(weights * flux[selected])
        uncertainty = np.sqrt((weights @ (covariance @ weights)) * covariance_scale) * flux_unit
        expected = (magnitude * u.ABmag).to(flux_unit, equivalencies=u.spectral_density(wave[selected]))
        input_mean = np.sum(weights * expected)
        wavelength = np.sum(weights * wave[selected])
        rows.append((arm, wavelength, covered_width, measured, uncertainty, input_mean, measured/input_mean, uncertainty/input_mean, measured/uncertainty))
    return QTable(rows=rows, names=('arm', 'wavelength', 'covered_width', 'flux', 'uncertainty', 'expected_flux', 'ratio', 'ratio_uncertainty', 'snr'))


def flat_ab_extraction_bins(files, *, intervals, max_gap, magnitude, bin_R):
    """Measure flat-AB recovery and S/N in finite wavelength integration bins.

    ``files`` maps arm names to flux-calibrated FITS files: one spliced spectrum
    per arm, with flux density and full sparse covariance. Required ``BUNIT``
    and ``COVUNIT`` headers supply units convertible to FLAM and FLAM squared.
    ``intervals`` contains explicitly retained wavelength intervals per arm,
    and ``max_gap`` gives that arm's largest allowed adjacent native-sample gap;
    both require wavelength Quantities. Select comparison windows explicitly,
    describing retained telluric or response structure and any exclusions;
    do not select on noisy recovered flux. Gaps and masked samples split coverage. Native pixel
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
        product = Spectra.read(filename, raw=True)
        spectrum, = product.data
        if spectrum.covariance is None:
            raise ValueError(f'{arm} has no propagated spectral covariance')
        native_flux = (spectrum.spec * u.Unit(product.header['BUNIT'])).to(flux_unit)
        covariance_scale = (1 * u.Unit(product.header['COVUNIT'])).to_value(flux_unit**2)
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
                    flux = np.sum(weights * native_flux[sample])
                    uncertainty = np.sqrt((weights @ (covariance @ weights)) * covariance_scale) * flux_unit
                    input_flux = np.sum(weights * expected[sample])
                    wavelength = np.sum(weights * wave[sample])
                    rows.append((arm, wavelength, low*u.AA, high*u.AA, width*u.AA, flux, uncertainty, input_flux, flux/input_flux, flux/uncertainty))
    table = QTable(rows=rows, names=('arm', 'wavelength', 'wave_low', 'wave_high', 'covered_width', 'flux', 'uncertainty', 'expected_flux', 'ratio', 'snr'))
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for arm in files:
        part = table[table['arm'] == arm]
        # Bin midpoints and horizontal extents show actual supported widths;
        # the table separately retains the width-weighted native wavelength.
        center = (part['wave_low'] + part['wave_high']) / 2
        half_width = part['covered_width'] / 2
        axes[0].errorbar(center.to_value(u.um), part['ratio'].value, xerr=half_width.to_value(u.um), yerr=(part['uncertainty']/part['expected_flux']).value, fmt='.', ms=2, elinewidth=.4, label=arm)
        axes[1].errorbar(center.to_value(u.um), part['snr'].value, xerr=half_width.to_value(u.um), fmt='.', ms=2, elinewidth=.4)
    axes[0].axhline(1, color='0.4', lw=.7)
    axes[0].set(ylabel='Recovered / input flux', title=rf'Flat AB {magnitude:g}: $\lambda/\Delta\lambda\approx{bin_R:g}$ integration bins')
    axes[0].legend()
    axes[1].set(xlabel=r'Wavelength [$\mu$m]', ylabel='S/N in integration bin')
    return table, figure
