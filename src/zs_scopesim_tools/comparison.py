"""Boxcar forecasts and count-conserving spectral binning for instrument comparisons."""

import numpy as np
import astropy.units as u
from scipy import sparse
from scipy.special import ndtr

from scopesim.detector import Detector
from scopesim.effects import SelectorWheel, SpectralTraceList
from scopesim.effects.electronic import ExposureIntegration, DarkCurrent, BasicReadoutNoise, OnChipBinning
from scopesim.effects.electronic.pixels import BinnedImageBase
from scopesim.utils import from_currsys


def detector_rates(train):
    """Extract noiseless electron rates through the configured detector geometry and binning."""
    rates = []
    for manager in train.detector_managers:
        native, = manager
        detector = Detector(native._native_header, cmds=train.cmds)
        plane, = [plane for plane in train.image_planes if plane.id == manager._detector_list.image_plane_id]
        detector.extract_from(plane)
        for effect in train.optics_manager.detector_effects:
            selected = effect.get_effect(detector.det_id) if isinstance(effect, SelectorWheel) else effect
            if isinstance(selected, BinnedImageBase):
                selected.apply_to(detector)
        rates.append(detector.data.copy())
    return rates


def detector_noise(train, readout_index):
    """Exposure and per-output-pixel noise in summed electrons, after a real readout initializes RON tables."""
    detector, = train.detector_managers[readout_index]
    effects = []
    for effect in train.optics_manager.detector_effects:
        selected = effect.get_effect(detector.det_id) if isinstance(effect, SelectorWheel) else effect
        if selected is not None:
            effects.append(selected)
    integration, = [effect for effect in effects if isinstance(effect, ExposureIntegration)]
    dark, = [effect for effect in effects if isinstance(effect, DarkCurrent)]
    read_noise, = [effect for effect in effects if isinstance(effect, BasicReadoutNoise)]
    dit = from_currsys(integration.meta['dit'], train.cmds)
    ndit = from_currsys(integration.meta['ndit'], train.cmds)
    hardware_area = np.prod([np.prod(effect.bin_factors) for effect in effects if isinstance(effect, OnChipBinning)])
    software_area = np.prod([np.prod(effect.bin_factors) for effect in effects if isinstance(effect, BinnedImageBase) and not isinstance(effect, OnChipBinning)])
    dark_e = dark._get_dark_level(detector.det_id) * dit * ndit * hardware_area * software_area
    read_variance = read_noise.noise_std**2 * ndit * software_area
    return dit * ndit, dark_e, read_variance


def trace_pixel_grid(train, trace_id):
    """Native pixel edges and spatial centers from the train's actual trace-to-detector mapping."""
    trace_list, = train.optics_manager.get_all(SpectralTraceList)
    trace = trace_list.spectral_traces[trace_id]
    fovs = [fov for fov in train.fov_manager.fovs if str(fov.trace_id) == str(trace_id)]
    low = min(fov.meta['wave_min'].to_value(u.um) for fov in fovs)
    high = max(fov.meta['wave_max'].to_value(u.um) for fov in fovs)
    wave = np.linspace(low, high, 4097) * u.um
    coordinates = train.trace_detector_coordinates(xi=0 * u.arcsec, wavelengths={trace_id: wave})
    axis = trace.dispersion_axis
    spatial_axis = 'y' if axis == 'x' else 'x'
    dispersion = coordinates[f'detector_{axis}'].to_value(u.pixel)
    spatial = coordinates[f'detector_{spatial_axis}'].to_value(u.pixel)
    assert np.all(np.diff(dispersion) > 0) or np.all(np.diff(dispersion) < 0)
    order = np.argsort(dispersion)
    dispersion, spatial, wave = dispersion[order], spatial[order], wave[order]
    readout_index = coordinates['readout_index'][0]
    detector, = train.detector_managers[readout_index]
    size = detector.data.shape[1 if axis == 'x' else 0]
    first = max(0, int(np.ceil(dispersion[0] + .5)))
    last = min(size - 1, int(np.floor(dispersion[-1] - .5)))
    pixels = np.arange(first, last + 1)
    centers = np.interp(pixels, dispersion, spatial)
    edges = np.interp(np.arange(first, last + 2) - .5, dispersion, wave.to_value(u.um)) * u.um
    if edges[-1] < edges[0]:
        edges, pixels, centers = edges[::-1], pixels[::-1], centers[::-1]
    return readout_index, axis, pixels, centers, edges


def boxcar_sum(image, axis, pixels, centers, widths):
    """Sum integer spatial apertures without interpolating or sharing detector pixels."""
    widths = np.broadcast_to(widths, pixels.shape)
    starts = np.floor(centers - (widths - 1) / 2 + .5).astype(int)
    spatial = starts[None, :] + np.arange(widths.max())[:, None]
    included = np.arange(widths.max())[:, None] < widths
    size = image.shape[0 if axis == 'x' else 1]
    assert np.all(starts >= 0) and np.all(starts + widths <= size)
    # The unused tail of a shorter aperture must not index beyond the image.
    spatial = np.minimum(spatial, size - 1)
    values = image[spatial, pixels] if axis == 'x' else image[pixels, spatial]
    return (values * included).sum(axis=0, dtype=np.float64)


def limiting_abmag(signal, source_variance, background_variance, reference_mag=20., snr=5.):
    """Solve a^2 S^2 = snr^2 (a V_source + V_background) for positive source amplitude a."""
    q2 = snr**2
    amplitude = (q2 * source_variance + np.sqrt(q2**2 * source_variance**2 + 4 * q2 * signal**2 * background_variance)) / (2 * signal**2)
    return reference_mag - 2.5 * np.log10(amplitude)


def choose_boxcar(source, background, axis, pixels, centers, max_width, noise_per_pixel, sky_variance_factor=2., reference_mag=20., excess_noise_factor=1.):
    """Choose the smallest aperture within 0.01 mag of the best expected limit at nine wavelengths."""
    samples = np.linspace(0, len(pixels) - 1, 9).astype(int)
    limits = []
    for width in range(1, max_width + 1):
        signal = boxcar_sum(source, axis, pixels[samples], centers[samples], width)
        sky = boxcar_sum(background, axis, pixels[samples], centers[samples], width)
        variance = sky_variance_factor * (excess_noise_factor * sky + width * noise_per_pixel)
        limits.append(limiting_abmag(signal, excess_noise_factor * signal, variance, reference_mag))
    limits = np.array(limits)
    choices = np.argmax(limits >= limits.max(axis=0) - .01, axis=0) + 1
    return np.rint(np.interp(np.arange(len(pixels)), samples, choices)).astype(int)


def gaussian_bin_matrix(input_centers, output_edges, sigma):
    """Redistribute integrated counts with a Gaussian in log wavelength, integrated over output bins.

    The input samples are measured pixel-integrated counts at their wavelength centers.
    Six-sigma tails are retained without renormalizing losses beyond output boundaries.
    Propagate variance as W diag(V) W.T; adjacent output bins are correlated.
    """
    x = np.log(input_centers.to_value(u.um))
    edges = np.log(output_edges.to_value(u.um))
    sigma = np.broadcast_to(sigma, x.shape)
    if np.any(sigma <= 0):
        raise ValueError('The target resolution must be broader than the native line profile and output bin.')
    rows, columns, weights = [], [], []
    for column, (center, width) in enumerate(zip(x, sigma)):
        first = max(0, np.searchsorted(edges, center - 6 * width) - 1)
        last = min(len(edges) - 1, np.searchsorted(edges, center + 6 * width))
        bins = np.arange(first, last)
        probability = np.diff(ndtr((edges[first:last + 1] - center) / width))
        rows.extend(bins)
        columns.extend([column] * len(bins))
        weights.extend(probability)
    return sparse.csr_matrix((weights, (rows, columns)), shape=(len(edges) - 1, len(x)))


def overlap_bin_matrix(input_edges, output_edges):
    """Redistribute native integrated counts by finite wavelength-bin overlap."""
    edges = input_edges.to_value(u.um)
    target = output_edges.to_value(u.um)
    rows, columns, weights = [], [], []
    for column, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        first = max(0, np.searchsorted(target, low, side='right')-1)
        last = min(len(target)-1, np.searchsorted(target, high))
        bins = np.arange(first, last)
        overlap = np.maximum(0, np.minimum(high, target[bins+1])-np.maximum(low, target[bins])) / (high-low)
        rows.extend(bins)
        columns.extend([column]*len(bins))
        weights.extend(overlap)
    return sparse.csr_matrix((weights, (rows, columns)), shape=(len(target)-1, len(edges)-1))


def line_width(wavelength, counts):
    """Centroid, RMS, and interpolated FWHM of a nonnegative isolated line."""
    peak = np.argmax(counts)
    photons = counts.sum(dtype=np.float64)
    # Work about the line rather than its absolute wavelength: float32 image
    # sums otherwise turn tiny flux roundoff into measurable centroid shifts.
    center = wavelength[peak] + (counts * (wavelength-wavelength[peak])).sum(dtype=np.float64) / photons
    sigma = np.sqrt((counts * (wavelength-center)**2).sum(dtype=np.float64) / photons)
    left = np.flatnonzero(counts[:peak] < counts[peak] / 2)[-1]
    right = peak + np.flatnonzero(counts[peak:] < counts[peak] / 2)[0]
    half = counts[peak] / 2
    lo = wavelength[left] + (wavelength[left + 1] - wavelength[left]) * (half - counts[left]) / (counts[left + 1] - counts[left])
    hi = wavelength[right - 1] + (wavelength[right] - wavelength[right - 1]) * (half - counts[right - 1]) / (counts[right] - counts[right - 1])
    return center, sigma, hi - lo


def trace_line_profiles(train, extracted, centers, eta=0.):
    """Controlled isolated-line frames through the configured PSF, slit and trace.

    Each order is illuminated separately, allowing identical fractional FSR
    positions without accidental doublets from another arm's probe grid.
    Transmission and additive backgrounds are excluded from this line-shape
    measurement. The native detector geometry and charge binning are retained.
    """
    from synphot import SourceSpectrum, Empirical1D
    from synphot.units import PHOTLAM
    from scopesim import Source
    from scopesim.optics.fov import FieldOfView3D

    traces, = train.optics_manager.get_all(SpectralTraceList)
    psf = train['seeing_psf']
    for plane in train.image_planes:
        plane.data.fill(0)
    for original in train.fov_manager.fovs:
        wavelengths = centers[original.trace_id]
        if len(wavelengths) == 0:
            continue
        points, density = [], []
        for center in wavelengths:
            sigma = center / (200000 * np.sqrt(8*np.log(2)))
            wave = center + np.linspace(-8, 8, 161) * sigma
            points.extend(wave)
            density.extend(np.exp(-.5*((wave-center)/sigma)**2))
        order = np.argsort(points)
        spectrum = SourceSpectrum(Empirical1D, points=np.array(points)[order]*u.um, lookup_table=np.array(density)[order]*PHOTLAM, fill_value=0)
        source = Source(x=[0.], y=[eta], ref=[0], weight=[1.], spectra=[spectrum])
        low, high = min(wavelengths)*.998, max(wavelengths)*1.002
        low = max(low*u.um, original.meta['wave_min'])
        high = min(high*u.um, original.meta['wave_max'])
        metadata = original.meta | {'wave_min': low, 'wave_max': high}
        probe = FieldOfView3D(original.header.copy(), (low, high), detector_header=original.detector_header, cmds=train.cmds, **metadata)
        probe.image_plane_id = original.image_plane_id
        trace = traces.spectral_traces[original.trace_id]
        probe.meta['spectral_padding'] = trace.spectral_padding(probe)
        probe.extract_from(source, spatial_header=psf.padded_header(probe, point_sources=True))
        psf.apply_to(probe)
        probe.cube = probe.hdu
        image = trace.map_spectra_to_focal_plane(probe)
        train.image_planes[probe.image_plane_id].add(image, wcs_suffix='D')
    rates = detector_rates(train)
    profiles = {}
    for trace_id, data in extracted.items():
        profiles[trace_id] = boxcar_sum(rates[data['index']], data['axis'], data['pixels'], data['centers'], data['widths'])
    return profiles


def select_spectral_samples(signal, variance_background, reference_mag=20., excess_noise_factor=1.):
    """Hard wavelength selection for the best expected faint-continuum limit, separate from extraction.

    Ranking by S/V_background gives candidate hard cuts. Source Poisson noise is
    included when choosing the cut. This changes the spectral response inside
    the interval and must not be described as a matched Gaussian resolution.
    """
    rank = np.argsort(signal / variance_background)[::-1]
    cumulative_signal = np.cumsum(signal[rank])
    cumulative_background = np.cumsum(variance_background[rank])
    limits = limiting_abmag(cumulative_signal, excess_noise_factor*cumulative_signal, cumulative_background, reference_mag)
    count = np.argmax(limits) + 1
    return rank[:count], limits[count - 1]
