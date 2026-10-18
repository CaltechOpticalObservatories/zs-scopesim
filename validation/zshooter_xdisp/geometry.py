"""Reproduce the rejected coverage-only rotation trial; not a prescription optimizer."""

import argparse
import copy
import importlib.util
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy import units as u
from astropy.table import Table

import scopesim
from scopesim.commands import UserCommands
from scopesim.effects.data_container import DataContainer
from scopesim.effects.spectral_trace_list import EchelleSpectralTraceList
from scopesim.optics.echelle import GratingSetup


def trace_intervals(hdu, row, full_slit):
    """Wavelength intervals whose center or both slit edges lie on the detector.

    Intersect each segment of the generated trace with the detector rectangle.
    All three slit nodes share the same wavelength parameter. Requiring all
    nodes inside also includes the intervening slit, since its mapping is affine.
    """
    wave = (hdu.data['wavelength'].reshape(3, -1)[1] * u.Unit(hdu.columns['wavelength'].unit)).to_value(u.nm)
    entering, leaving = np.zeros(wave.size - 1), np.ones(wave.size - 1)
    for name, count in [('x', row['n_disp']), ('y', row['n_xdisp'])]:
        coordinate = (hdu.data[name].reshape(3, -1) * u.Unit(hdu.columns[name].unit)).to_value(u.mm)
        if not full_slit:
            coordinate = coordinate[1:2]
        half_size = count * row['pixel_size'] / 2
        start, step = coordinate[:, :-1], np.diff(coordinate, axis=1)
        parallel = step == 0
        first = np.full(step.shape, -np.inf)
        last = np.full(step.shape, np.inf)
        np.divide(-half_size - start, step, out=first, where=~parallel)
        np.divide(half_size - start, step, out=last, where=~parallel)
        lower, upper = np.minimum(first, last), np.maximum(first, last)
        outside = parallel & (np.abs(start) > half_size)
        lower[outside], upper[outside] = np.inf, -np.inf
        entering = np.maximum(entering, lower.max(axis=0))
        leaving = np.minimum(leaving, upper.min(axis=0))
    valid = entering < leaving
    low = np.maximum(wave[:-1][valid] + entering[valid] * np.diff(wave)[valid], row['min_wave'])
    high = np.minimum(wave[:-1][valid] + leaving[valid] * np.diff(wave)[valid], row['max_wave'])
    return list(zip(low[low < high], high[low < high]))


def band_coverage(hdul, row, full_slit=True):
    intervals = []
    for hdu in hdul[2:]:
        if hdu.name.startswith(row['prefix'] + '_'):
            intervals.extend(trace_intervals(hdu, row, full_slit))
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    covered = sum(high - low for low, high in merged)
    gaps, end = [], row['min_wave']
    for low, high in merged:
        if low > end:
            gaps.append((end, low))
        end = high
    if end < row['max_wave']:
        gaps.append((end, row['max_wave']))
    return covered / (row['max_wave'] - row['min_wave']), gaps


def minimum_vertical_slit_gap(orders, row):
    """Smallest gap between neighboring slit edges at a shared detector x."""
    gaps = []
    for first, second in zip(orders[:-1], orders[1:]):
        edges = []
        for hdu in [first, second]:
            x = hdu.data['x'].reshape(3, -1)[[0, 2]] / row['pixel_size'] + row['n_disp']/2
            y = hdu.data['y'].reshape(3, -1)[[0, 2]] / row['pixel_size'] + row['n_xdisp']/2
            edges.extend(zip(x, y))
        low = max(0, max(x[0] for x, y in edges))
        high = min(row['n_disp'], min(x[-1] for x, y in edges))
        if low >= high:
            continue
        x_grid = np.unique(np.r_[low, high, *(x[(x > low) & (x < high)] for x, y in edges)])
        y_grid = np.array([np.interp(x_grid, x, y) for x, y in edges])
        first_low, first_high = y_grid[:2].min(axis=0), y_grid[:2].max(axis=0)
        second_low, second_high = y_grid[2:].min(axis=0), y_grid[2:].max(axis=0)
        on_detector = (first_high >= 0) & (first_low <= row['n_xdisp']) & (second_high >= 0) & (second_low <= row['n_xdisp'])
        if on_detector.any():
            gaps.append(np.maximum(first_low - second_high, second_low - first_high)[on_detector].min())
    return min(gaps)


def retune_rotations(effect, parameters):
    """Rejected coverage-only objective, retained to reproduce and inspect the trial."""
    tuned = copy.copy(parameters)
    tuned.table = parameters.table.copy()
    tuned.table['detector_angle'] = tuned.table['detector_angle'].astype(float)
    for index, row in enumerate(parameters.table):
        candidate = copy.copy(tuned)
        candidate.table = tuned.table[index:index + 1].copy()
        old_angle = row['detector_angle']
        scores = {}

        def score(angle):
            if angle not in scores:
                candidate.table['detector_angle'][0] = angle
                hdul = effect._generate_trace_hdulist(candidate)
                scores[angle] = band_coverage(hdul, candidate.table[0])[0]
            return scores[angle]

        for angle in np.unique(np.r_[np.arange(-30., 30.01, 1.), old_angle]):
            score(angle)
        best = max(scores, key=scores.__getitem__)
        if abs(best) == 30:
            raise ValueError(f"{row['prefix']}: best rotation lies at the search boundary; inspect the footprint.")
        for angle in np.round(np.arange(best - 1, best + 1.001, .01), 2):
            score(angle)
        maximum = max(scores.values())
        # The tolerance is numerical interval-integration precision, not optical accuracy.
        tied = [angle for angle, value in scores.items() if maximum - value < 1e-9]
        angle = min(tied, key=lambda value: abs(value - old_angle))
        tuned.table['detector_angle'][index] = angle
        print(f"{row['prefix']}: {old_angle:+.2f} -> {angle:+.2f} deg; full-slit coverage {score(old_angle):.5f} -> {score(angle):.5f}", flush=True)
    return tuned


def compare_geometry(parameters, baseline_parameters, baseline_echelle, output, rotations=None):
    """Evaluate the saved baseline, corrected prescription and retuned placement."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = DataContainer(filename=str(parameters))
    baseline = DataContainer(filename=str(baseline_parameters))
    current.table['detector_angle'] = baseline.table['detector_angle'].copy()
    effect = EchelleSpectralTraceList(cmds=UserCommands(), filename=str(parameters), wave_colname='wavelength', s_colname='s')
    spec = importlib.util.spec_from_file_location('scopesim.optics._baseline_echelle', baseline_echelle)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    with patch.object(GratingSetup, 'estimate_xdisp_angle_with_groove_len', staticmethod(old_module.GratingSetup.estimate_xdisp_angle_with_groove_len)):
        old_hdul = effect._generate_trace_hdulist(baseline)
    corrected_hdul = effect._generate_trace_hdulist(current)
    start = perf_counter()
    if rotations is None:
        tuned = retune_rotations(effect, current)
    else:
        angles = Table.read(rotations)
        tuned = copy.copy(current)
        tuned.table = current.table.copy()
        tuned.table['detector_angle'] = [angles['detector_angle'][angles['prefix'] == row['prefix']][0] for row in current.table]
    tuning_seconds = perf_counter() - start
    tuned_hdul = effect._generate_trace_hdulist(tuned)
    variants = [('baseline', baseline, old_hdul), ('corrected', current, corrected_hdul), ('retuned', tuned, tuned_hdul)]
    rows, gaps, sampling = [], [], []
    fig, axes = plt.subplots(3, 6, figsize=(17, 14), sharex='col', sharey='col', layout='constrained')
    for variant_index, (name, params, hdul) in enumerate(variants):
        hdul.writeto(output / f'{name}_traces.fits', overwrite=True)
        for channel_index, row in enumerate(params.table):
            full, missing = band_coverage(hdul, row)
            center, _ = band_coverage(hdul, row, full_slit=False)
            orders = [hdu for hdu in hdul[2:] if hdu.name.startswith(row['prefix'] + '_')]
            slit_gap = minimum_vertical_slit_gap(orders, row)
            rows.append((name, row['prefix'], row['xdisp_freq'], row['detector_angle'], len(orders), center, full, slit_gap))
            gaps.extend((name, row['prefix'], low, high) for low, high in missing)
            axis = axes[variant_index, channel_index]
            for hdu in orders:
                x = hdu.data['x'].reshape(3, -1) / row['pixel_size'] + row['n_disp'] / 2
                y = hdu.data['y'].reshape(3, -1) / row['pixel_size'] + row['n_xdisp'] / 2
                wave = hdu.data['wavelength'].reshape(3, -1)[1] * 1000
                # A common wavelength for each trace, independent of endpoint clipping.
                old_trace = old_hdul[hdu.name].data
                old_wave = old_trace['wavelength'].reshape(3, -1)[1] * 1000
                low = max(wave[0], old_wave[0], row['min_wave'])
                high = min(wave[-1], old_wave[-1], row['max_wave'])
                at_wave = (max(old_wave[0], row['min_wave']) + min(old_wave[-1], row['max_wave']))/2
                if low <= at_wave <= high:
                    dx = np.interp(at_wave, wave, np.gradient(x[1], wave))
                    dy = np.interp(at_wave, wave, np.gradient(y[1], wave))
                    sampling.append((name, row['prefix'], hdu.name, at_wave, 1/np.hypot(dx, dy), 1/abs(dx)))
                points = np.column_stack((x[1], y[1]))
                segments = np.stack((points[:-1], points[1:]), axis=1)
                colors = LineCollection(segments, cmap='viridis', linewidths=.8, rasterized=True)
                colors.set_array((wave[:-1] + wave[1:]) / 2)
                colors.set_clim(row['min_wave'], row['max_wave'])
                axis.add_collection(colors)
                axis.plot(x[[0, 2]].T, y[[0, 2]].T, color='0.65', linewidth=.35)
            axis.add_patch(plt.Rectangle((0, 0), row['n_disp'], row['n_xdisp'], fill=False, edgecolor='black', linewidth=.8))
            axis.set(xlim=(-100, row['n_disp'] + 100), ylim=(-100, row['n_xdisp'] + 100), aspect='equal')
            axis.set_title(f"{row['prefix'].upper()} | {row['detector_angle']:+.2f} deg\nFull slit: {100*full:.2f}%", fontsize=10)
            if channel_index == 0:
                axis.set_ylabel(f'{name}\nDetector y [pixel]')
            if variant_index == 2:
                axis.set_xlabel('Detector x [pixel]')
    fig.savefig(output / 'detector_footprints.png', dpi=160)
    plt.close(fig)
    Table(rows=rows, names=['variant', 'channel', 'grooves_per_mm', 'rotation_deg', 'orders', 'center_coverage', 'full_slit_coverage', 'minimum_vertical_slit_gap_pixels']).write(output / 'coverage.csv', overwrite=True)
    Table(rows=gaps, names=['variant', 'channel', 'missing_low_nm', 'missing_high_nm']).write(output / 'gaps.csv', overwrite=True)
    Table(rows=sampling, names=['variant', 'channel', 'trace', 'wavelength_nm', 'along_trace_nm_per_pixel', 'detector_x_nm_per_pixel']).write(output / 'sampling.csv', overwrite=True)
    tuned.table[['prefix', 'detector_angle']].write(output / 'rotations.csv', overwrite=True)
    print(f'ScopeSim import: {scopesim.__file__}\nRotation preparation: {tuning_seconds:.2f} s', flush=True)
    return variants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--baseline-parameters', type=Path, required=True)
    parser.add_argument('--baseline-echelle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rotations', type=Path, help='Reuse explicit saved rotations instead of repeating the search.')
    args = parser.parse_args()
    compare_geometry(args.parameters, args.baseline_parameters, args.baseline_echelle, args.output, args.rotations)
