"""Inspect complete FSRs through the configured analytical trace mapping."""

import argparse
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


def fsr_footprints(spectrograph, samples=513):
    """All configured orders, including off-detector FSR endpoints and slit ends."""
    footprints = []
    for order, edges in zip(spectrograph.orders, spectrograph.edge_wave(fsr=True)):
        wave = np.linspace(edges[0], edges[1], samples)
        x, y = spectrograph._detector_pixels(wave, order)
        footprints.append((order, wave, x, y))
    return footprints


def plot_footprints(axis, footprints, row, wave_limits):
    for order, wave, x, y in footprints:
        points = np.column_stack((x[1], y[1]))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        colors = LineCollection(segments, cmap='viridis', linewidths=.8, rasterized=True)
        colors.set_array(((wave[:-1] + wave[1:])/2).to_value(u.nm))
        colors.set_clim(*wave_limits)
        axis.add_collection(colors)
        axis.plot(x[[0, 2]].T, y[[0, 2]].T, color='0.65', linewidth=.35)
        axis.scatter(x[1, [0, -1]], y[1, [0, -1]], c=wave[[0, -1]].to_value(u.nm), cmap='viridis', vmin=wave_limits[0], vmax=wave_limits[1], s=4, linewidths=0)
    axis.add_patch(plt.Rectangle((0, 0), row['n_disp'], row['n_xdisp'], fill=False, edgecolor='black', linewidth=.8))
    axis.set_aspect('equal')


def compare_geometry(parameters, baseline_parameters, baseline_echelle, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    current = DataContainer(filename=str(parameters))
    baseline = DataContainer(filename=str(baseline_parameters))
    spec = importlib.util.spec_from_file_location('scopesim.optics._baseline_echelle', baseline_echelle)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    with patch.object(GratingSetup, 'estimate_xdisp_angle_with_groove_len', staticmethod(old_module.GratingSetup.estimate_xdisp_angle_with_groove_len)):
        old_effect = EchelleSpectralTraceList(cmds=UserCommands(), filename=str(baseline_parameters), wave_colname='wavelength', s_colname='s')
    current_effect = EchelleSpectralTraceList(cmds=UserCommands(), filename=str(parameters), wave_colname='wavelength', s_colname='s')
    variants = [('baseline', baseline, old_effect), ('corrected_equation', current, current_effect)]
    fig, axes = plt.subplots(2, 6, figsize=(18, 9.5), sharex=True, sharey='col', layout='constrained')
    orders, summary = [], []
    x_low, x_high = 0, current.table['n_disp'].max()
    y_limits = [[0, row['n_xdisp']] for row in current.table]
    for variant_index, (name, params, effect) in enumerate(variants):
        for channel_index, row in enumerate(params.table):
            spectrograph = effect._spectrographs[row['prefix']]
            footprints = fsr_footprints(spectrograph)
            tilts, raw_widths = [], []
            channel_x, channel_y = [], []
            for order, wave, x, y in footprints:
                tilt = np.rad2deg(np.arctan2(y[1, -1] - y[1, 0], x[1, -1] - x[1, 0]))
                # This is the echelle's unrotated x span, not detector x or arc length.
                raw_x = spectrograph.wavelength_to_x_pixel(wave[[0, -1]], order)
                raw_width = (raw_x[1] - raw_x[0]).to_value(u.dimensionless_unscaled)
                orders.append((name, row['prefix'], order, wave[0].to_value(u.nm), wave[-1].to_value(u.nm), raw_width, tilt, x.min(), x.max(), y.min(), y.max()))
                tilts.append(tilt)
                raw_widths.append(raw_width)
                channel_x.extend([x.min(), x.max()])
                channel_y.extend([y.min(), y.max()])
            x_low, x_high = min(x_low, min(channel_x)), max(x_high, max(channel_x))
            y_limits[channel_index][0] = min(y_limits[channel_index][0], min(channel_y))
            y_limits[channel_index][1] = max(y_limits[channel_index][1], max(channel_y))
            mean_tilt = np.mean(tilts)  # Each configured order has equal weight.
            summary.append((name, row['prefix'], row['detector_angle'], len(footprints), mean_tilt, min(tilts), max(tilts), max(raw_widths), min(channel_x), max(channel_x), min(channel_y), max(channel_y)))
            axis = axes[variant_index, channel_index]
            edges = spectrograph.edge_wave(fsr=True).to_value(u.nm)
            plot_footprints(axis, footprints, row, (edges.min(), edges.max()))
            axis.set_title(f"{row['prefix'].upper()} | orders {spectrograph.orders[0]}-{spectrograph.orders[-1]}\nRotation {row['detector_angle']:+.2f} deg; mean tilt {mean_tilt:+.2f} deg", fontsize=9)
            if channel_index == 0:
                label = 'Baseline analytical model' if variant_index == 0 else 'Corrected equation, original rotations'
                axis.set_ylabel(f'{label}\nDetector y [pixel]')
            if variant_index == 1:
                axis.set_xlabel('Detector x [pixel]')
    for channel_index, limits in enumerate(y_limits):
        for axis in axes[:, channel_index]:
            axis.set(xlim=(x_low - 75, x_high + 75), ylim=(limits[0] - 75, limits[1] + 75))
    fig.suptitle('Complete analytical FSRs and slit ends; model discrepancy under investigation', fontsize=13)
    fig.savefig(output / 'detector_footprints.png', dpi=160)
    plt.close(fig)
    Table(rows=orders, names=['variant', 'channel', 'order', 'fsr_low_nm', 'fsr_high_nm', 'unrotated_fsr_width_pix', 'tilt_deg', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix']).write(output / 'orders.csv', overwrite=True)
    Table(rows=summary, names=['variant', 'channel', 'rotation_deg', 'orders', 'mean_tilt_deg', 'min_tilt_deg', 'max_tilt_deg', 'widest_unrotated_fsr_pix', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix']).write(output / 'footprints.csv', overwrite=True)
    print(f'ScopeSim import: {scopesim.__file__}\nAssessment time: {perf_counter() - start:.2f} s', flush=True)
    return variants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--baseline-parameters', type=Path, required=True)
    parser.add_argument('--baseline-echelle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare_geometry(args.parameters, args.baseline_parameters, args.baseline_echelle, args.output)
