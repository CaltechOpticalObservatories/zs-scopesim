"""Inspect complete FSRs through the configured analytical trace mapping."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy import units as u
from astropy.table import Table

from scopesim.commands import UserCommands
from scopesim.effects.data_container import DataContainer
from scopesim.effects.spectral_trace_list import EchelleSpectralTraceList
from scopesim.utils import from_currsys


SCOPESIM_MAIN = '4830d68ace8a600885bfd2e06c1f341a6766c534'
IRDB_MAIN = 'd1415c4a6bdad7d3ce0fdee58e17cd9d2424cc8d'
INSPECTION_REVISION = 'aefbe60de1e3af42529bec1ea71eb89db7eb9d9a'
LABELS = ['1. Corrected CX estimation', '2. Explicit Zemax CX angles', '3. Main']


def _git_module(repo, revision, path, name):
    """Load a committed calculation without changing any repository checkout."""
    source = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:{path}'], text=True)
    module = ModuleType(name)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
    return module


def comparison_parameters(main, archive):
    references = [archive[archive['channel'] == prefix][0] for prefix in main['prefix']]
    estimated = main.copy()
    estimated['disp_freq'] = [row['echelle_grooves_per_mm'] for row in references]
    estimated['xdisp_freq'] = [row['cx_grooves_per_mm'] for row in references]
    estimated['m0'] = [row['max_order'] for row in references]
    estimated['n'] = [row['max_order'] - row['min_order'] + 1 for row in references]
    # Zemax rotates detector axes; the analytical model rotates points.
    estimated['detector_angle'] = [-row['detector_coordinate_rotation_z_deg'] for row in references]
    estimated['xbeta_center'] = np.zeros(len(estimated))
    explicit = estimated.copy()
    # Unfold the CX tilt using the archive's signed local propagation direction.
    explicit['xbeta_center'] = [-np.sign(row['camera_space_mm'])*row['cx_coordinate_tilt_x_deg'] for row in references]
    return estimated, explicit, main.copy()


def build_comparison(scopesim_repo, irdb_repo, archive):
    main_optics = _git_module(scopesim_repo, SCOPESIM_MAIN, 'scopesim/optics/echelle.py', 'scopesim.optics._comparison_main')
    inspected_optics = _git_module(scopesim_repo, INSPECTION_REVISION, 'scopesim/optics/echelle.py', 'scopesim.optics._comparison_inspection')
    main_traces = _git_module(scopesim_repo, SCOPESIM_MAIN, 'scopesim/effects/spectral_trace_list.py', 'scopesim.effects._comparison_main')
    inspected_traces = _git_module(scopesim_repo, INSPECTION_REVISION, 'scopesim/effects/spectral_trace_list.py', 'scopesim.effects._comparison_inspection')
    main_traces.echelle = main_optics
    inspected_traces.echelle = main_optics
    source = subprocess.check_output(['git', '-C', str(irdb_repo), 'show', f'{IRDB_MAIN}:ZShooter_v2/traces/echelle_trace_parameters.dat'], text=True)
    with TemporaryDirectory(prefix='zshooter-geometry-') as directory:
        main_path = Path(directory) / 'main.dat'
        main_path.write_text(source)
        main = DataContainer(filename=str(main_path))
        reference = main_traces.EchelleSpectralTraceList(cmds=UserCommands(), filename=str(main_path), wave_colname='wavelength', s_colname='s')

        # Retain main's equations, FSR convention and placement. These two
        # inspection methods only expose the mapping used by its trace generator.
        main_optics.SpectrographSetup._rotated_detector_pixels = inspected_optics.SpectrographSetup._rotated_detector_pixels
        main_optics.SpectrographSetup._detector_pixels = inspected_optics.SpectrographSetup._detector_pixels
        original_equation = main_optics.GratingSetup.estimate_xdisp_angle_with_groove_len
        corrected_equation = inspected_optics.GratingSetup.estimate_xdisp_angle_with_groove_len
        variants = []
        for option, table in enumerate(comparison_parameters(main.table, archive), start=1):
            path = Path(directory) / f'option_{option}.dat'
            table.write(path, format='ascii.basic')
            equation = original_equation if option == 3 else corrected_equation
            with patch.object(main_optics.GratingSetup, 'estimate_xdisp_angle_with_groove_len', staticmethod(equation)):
                effect = inspected_traces.EchelleSpectralTraceList(cmds=UserCommands(), filename=str(path), wave_colname='wavelength', s_colname='s')
            variants.append((option, DataContainer(filename=str(path)), effect))
    return variants, reference


def fsr_footprints(spectrograph, samples=513):
    """All configured orders, including off-detector FSR endpoints and slit ends."""
    footprints = []
    for order, edges in zip(spectrograph.orders, spectrograph.edge_wave(fsr=True)):
        wave = np.linspace(edges[0], edges[1], samples)
        x, y = spectrograph._detector_pixels(wave, order)
        slit_nodes = [0, np.argmin(np.abs(spectrograph._slit_offset_pix)), -1]
        footprints.append((order, wave, x[slit_nodes], y[slit_nodes]))
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


def build_selected(irdb_repo):
    path = irdb_repo / 'ZShooter_v2/traces/echelle_trace_parameters.dat'
    commands = UserCommands(properties={'!SIM.file.local_packages_path': str(irdb_repo.resolve())})
    commands.update(use_instrument='ZShooter_v2', set_modes=['SPEC'])
    params = DataContainer(filename=str(path))
    params.table = from_currsys(params.table, commands)
    specification = next(effect for document in commands.yaml_dicts if 'effects' in document for effect in document['effects'] if effect['name'] == 'trace_list_analytical')
    settings = specification['kwargs'] | {'filename': str(path)}
    effect = EchelleSpectralTraceList(cmds=commands, **settings)
    return params, effect


def selected_geometry(irdb_repo, output):
    """Plot the selected prescription through the installed trace generator."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    params, effect = build_selected(irdb_repo)
    construction_seconds = perf_counter() - start
    fig, axes = plt.subplots(1, 6, figsize=(18, 6), sharex=True, layout='constrained')
    orders, summary = [], []
    x_low, x_high = 0, params.table['n_disp'].max()
    for axis, row in zip(axes, params.table, strict=True):
        model = effect._spectrographs[row['prefix']]
        footprints = fsr_footprints(model, samples=row['n_disp'] + 1)
        wave_limits = (min(wave[0].to_value(u.nm) for order, wave, x, y in footprints), max(wave[-1].to_value(u.nm) for order, wave, x, y in footprints))
        plot_footprints(axis, footprints, row, wave_limits)
        bounds, complete_orders = [], []
        for order, wave, x, y in footprints:
            inside = (x >= 0) & (x <= row['n_disp']) & (y >= 0) & (y <= row['n_xdisp'])
            complete = inside.all()
            if complete:
                complete_orders.append(order)
            tilt = np.rad2deg(np.arctan2(y[1, -1] - y[1, 0], x[1, -1] - x[1, 0]))
            orders.append((row['prefix'], order, wave[0].to_value(u.nm), wave[-1].to_value(u.nm), tilt, x.min(), x.max(), y.min(), y.max(), complete))
            bounds.append([x.min(), x.max(), y.min(), y.max()])
        bounds = np.array(bounds)
        xmin, xmax = bounds[:, 0].min(), bounds[:, 1].max()
        ymin, ymax = bounds[:, 2].min(), bounds[:, 3].max()
        summary.append((row['prefix'], len(model.orders), len(complete_orders), xmin, xmax, ymin, ymax))
        x_low, x_high = min(x_low, xmin), max(x_high, xmax)
        axis.set_ylim(min(0, ymin) - 75, max(row['n_xdisp'], ymax) + 75)
        axis.set_title(f"{row['prefix'].upper()} | orders {model.orders[0]}-{model.orders[-1]}\n{len(complete_orders)}/{len(model.orders)} full FSR + slit ends fit", fontsize=10)
        axis.set_xlabel('Detector x [pixel]')
    axes[0].set_ylabel('Detector y [pixel]')
    axes[0].set_xlim(x_low - 75, x_high + 75)
    fig.suptitle('Selected geometry: Zemax rulings and angles, fixed detector placement, 15 um pixels', fontsize=13)
    fig.savefig(output / 'selected_footprints.png', dpi=160)
    plt.close(fig)
    Table(rows=orders, names=['channel', 'order', 'fsr_low_nm', 'fsr_high_nm', 'tilt_deg', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix', 'full_fsr_and_slit_on_detector']).write(output / 'selected_orders.csv', overwrite=True)
    Table(rows=summary, names=['channel', 'orders', 'complete_orders', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix']).write(output / 'selected_footprints.csv', overwrite=True)
    print(f'Selected trace construction: {construction_seconds:.3f} s; {len(effect.spectral_traces)} orders. Figure and tables: {perf_counter() - start:.2f} s total.', flush=True)
    return params, effect, fig


def compare_geometry(scopesim_repo, irdb_repo, zemax_parameters, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    archive = Table.read(zemax_parameters)
    variants, reference = build_comparison(scopesim_repo, irdb_repo, archive)
    for actual, expected in zip(variants[2][2].data_container._file[1:], reference.data_container._file[1:], strict=True):
        assert actual.name == expected.name
        np.testing.assert_array_equal(actual.data, expected.data)
    print(f'Option 3 matches all {len(reference.spectral_traces)} unmodified-main trace tables exactly.', flush=True)

    all_footprints = []
    for option, params, effect in variants:
        all_footprints.append([fsr_footprints(effect._spectrographs[row['prefix']], samples=row['n_disp'] + 1) for row in params.table])
    wave_limits = []
    for channel_index in range(6):
        endpoints = [wave[[0, -1]].to_value(u.nm) for footprints in all_footprints for order, wave, x, y in footprints[channel_index]]
        wave_limits.append((min(wave[0] for wave in endpoints), max(wave[1] for wave in endpoints)))

    fig, axes = plt.subplots(3, 6, figsize=(18, 14), sharex=True, sharey='col', layout='constrained')
    orders, summary, settings = [], [], []
    x_low, x_high = 0, variants[2][1].table['n_disp'].max()
    y_limits = [[0, row['n_xdisp']] for row in variants[2][1].table]
    for variant_index, (option, params, effect) in enumerate(variants):
        for channel_index, row in enumerate(params.table):
            spectrograph = effect._spectrographs[row['prefix']]
            footprints = all_footprints[variant_index][channel_index]
            settings.append((option, row['prefix'], spectrograph.orders[0], spectrograph.orders[-1], row['disp_freq'], row['xdisp_freq'], spectrograph.cross_disperser.alpha.to_value(u.deg), spectrograph.cross_disperser.beta_center.to_value(u.deg), row['detector_angle']))
            tilts, raw_widths = [], []
            channel_x, channel_y = [], []
            for order, wave, x, y in footprints:
                tilt = np.rad2deg(np.arctan2(y[1, -1] - y[1, 0], x[1, -1] - x[1, 0]))
                # This is the echelle's unrotated x span, not detector x or arc length.
                raw_x = spectrograph.wavelength_to_x_pixel(wave[[0, -1]], order)
                raw_width = (raw_x[1] - raw_x[0]).to_value(u.dimensionless_unscaled)
                orders.append((option, row['prefix'], order, wave[0].to_value(u.nm), wave[-1].to_value(u.nm), raw_width, tilt, x.min(), x.max(), y.min(), y.max()))
                tilts.append(tilt)
                raw_widths.append(raw_width)
                channel_x.extend([x.min(), x.max()])
                channel_y.extend([y.min(), y.max()])
            x_low, x_high = min(x_low, min(channel_x)), max(x_high, max(channel_x))
            y_limits[channel_index][0] = min(y_limits[channel_index][0], min(channel_y))
            y_limits[channel_index][1] = max(y_limits[channel_index][1], max(channel_y))
            mean_tilt = np.mean(tilts)  # Each configured order has equal weight.
            pixel_mm = (row['pixel_size'] * u.Unit(params.meta['pixel_size_unit'])).to_value(u.mm)
            margins = np.array([min(channel_x), row['n_disp'] - max(channel_x), min(channel_y), row['n_xdisp'] - max(channel_y)]) * pixel_mm
            summary.append((option, row['prefix'], row['detector_angle'], len(footprints), mean_tilt, min(tilts), max(tilts), max(raw_widths), min(channel_x), max(channel_x), min(channel_y), max(channel_y), *margins))
            axis = axes[variant_index, channel_index]
            plot_footprints(axis, footprints, row, wave_limits[channel_index])
            axis.set_title(f"{row['prefix'].upper()} | orders {spectrograph.orders[0]}-{spectrograph.orders[-1]}\nRotation {row['detector_angle']:+.2f} deg; mean tilt {mean_tilt:+.2f} deg", fontsize=9)
            if channel_index == 0:
                axis.set_ylabel(f'{LABELS[variant_index]}\nDetector y [pixel]')
            if variant_index == 2:
                axis.set_xlabel('Detector x [pixel]')
    for channel_index, limits in enumerate(y_limits):
        for axis in axes[:, channel_index]:
            axis.set(xlim=(x_low - 75, x_high + 75), ylim=(limits[0] - 75, limits[1] + 75))
    fig.suptitle('Three analytical options: complete FSRs and slit ends; main detector placement and 15 um pixels', fontsize=13)
    fig.savefig(output / 'detector_footprints.png', dpi=160)
    plt.close(fig)
    Table(rows=orders, names=['option', 'channel', 'order', 'fsr_low_nm', 'fsr_high_nm', 'unrotated_fsr_width_pix', 'tilt_deg', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix']).write(output / 'orders.csv', overwrite=True)
    Table(rows=summary, names=['option', 'channel', 'rotation_deg', 'orders', 'mean_tilt_deg', 'min_tilt_deg', 'max_tilt_deg', 'widest_unrotated_fsr_pix', 'x_min_pix', 'x_max_pix', 'y_min_pix', 'y_max_pix', 'left_margin_mm', 'right_margin_mm', 'bottom_margin_mm', 'top_margin_mm']).write(output / 'footprints.csv', overwrite=True)
    Table(rows=settings, names=['option', 'channel', 'min_order', 'max_order', 'echelle_grooves_per_mm', 'cx_grooves_per_mm', 'cx_incidence_deg', 'cx_camera_angle_deg', 'detector_rotation_deg']).write(output / 'settings.csv', overwrite=True)
    counts = [sum(len(channel) for channel in footprints) for footprints in all_footprints]
    print(f'Orders plotted in options 1/2/3: {counts}. Assessment time: {perf_counter() - start:.2f} s.', flush=True)
    return variants, fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scopesim', type=Path, required=True)
    parser.add_argument('--irdb', type=Path, required=True)
    parser.add_argument('--zemax-parameters', type=Path, default=Path(__file__).with_name('data') / 'zemax_channels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selected', action='store_true', help='Plot the current configured prescription instead of the historical three-option comparison')
    args = parser.parse_args()
    if args.selected:
        selected_geometry(args.irdb, args.output)
    else:
        compare_geometry(args.scopesim, args.irdb, args.zemax_parameters, args.output)
