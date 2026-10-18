"""FSR plots retain the actual model's complete wavelength and slit domain."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy import units as u
from astropy.table import Table

from scopesim.optics.echelle import spectrograph_factory
from scopesim.optics import echelle
from scopesim.effects.spectral_trace_list import EchelleSpectralTraceList
from scopesim.commands import UserCommands
from validation.zshooter_xdisp import geometry
from validation.zshooter_xdisp.geometry import fsr_footprints, plot_footprints


ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / 'validation/zshooter_xdisp/data/zemax_channels.csv'


@pytest.fixture(scope='module')
def comparison():
    return geometry.build_comparison(ROOT.parent / 'ScopeSim', ROOT.parent / 'irdb', Table.read(ARCHIVE))


@pytest.fixture(scope='module')
def selected():
    return geometry.build_selected(ROOT.parent / 'irdb')


@pytest.mark.parametrize('option', [2, 3])
def test_absent_placement_columns_reproduce_original_trace_tables(comparison, tmp_path, option):
    variants, reference = comparison
    number, params, original = variants[option - 1]
    path = tmp_path / 'original.dat'
    params.table.write(path, format='ascii.basic')
    original_equation = type(original._spectrographs['b'].cross_disperser).estimate_xdisp_angle_with_groove_len
    with patch.object(echelle.GratingSetup, 'estimate_xdisp_angle_with_groove_len', staticmethod(original_equation)):
        actual = EchelleSpectralTraceList(cmds=UserCommands(), filename=str(path), wave_colname='wavelength', s_colname='s')
    for actual_hdu, expected_hdu in zip(actual.data_container._file[1:], original.data_container._file[1:], strict=True):
        np.testing.assert_array_equal(actual_hdu.data, expected_hdu.data)


def test_selected_changes_only_placement_and_yj_order_57(selected, comparison):
    params, effect = selected
    option, original_params, original = comparison[0][1]
    expected_shifts = {'b': (0, 0), 'g': (0, 0), 'r': (-150, 0), 'yj': (0, 50), 'h': (-130, 0), 'k': (0, 0)}
    for column in original_params.table.colnames:
        if column != 'n':
            np.testing.assert_array_equal(params.table[column], original_params.table[column])
    np.testing.assert_array_equal(params.table['n'], [34, 30, 20, 23, 10, 8])
    for axis in ['x', 'y']:
        assert u.Unit(params.meta[f'detector_offset_{axis}_unit']) == u.mm
    for row in params.table:
        prefix = row['prefix']
        model, previous = effect._spectrographs[prefix], original._spectrographs[prefix]
        dx, dy = expected_shifts[prefix]
        for order, wave, x, y in fsr_footprints(model, samples=row['n_disp'] + 1):
            old_x, old_y = previous._detector_pixels(wave, order)
            # Six decimals in mm are much finer than the 0.001-pixel tolerance.
            np.testing.assert_allclose(x - old_x, dx, atol=.001, rtol=0)
            np.testing.assert_allclose(y - old_y, dy, atol=.001, rtol=0)
            np.testing.assert_allclose(np.diff(x), np.diff(old_x), atol=1e-9, rtol=0)
            np.testing.assert_allclose(np.diff(y), np.diff(old_y), atol=1e-9, rtol=0)
            np.testing.assert_allclose(x[2] - x[0], old_x[2] - old_x[0], atol=1e-9, rtol=0)
            np.testing.assert_allclose(y[2] - y[0], old_y[2] - old_y[0], atol=1e-9, rtol=0)
            edges = previous.edge_wave(fsr=True)[previous.orders == order][0]
            np.testing.assert_array_equal(wave[[0, -1]], edges)
    assert 'yj_57' not in effect.spectral_traces
    assert all(f'r_{order}' in effect.spectral_traces for order in [37, 38, 39])


def test_selected_figure_uses_every_configured_fsr_and_both_slit_ends(selected, monkeypatch, tmp_path):
    monkeypatch.setattr(geometry, 'build_selected', lambda irdb_repo: selected)
    params, effect, fig = geometry.selected_geometry(ROOT.parent / 'irdb', tmp_path)
    assert len(fig.axes) == 6
    for axis, row in zip(fig.axes, params.table, strict=True):
        centerlines = [collection for collection in axis.collections if isinstance(collection, LineCollection)]
        assert len(centerlines) == row['n']
        assert len(axis.lines) == 2*row['n']
        assert all(len(trace.get_segments()) == row['n_disp'] for trace in centerlines)
        assert axis.get_aspect() == 1
        assert axis.get_xlim() == fig.axes[0].get_xlim()
    orders = Table.read(tmp_path / 'selected_orders.csv')
    assert len(orders) == 125
    for channel, expected in [('r', np.arange(40, 57)), ('h', np.arange(42, 52)), ('yj', np.arange(58, 81))]:
        complete = orders[(orders['channel'] == channel) & (orders['full_fsr_and_slit_on_detector'] == 'True')]
        np.testing.assert_array_equal(complete['order'], expected)
    assert (tmp_path / 'selected_footprints.png').is_file()


def test_main_option_reproduces_unmodified_main_trace_tables(comparison):
    variants, reference = comparison
    assert [len(effect.spectral_traces) for option, params, effect in variants] == [126, 126, 127]
    actual = variants[2][2].data_container._file
    expected = reference.data_container._file
    for actual_hdu, expected_hdu in zip(actual[1:], expected[1:], strict=True):
        assert actual_hdu.name == expected_hdu.name
        np.testing.assert_array_equal(actual_hdu.data, expected_hdu.data)
    main = variants[2][1].table
    assert main[main['prefix'] == 'yj']['xdisp_freq'][0] == 325
    h = main[main['prefix'] == 'h'][0]
    assert h['m0'] - h['n'] + 1 == 41


def test_zemax_options_differ_only_in_cx_angle_selection(comparison):
    variants, reference = comparison
    estimated, explicit, main = [params.table for option, params, effect in variants]
    for column in estimated.colnames:
        if column != 'xbeta_center':
            np.testing.assert_array_equal(estimated[column], explicit[column])
        if column not in ['disp_freq', 'xdisp_freq', 'm0', 'n', 'detector_angle', 'xbeta_center']:
            np.testing.assert_array_equal(estimated[column], main[column])
    np.testing.assert_array_equal(estimated['xbeta_center'], 0)
    np.testing.assert_allclose(explicit['xbeta_center'], [21.2, 16.75, 14.1, 10.9, 13.7, 16.8])
    np.testing.assert_allclose(estimated['detector_angle'], [-9.5, -7., -6., -5., -6.1, -6.5])
    np.testing.assert_array_equal(estimated['disp_freq'], [50.15, 50.15, 50.15, 24., 24., 24.])
    np.testing.assert_array_equal(estimated['xdisp_freq'], [2000, 1100, 625, 350, 300, 270])
    np.testing.assert_array_equal(estimated['m0'] - estimated['n'] + 1, [85, 56, 37, 57, 42, 31])


def test_corrected_estimation_maps_requested_band_to_detector_height(comparison):
    variants, reference = comparison
    option, params, effect = variants[0]
    for row in params.table:
        model = effect._spectrographs[row['prefix']]
        wave = np.array([row['min_wave'], row['max_wave']]) * u.Unit(params.meta['min_wave_unit'])
        height = model.wavelength_to_y_pixel(wave).to_value(u.dimensionless_unscaled)
        # The endpoint equation should agree well below a millionth of a pixel.
        np.testing.assert_allclose(height, [0, row['n_xdisp']], atol=1e-6, rtol=0)


def test_explicit_angles_do_not_call_angle_estimator(comparison):
    variants, reference = comparison
    option, params, effect = variants[1]
    grating_class = type(effect._spectrographs['b'].cross_disperser)
    with patch.object(grating_class, 'estimate_xdisp_angle_with_groove_len', side_effect=AssertionError('Explicit CX angles must bypass estimation')):
        regenerated = effect._generate_trace_hdulist(params)
    for actual, expected in zip(regenerated[1:], effect.data_container._file[1:], strict=True):
        np.testing.assert_array_equal(actual.data, expected.data)


def test_missing_archive_angles_fail(comparison):
    variants, reference = comparison
    archive = Table.read(ARCHIVE)
    archive.remove_column('cx_coordinate_tilt_x_deg')
    with pytest.raises(KeyError, match='cx_coordinate_tilt_x_deg'):
        geometry.comparison_parameters(variants[2][1].table, archive)


def test_three_rows_plot_every_full_fsr_on_common_axes(comparison, monkeypatch, tmp_path):
    monkeypatch.setattr(geometry, 'build_comparison', lambda scopesim_repo, irdb_repo, archive: comparison)
    variants, fig = geometry.compare_geometry(ROOT.parent / 'ScopeSim', ROOT.parent / 'irdb', ARCHIVE, tmp_path)
    assert len(fig.axes) == 18
    for row_index, (option, params, effect) in enumerate(variants):
        for channel_index, row in enumerate(params.table):
            axis = fig.axes[6*row_index + channel_index]
            centerlines = [collection for collection in axis.collections if isinstance(collection, LineCollection)]
            assert len(centerlines) == row['n']
            assert len(axis.lines) == 2*row['n']
            assert all(len(trace.get_segments()) == row['n_disp'] for trace in centerlines)
            assert axis.get_xlim() == fig.axes[0].get_xlim()
            assert axis.get_ylim() == fig.axes[channel_index].get_ylim()
            assert centerlines[0].get_clim() == fig.axes[channel_index].collections[0].get_clim()
        assert fig.axes[6*row_index].get_ylabel().startswith(geometry.LABELS[row_index])
    orders = Table.read(tmp_path / 'orders.csv')
    assert [sum(orders['option'] == option) for option in [1, 2, 3]] == [126, 126, 127]
    k = orders[(orders['option'] == 3) & (orders['channel'] == 'k')]
    assert k['x_max_pix'].max() > 2048
    assert (tmp_path / 'detector_footprints.png').is_file()


def test_full_fsr_is_not_truncated_to_the_detector():
    model = spectrograph_factory(1990, 2450, 225, 18000, 65.6, 31, 38, 1/24, 4.46,
                                2048, 2048, .015, xdisp_groove_length=1/270, dispersion_focal_len=270)
    model._slit_offset_pix = np.array([-32., 0., 32.])
    model._detector_cos, model._detector_sin = 1., 0.
    model._detector_center = (1024., 1024.)
    footprints = fsr_footprints(model)
    assert [order for order, wave, x, y in footprints] == list(range(31, 39))
    for order, wave, x, y in footprints:
        assert wave.unit.is_equivalent(u.nm)
        center = model.x_pixel_to_wavelength(model.detector.n_pix_x//2 + .5, order)
        np.testing.assert_allclose((wave[-1] - wave[0]).to_value(u.nm), (center/order).to_value(u.nm), rtol=1e-12)
        np.testing.assert_allclose(y[2] - y[0], 64., atol=1e-10)
    assert footprints[0][2].min() < 0
    assert footprints[0][2].max() > model.detector.n_pix_x
    fig, axis = plt.subplots()
    plot_footprints(axis, footprints, {'n_disp': 2048, 'n_xdisp': 2048}, (1900, 2500))
    # The continuous centerline includes every FSR segment, not a sparse display sample.
    assert len(axis.collections[0].get_segments()) == footprints[0][1].size - 1
    assert len(axis.lines) == 2*len(model.orders)
    plt.close(fig)


def test_footprint_requires_the_trace_generators_placement():
    model = spectrograph_factory(1990, 2450, 225, 18000, 65.6, 31, 38, 1/24, 4.46,
                                2048, 2048, .015, xdisp_groove_length=1/270, dispersion_focal_len=270)
    with pytest.raises(AttributeError):
        fsr_footprints(model)
