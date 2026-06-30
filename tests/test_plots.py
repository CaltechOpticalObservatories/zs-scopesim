from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import Table

from zs_scopesim_tools import plots
from zs_scopesim_tools import validation


class FakeSpectrum:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, wave):
        return np.full(wave.size, self.scale)


class FakeTableSourceField:
    def __init__(self):
        self.field = Table({
            "x": [0.0, 1.0],
            "y": [0.0, -1.0],
            "ref": [0, 0],
            "weight": [1.0, 0.5],
        })
        self.field["x"].unit = u.arcsec
        self.field["y"].unit = u.arcsec
        self.spectra = {0: FakeSpectrum(2.0)}


class FakeSource:
    meta = {"name": "fake_source"}
    fields = [FakeTableSourceField()]


class EmptySource:
    fields = []


def test_validation_reexports_plot_helpers():
    assert validation.plot_source is plots.plot_source
    assert validation.plot_transmission_sanity is plots.plot_transmission_sanity
    assert validation.plot_emissivity_sanity is plots.plot_emissivity_sanity
    assert (
        validation.plot_post_disperser_diffuse_background
        is plots.plot_post_disperser_diffuse_background
    )
    assert (
        validation.plot_detector_background_budget
        is plots.plot_detector_background_budget
    )
    assert validation.plot_slit_adc_psf_scenes is plots.plot_slit_adc_psf_scenes
    assert validation.plot_slit_loss_by_arm is plots.plot_slit_loss_by_arm
    assert validation.plot_slit_width_loss is plots.plot_slit_width_loss
    assert validation.plot_slit_pair_geometry is plots.plot_slit_pair_geometry
    assert (
        validation.plot_readout_cross_dispersion_cut
        is plots.plot_readout_cross_dispersion_cut
    )
    assert validation.plot_readout_delta_overview is plots.plot_readout_delta_overview
    assert validation.plot_readout_overview is plots.plot_readout_overview


def test_plot_source_sums_table_source_fields_by_default():
    fig, axes = plots.plot_source(FakeSource(), wave=np.linspace(0.4, 0.8, 4) * u.um)

    assert axes.shape == (1, 2)
    assert len(axes[0, 0].collections) == 1
    assert len(axes[0, 1].lines) == 1
    assert axes[0, 1].lines[0].get_label() == "fake_source total (2 points)"
    np.testing.assert_allclose(axes[0, 1].lines[0].get_ydata(), 3.0)
    fig.clf()


def test_plot_source_can_plot_individual_table_source_rows():
    fig, axes = plots.plot_source(
        FakeSource(), wave=np.linspace(0.4, 0.8, 4) * u.um,
        individual=True,
    )

    assert len(axes[0, 1].lines) == 2
    labels = [line.get_label() for line in axes[0, 1].lines]
    assert labels == ["fake_source row 0", "fake_source row 1"]
    fig.clf()


def test_plot_source_reports_empty_source_fields():
    with np.testing.assert_raises_regex(ValueError, "Source contains no fields"):
        plots.plot_source(EmptySource())


def test_plot_source_accepts_log_spectrum_options():
    fig, axes = plots.plot_source(
        FakeSource(),
        wave=np.linspace(0.4, 0.8, 4) * u.um,
        spectrum_yscale="log",
        spectrum_linewidth=0.7,
    )

    assert axes[0, 1].get_yscale() == "log"
    assert axes[0, 1].lines[0].get_linewidth() == 0.7
    fig.clf()


def test_detector_background_budget_plot_handles_saturation_annotation():
    table = Table({
        "channel": ["B", "K"],
        "post_diffuse_e_pix": [60.0, 1.0e8],
        "dark_current_e_pix": [3.0, 20.0],
        "additive_signal_e_pix": [63.0, 1.0e8 + 20.0],
        "bias_e_pix": [1040.0, 1040.0],
        "full_well_e": [64000.0, 64000.0],
        "signal_fraction_of_full_well": [
            63.0 / 64000.0,
            (1.0e8 + 20.0) / 64000.0,
        ],
        "saturation_status": ["ok", "saturated"],
        "diffuse_shot_noise_e_rms": [60.0**0.5, 1.0e8**0.5],
        "dark_shot_noise_e_rms": [3.0**0.5, 20.0**0.5],
        "read_noise_e_rms": [5.0, 0.5],
        "total_noise_e_rms": [10.0, 1.0e4],
    })

    fig, axes = plots.plot_detector_background_budget(table)

    assert "SATURATED: K" in axes[0].texts[0].get_text()
    fig.clf()


def test_post_disperser_diffuse_plot_annotates_ir_blocking():
    wave = np.array([300.0, 400.0]) * u.nm
    channels = {}
    for idx, label in enumerate(["B", "G", "R", "YJ", "H", "K"]):
        channels[idx] = {
            "label": label,
            "image_plane_id": idx,
            "trace_wave_min_nm": 300.0,
            "trace_wave_max_nm": 400.0,
            "spectra": {"camera": np.ones(2)},
            "total_spectrum": np.full(2, 2.0),
            "total_spectrum_without_blocking": np.full(2, 3.0),
            "total_rate_ph_s_pix": 2.0,
            "total_rate_without_blocking_ph_s_pix": 3.0,
            "blocking_delta_rate_ph_s_pix": 1.0,
        }

    fig, axes = plots.plot_post_disperser_diffuse_background({
        "wave_nm": wave,
        "channels": channels,
    })

    assert "IR block removes 1" in axes.flat[0].texts[0].get_text()
    assert "unblocked diffuse" in axes.flat[0].texts[0].get_text()
    assert "Image Plane 0" in axes.flat[0].get_title()
    fig.clf()


def test_slit_adc_psf_scene_plot_smoke():
    data = {
        "airmass": 1.3,
        "seeing_arcsec": 0.6 * u.arcsec,
        "slit_width_arcsec": 0.7 * u.arcsec,
        "slit_length_arcsec": 4.0 * u.arcsec,
        "x_arcsec": np.linspace(-1, 1, 8) * u.arcsec,
        "y_arcsec": np.linspace(-2, 2, 10) * u.arcsec,
        "variants": {
            "ad_only": {"label": "AD only"},
            "adc_residual": {"label": "ADC residual"},
        },
        "scenarios": {
            "along": {
                "positions": Table({
                    "x": [0.0, 0.0],
                    "y": [-1.0, 1.0],
                    "weight": [1.0, 1.0],
                }, units=[u.arcsec, u.arcsec, None]),
                "images": {
                    "ad_only": np.ones((10, 8)),
                    "adc_residual": np.eye(10, 8),
                },
            },
        },
    }

    fig, axes = plots.plot_slit_adc_psf_scenes(data)

    assert axes.shape == (1, 2)
    fig.clf()


def test_slit_loss_plot_smoke():
    wave = np.linspace(310, 980, 5) * u.nm
    data = {
        "seeing_arcsec": 0.6 * u.arcsec,
        "arms": {
            "VIS": {
                "wave_nm": wave,
                "slit_width_arcsec": 0.7 * u.arcsec,
                "curves": {
                    "no_ao_zenith": {
                        "label": "no AO, zenith",
                        "loss": np.linspace(0.1, 0.2, wave.size),
                        "color": "tab:blue",
                        "linestyle": "-",
                    },
                    "ao_elevation_60_ad_only": {
                        "label": "AO, 60 deg elevation, AD only",
                        "loss": np.linspace(0.2, 0.4, wave.size),
                        "color": "tab:orange",
                        "linestyle": "--",
                    },
                },
            },
        },
    }

    fig, axes = plots.plot_slit_loss_by_arm(data)

    assert axes.shape == (1, 1)
    fig.clf()


def test_slit_width_loss_plot_smoke():
    slit_widths = np.linspace(0.2, 1.0, 5) * u.arcsec
    data = {
        "arms": {
            "VIS": {
                "slit_widths_arcsec": slit_widths,
                "current_slit_width_arcsec": 0.7 * u.arcsec,
                "selector_slit_widths_arcsec": [0.3, 0.7] * u.arcsec,
                "curves": {
                    "no_ao_500nm": {
                        "label": "no AO, 500 nm",
                        "loss": np.linspace(0.8, 0.1, slit_widths.size),
                        "linestyle": "-",
                        "wavelength_nm": 500.0,
                    },
                    "ao_500nm": {
                        "label": "AO, 500 nm",
                        "loss": np.linspace(0.2, 0.01, slit_widths.size),
                        "linestyle": "--",
                        "wavelength_nm": 500.0,
                    },
                },
            },
        },
    }

    fig, axes = plots.plot_slit_width_loss(data)

    assert axes.shape == (1, 1)
    assert len(axes[0, 0].lines) == 3
    assert len(axes[0, 0].collections) == 2
    fig.clf()


def test_readout_cross_dispersion_cut_plot_smoke():
    class FakeHDU:
        def __init__(self):
            self.data = np.arange(100, dtype=float).reshape(10, 10)

    fig, axes = plots.plot_readout_cross_dispersion_cut(
        [FakeHDU()], titles=["B"], central_columns=80,
    )

    assert axes.shape == (1, 1)
    assert "central 10 cols" in axes[0, 0].get_title()
    assert len(axes[0, 0].lines) == 1
    fig.clf()


def test_readout_delta_overview_plot_smoke():
    class FakeHDU:
        def __init__(self, value):
            self.data = np.full((4, 4), value, dtype=float)

    fig, axes = plots.plot_readout_delta_overview(
        [FakeHDU(3.0)], [FakeHDU(1.0)], titles=["B"],
    )

    assert axes.shape == (1, 1)
    assert "Source - Empty" in axes[0, 0].get_title()
    assert "max |delta| 2" in axes[0, 0].texts[0].get_text()
    fig.clf()


def test_readout_overview_uses_fractional_clip_as_quantile():
    class FakeHDU:
        data = np.arange(100, dtype=float).reshape(10, 10)

    fig, axes = plots.plot_readout_overview([FakeHDU()], titles=["B"], clip=0.5)

    assert axes[0, 0].images[0].get_clim() == (0.0, 49.5)
    fig.clf()


def test_readout_delta_overview_uses_absolute_clip_symmetrically():
    class FakeHDU:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)

    fig, axes = plots.plot_readout_delta_overview(
        [FakeHDU([[10, -10], [3, -3]])],
        [FakeHDU([[0, 0], [0, 0]])],
        titles=["B"],
        clip=5,
    )

    assert axes[0, 0].images[0].get_clim() == (-5.0, 5.0)
    fig.clf()
