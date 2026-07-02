from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM

from zs_scopesim_tools import plots
from zs_scopesim_tools import validation


class FakeSpectrum:
    def __init__(self, scale=1.0, unit=None):
        self.scale = scale
        self.unit = unit

    def __call__(self, wave):
        values = np.full(wave.size, self.scale)
        return values * self.unit if self.unit is not None else values


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
        validation.plot_trace_resolving_power_detector_maps
        is plots.plot_trace_resolving_power_detector_maps
    )
    assert (
        validation.plot_trace_sampling_detector_maps
        is plots.plot_trace_sampling_detector_maps
    )
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
    assert axes[0, 1].get_ylabel() == "Flux density"
    fig.clf()


def test_plot_source_converts_photlam_like_spectra_to_human_units():
    class PhotlamSource:
        meta = {"name": "photlam_source"}
        fields = [FakeTableSourceField()]

    source = PhotlamSource()
    source.fields[0].spectra = {0: FakeSpectrum(1.0, PHOTLAM)}

    fig, axes = plots.plot_source(source, wave=np.linspace(0.4, 0.8, 4) * u.um)

    assert "photons" in axes[0, 1].get_ylabel()
    np.testing.assert_allclose(axes[0, 1].lines[0].get_ydata(), 1.5e5)
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
    assert "photons" in axes.flat[0].get_ylabel()
    assert "Image Plane 0" in axes.flat[0].get_title()
    fig.clf()


def test_transmission_plot_uses_active_slit_for_total_throughput():
    wave = np.array([400.0, 500.0]) * u.nm
    data = {
        "wave_nm": wave,
        "channels": {
            0: {
                "label": "B",
                "optics_groups": {
                    "telescope": np.array([0.8, 0.8]),
                    "preoptics": np.array([0.5, 0.5]),
                    "ir_blocking_filter": np.array([0.2, 0.2]),
                },
                "instrument_optics_groups": {
                    "preoptics": np.array([0.5, 0.5]),
                    "ir_blocking_filter": np.array([0.2, 0.2]),
                },
                "telescope_throughput": np.array([0.8, 0.8]),
                "dichroic_total": np.array([0.9, 0.9]),
                "instrument_optics_total": np.array([0.5, 0.5]),
                "orders": {
                    "B1": {
                        "disperser": np.array([0.6, 0.7]),
                        "detector_qe": np.array([0.7, 0.8]),
                        "instrument": np.array([0.4, 0.6]),
                        "total_with_telescope_no_slit": np.array([0.2, 0.3]),
                    },
                    "B2": {
                        "disperser": np.array([np.nan, 0.65]),
                        "detector_qe": np.array([np.nan, 0.85]),
                        "instrument": np.array([np.nan, 0.55]),
                        "total_with_telescope_no_slit": np.array([np.nan, 0.32]),
                    },
                },
            },
            3: {
                "label": "YJ",
                "optics_groups": {
                    "telescope": np.array([0.8, 0.8]),
                    "camera": np.array([0.6, 0.6]),
                },
                "instrument_optics_groups": {
                    "camera": np.array([0.6, 0.6]),
                },
                "telescope_throughput": np.array([0.8, 0.8]),
                "dichroic_total": np.array([0.85, 0.85]),
                "instrument_optics_total": np.array([0.6, 0.6]),
                "orders": {
                    "YJ1": {
                        "disperser": np.array([0.5, 0.6]),
                        "detector_qe": np.array([0.65, 0.75]),
                        "instrument": np.array([0.5, 0.7]),
                        "total_with_telescope_no_slit": np.array([0.25, 0.35]),
                    },
                },
            },
        },
    }
    slit_loss_data = {
        "arms": {
            "VIS": {
                "wave_nm": wave,
                "curves": {
                    "no_ao_current_adc_residual": {
                        "throughput": np.array([0.5, 0.5]),
                    },
                },
            },
            "NIR": {
                "wave_nm": wave,
                "curves": {
                    "no_ao_current_adc_residual": {
                        "throughput": np.array([0.8, 0.8]),
                    },
                },
            },
        },
    }

    fig, axes = plots.plot_transmission_sanity(
        data,
        slit_loss_data=slit_loss_data,
    )

    assert axes["components"].shape == (2, 3)
    summary_ax = axes["summary"]
    component_labels = [
        line.get_label() for line in axes["components"][0, 0].lines
    ]
    assert "telescope" in component_labels
    assert "preoptics" in component_labels
    assert "ir_blocking_filter" not in component_labels
    assert "spectrograph optics" in component_labels
    assert "trace QE" in component_labels
    labels = [line.get_label() for line in summary_ax.lines]
    assert "B instrument" in labels
    assert "B total" in labels
    assert "active slit" in labels
    assert "trace QE median" in labels
    assert "spectrograph optics" in labels
    qe_median = next(
        line for line in summary_ax.lines if line.get_label() == "trace QE median"
    )
    np.testing.assert_allclose(qe_median.get_ydata(), [0.7, 0.825])
    assert any(
        np.allclose(line.get_ydata(), [np.nan, 0.85], equal_nan=True)
        for line in summary_ax.lines
    )
    b_total = next(line for line in summary_ax.lines if line.get_label() == "B total")
    np.testing.assert_allclose(b_total.get_ydata(), [0.1, 0.15])
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
        "psf_modes": {
            "no_ao": {"label": '0.6" NS'},
            "ao": {"label": "AO"},
        },
        "arms": {
            "VIS": {
                "slit_widths_arcsec": slit_widths,
                "current_slit_width_arcsec": 0.7 * u.arcsec,
                "selector_slit_widths_arcsec": [0.3, 0.7] * u.arcsec,
                "wavelengths_nm": [500.0] * u.nm,
                "curves": {
                    "no_ao_500nm": {
                        "loss": np.linspace(0.8, 0.1, slit_widths.size),
                        "linestyle": "-",
                        "psf_mode": "no_ao",
                        "wavelength_nm": 500.0,
                    },
                    "ao_500nm": {
                        "loss": np.linspace(0.2, 0.01, slit_widths.size),
                        "linestyle": "--",
                        "psf_mode": "ao",
                        "wavelength_nm": 500.0,
                    },
                },
            },
            "NIR": {
                "slit_widths_arcsec": slit_widths,
                "current_slit_width_arcsec": 0.7 * u.arcsec,
                "selector_slit_widths_arcsec": [0.3, 0.7] * u.arcsec,
                "wavelengths_nm": [1250.0] * u.nm,
                "curves": {
                    "no_ao_1250nm": {
                        "loss": np.linspace(0.7, 0.08, slit_widths.size),
                        "linestyle": "-",
                        "psf_mode": "no_ao",
                        "wavelength_nm": 1250.0,
                    },
                    "ao_1250nm": {
                        "loss": np.linspace(0.1, 0.005, slit_widths.size),
                        "linestyle": "--",
                        "psf_mode": "ao",
                        "wavelength_nm": 1250.0,
                    },
                },
            },
        },
    }

    fig, axes = plots.plot_slit_width_loss(data)

    assert axes.shape == (1, 2)
    assert len(axes[0, 0].lines) == 3
    assert len(axes[0, 0].collections) == 2
    labels = [text.get_text() for text in fig.legends[0].texts]
    assert '0.6" NS, 500/1250 nm' in labels
    assert "AO, 500/1250 nm" in labels
    assert "available slit widths" in labels
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


def test_readout_overview_can_share_color_scale():
    class FakeHDU:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)

    hdul = [
        FakeHDU([[0, 1], [2, 3]]),
        FakeHDU([[100, 101], [102, 103]]),
    ]

    fig, axes = plots.plot_readout_overview(
        hdul,
        titles=["YJ", "H"],
        clip=None,
        shared_scale=[["YJ", "H"]],
        annotate_stats=True,
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 103.0)
    assert axes[0, 1].images[0].get_clim() == (0.0, 103.0)
    assert "mean" in axes[0, 0].texts[0].get_text()
    fig.clf()


def test_readout_delta_overview_uses_absolute_clip_from_zero():
    class FakeHDU:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)

    fig, axes = plots.plot_readout_delta_overview(
        [FakeHDU([[10, -10], [3, -3]])],
        [FakeHDU([[0, 0], [0, 0]])],
        titles=["B"],
        clip=5,
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 5.0)
    assert "min delta -10" in axes[0, 0].texts[0].get_text()
    fig.clf()


def test_trace_resolution_detector_map_plot_smoke():
    table = Table(rows=[
        {
            "channel": "B",
            "image_plane_id": 0,
            "trace_id": "B_1",
            "detector_naxis1": 12,
            "detector_naxis2": 10,
            "detector_x_pix": 2.0,
            "detector_y_pix": 3.0,
            "wave_nm": 1000.0,
            "dispersion_nm_pix": 0.05,
            "resolving_power_R": 20000.0,
            "spectral_element_width_pix": 4.2,
            "seeing_fwhm_arcsec": 0.6,
            "spatial_fwhm_pix": 4.0,
        },
        {
            "channel": "B",
            "image_plane_id": 0,
            "trace_id": "B_1",
            "detector_naxis1": 12,
            "detector_naxis2": 10,
            "detector_x_pix": 3.0,
            "detector_y_pix": 3.0,
            "wave_nm": 1005.0,
            "dispersion_nm_pix": 0.05,
            "resolving_power_R": 20100.0,
            "spectral_element_width_pix": 4.3,
            "seeing_fwhm_arcsec": 0.6,
            "spatial_fwhm_pix": 4.0,
        },
    ])

    fig, axes = plots.plot_trace_resolving_power_detector_maps(table)

    assert axes[0, 0].collections
    assert "B (id 0)" in axes[0, 0].get_title()
    fig.clf()

    fig, axes = plots.plot_trace_sampling_detector_maps(table)

    assert axes[0, 0].collections
    assert "pix/resel" in axes[0, 0].get_title()
    fig.clf()
