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
    assert validation.plot_detector_image_grid is plots.plot_detector_image_grid
    assert validation.plot_slit_adc_psf_scenes is plots.plot_slit_adc_psf_scenes
    assert validation.plot_slit_loss_by_arm is plots.plot_slit_loss_by_arm
    assert validation.plot_slit_width_loss is plots.plot_slit_width_loss
    assert validation.plot_slit_pair_geometry is plots.plot_slit_pair_geometry
    assert (
        validation.plot_resolving_power_echellogram
        is plots.plot_resolving_power_echellogram
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


def test_detector_background_budget_plot_omits_disabled_diffuse_terms():
    table = Table({
        "channel": ["B", "K"],
        "post_diffuse_e_pix": [0.0, 0.0],
        "dark_current_e_pix": [3.0, 20.0],
        "bias_e_pix": [1040.0, 1040.0],
        "full_well_e": [64000.0, 64000.0],
        "signal_fraction_of_full_well": [3.0 / 64000.0, 20.0 / 64000.0],
        "saturation_status": ["ok", "ok"],
        "diffuse_shot_noise_e_rms": [0.0, 0.0],
        "dark_shot_noise_e_rms": [3.0**0.5, 20.0**0.5],
        "read_noise_e_rms": [5.0, 0.5],
        "total_noise_e_rms": [(3.0 + 25.0)**0.5, (20.0 + 0.25)**0.5],
    })

    fig, axes = plots.plot_detector_background_budget(table)

    labels = [
        label
        for ax in axes
        for label in ax.get_legend_handles_labels()[1]
    ]
    assert not any("diffuse" in label for label in labels)
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
                        "wave_min": 400 * u.nm,
                        "wave_max": 500 * u.nm,
                    },
                    "B2": {
                        "disperser": np.array([np.nan, 0.65]),
                        "detector_qe": np.array([np.nan, 0.85]),
                        "instrument": np.array([np.nan, 0.55]),
                        "total_with_telescope_no_slit": np.array([np.nan, 0.32]),
                        "wave_min": 450 * u.nm,
                        "wave_max": 500 * u.nm,
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
                        "wave_min": 400 * u.nm,
                        "wave_max": 500 * u.nm,
                    },
                },
            },
        },
    }
    slit_loss_data = {
        "active_psf_mode": "no_ao",
        "arms": {
            "VIS": {
                "wave_nm": wave,
                "curves": {
                    "no_ao_current_adc_residual": {
                        "throughput": np.array([0.5, 0.5]),
                        "slit_width_arcsec": 0.7 * u.arcsec,
                        "psf_mode": "no_ao",
                    },
                },
            },
            "NIR": {
                "wave_nm": wave,
                "curves": {
                    "no_ao_current_adc_residual": {
                        "throughput": np.array([0.8, 0.8]),
                        "slit_width_arcsec": 0.7 * u.arcsec,
                        "psf_mode": "no_ao",
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
    assert "instrument" in component_labels
    assert '0.7" slit, Natural seeing' in component_labels
    assert axes["components"][0, 0].get_title() == "B"
    assert "aperture" not in axes["components"][0, 0].get_title()
    assert axes["components"][0, 0].get_xlim() == (400.0, 500.0)
    labels = [line.get_label() for line in summary_ax.lines]
    assert "B instrument" in labels
    assert "B total" in labels
    assert '0.7" slit, Natural seeing' in labels
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


def test_transmission_plot_supports_individual_and_combined_only_modes():
    wave = np.array([400.0, 500.0]) * u.nm
    order = {
        "disperser": np.array([0.6, 0.7]),
        "detector_qe": np.array([0.7, 0.8]),
        "instrument": np.array([0.4, 0.6]),
        "total_with_telescope_no_slit": np.array([0.2, 0.3]),
        "wave_min": 400 * u.nm,
        "wave_max": 500 * u.nm,
    }
    channel = {
        "label": "B",
        "optics_groups": {"telescope": np.array([0.8, 0.8])},
        "instrument_optics_groups": {"camera": np.array([0.6, 0.6])},
        "telescope_throughput": np.array([0.8, 0.8]),
        "dichroic_total": np.array([0.9, 0.9]),
        "orders": {"B1": order},
    }
    data = {"wave_nm": wave, "channels": {0: channel}}

    fig, axes = plots.plot_transmission_sanity(
        data, show_combined_channels=False,
    )
    assert axes["components"].shape == (2, 3)
    assert axes["summary"] is None
    fig.clf()

    fig, axes = plots.plot_transmission_sanity(
        data,
        show_individual_channels=False,
        combined_title="Combined",
    )
    assert axes["components"] is None
    assert axes["summary"].get_title() == "Combined"
    fig.clf()


def test_transmission_plot_excludes_displayed_component_globally():
    wave = np.array([400.0, 500.0]) * u.nm
    order = {
        "disperser": np.array([0.6, 0.7]),
        "detector_qe": np.array([0.7, 0.8]),
        "instrument": np.array([0.4, 0.6]),
        "total_with_telescope_no_slit": np.array([0.2, 0.3]),
        "wave_min": 400 * u.nm,
        "wave_max": 500 * u.nm,
    }
    data = {
        "wave_nm": wave,
        "channels": {
            0: {
                "label": "B",
                "optics_groups": {"telescope": np.array([0.8, 0.8])},
                "instrument_optics_groups": {
                    "camera": np.array([0.6, 0.6]),
                },
                "telescope_throughput": np.array([0.8, 0.8]),
                "dichroic_total": np.array([0.9, 0.9]),
                "orders": {"B1": order},
            },
        },
    }

    fig, axes = plots.plot_transmission_sanity(
        data, exclude_lines=["instrument", "spectrograph optics"],
    )
    component_labels = [
        line.get_label() for line in axes["components"][0, 0].lines
    ]
    summary_labels = [line.get_label() for line in axes["summary"].lines]
    assert "instrument" not in component_labels
    assert "spectrograph optics" not in component_labels
    assert "B instrument" not in summary_labels
    assert "spectrograph optics" not in summary_labels
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


def test_slit_loss_legend_marks_arm_specific_current_extrema():
    wave = np.linspace(310, 980, 5) * u.nm

    def curve(role, width, current):
        return {
            "label": f"{role} slit",
            "loss": np.linspace(0.1, 0.2, wave.size),
            "psf_mode": "no_ao",
            "psf_label": '0.65" NS',
            "current_psf": True,
            "slit_role": role,
            "slit_label": f'{width:.2f}"',
            "slit_width_arcsec": width * u.arcsec,
            "current_slit": current,
            "airmass": 1.1,
            "airmass_label": "X=1.1",
            "current_airmass": True,
            "adc_state": "adc_residual",
            "current_adc": True,
        }

    data = {
        "psf_modes": {
            "no_ao": {"label": '0.65" NS', "current": True},
        },
        "arms": {
            "VIS": {
                "wave_nm": wave,
                "slit_width_arcsec": 0.33 * u.arcsec,
                "selector_slit_widths_arcsec": [0.33, 0.70, 1.25] * u.arcsec,
                "curves": {
                    "vis_narrow": curve("narrowest", 0.33, True),
                    "vis_wide": curve("widest", 1.25, False),
                },
            },
            "NIR": {
                "wave_nm": wave,
                "slit_width_arcsec": 1.25 * u.arcsec,
                "selector_slit_widths_arcsec": [0.33, 0.70, 1.25] * u.arcsec,
                "curves": {
                    "nir_narrow": curve("narrowest", 0.33, False),
                    "nir_wide": curve("widest", 1.25, True),
                },
            },
        },
    }

    fig, _axes = plots.plot_slit_loss_by_arm(data)
    labels = [text.get_text() for text in fig.legends[0].texts]

    assert '0.33"*' in labels
    assert '1.25"^' in labels
    assert "*/^ Current slit" in labels
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
    assert " - reference" in axes[0, 0].get_title()
    assert not axes[0, 0].texts
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


def test_readout_overview_can_use_data_floor_for_biased_frames():
    class FakeHDU:
        data = np.arange(100, 200, dtype=float).reshape(10, 10)

    fig, axes = plots.plot_readout_overview(
        [FakeHDU()],
        titles=["B"],
        clip=None,
        zero_floor=False,
    )

    assert axes[0, 0].images[0].get_clim() == (100.0, 199.0)
    fig.clf()


def test_detector_image_grid_accepts_explicit_display_limits():
    import matplotlib.pyplot as plt

    plt.close("all")
    images = [np.arange(100, dtype=float).reshape(10, 10)]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B"],
        clip=0.5,
        vmin=10.0,
        vmax=20.0,
    )

    assert axes[0, 0].images[0].get_clim() == (10.0, 20.0)
    fig.clf()


def test_detector_image_grid_accepts_per_panel_display_limits():
    images = [
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[100.0, 101.0], [102.0, 103.0]]),
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["YJ", "H"],
        clip=None,
        vmin=[0.0, 100.0],
        vmax=[4.0, 104.0],
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 4.0)
    assert axes[0, 1].images[0].get_clim() == (100.0, 104.0)
    fig.clf()


def test_detector_image_grid_accepts_title_mapped_display_limits():
    images = [
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[100.0, 101.0], [102.0, 103.0]]),
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["YJ", "H"],
        clip=None,
        vmax={"YJ": 5.0, "H": 105.0},
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 5.0)
    assert axes[0, 1].images[0].get_clim() == (0.0, 105.0)
    fig.clf()


def test_detector_image_grid_supports_display_scale_and_interpolation():
    images = [np.array([[0.0, 1.0], [4.0, 9.0]])]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B"],
        clip=None,
        image_scale="sqrt",
        image_interpolation="hanning",
    )

    image = axes[0, 0].images[0]
    assert image.norm.__class__.__name__ == "PowerNorm"
    assert image.get_interpolation() == "hanning"
    fig.clf()


def test_detector_image_grid_uses_independent_panel_scales():
    images = [
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[100.0, 101.0], [102.0, 103.0]]),
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["YJ", "H"],
        clip=None,
        shared_scale=False,
        colorbar_mode="per-panel",
        colorbar_label="S/N",
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 3.0)
    assert axes[0, 1].images[0].get_clim() == (0.0, 103.0)
    fig.clf()


def test_detector_image_grid_rejects_shared_colorbar_without_shared_scale():
    import matplotlib.pyplot as plt

    plt.close("all")
    images = [
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[100.0, 101.0], [102.0, 103.0]]),
    ]

    with np.testing.assert_raises_regex(ValueError, "requires one shared scale"):
        plots.plot_detector_image_grid(
            images,
            titles=["YJ", "H"],
            clip=None,
            shared_scale=False,
            colorbar_mode="shared",
        )


def test_detector_image_grid_centers_mixed_aspect_axes():
    images = [
        np.zeros((8, 4), dtype=float),
        np.zeros((8, 4), dtype=float),
        np.zeros((8, 4), dtype=float),
        np.zeros((4, 4), dtype=float),
        np.zeros((4, 4), dtype=float),
        np.zeros((4, 4), dtype=float),
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B", "G", "R", "YJ", "H", "K"],
        clip=None,
        colorbar_mode="per-panel",
    )

    assert [ax.get_anchor() for ax in axes.flat[:6]] == ["C"] * 6
    fig.clf()


def test_detector_image_grid_shows_physical_pixel_axes_and_shared_labels():
    images = [
        np.zeros((1, 4096), dtype=float),
        np.zeros((4096, 1), dtype=float),
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["wide", "tall"],
        clip=None,
    )

    assert np.array_equal(axes[0, 0].get_xticks(), [0, 2048, 4096])
    assert np.array_equal(axes[0, 1].get_yticks(), [0, 2048, 4096])
    assert fig._supxlabel.get_text() == "Pixels"
    assert fig._supylabel.get_text() == "Pixels"
    fig.clf()


def test_detector_image_grid_limits_per_panel_colorbar_tick_density():
    images = [
        np.linspace(1.0, 1.0e7 * (idx + 1), 64).reshape(8, 8)
        for idx in range(6)
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B", "G", "R", "YJ", "H", "K"],
        clip=None,
        colorbar_mode="per-panel",
    )

    colorbar_axes = [
        ax for ax in fig.axes
        if ax not in set(axes.flat)
    ]
    assert len(colorbar_axes) == 6
    assert all(len(ax.get_yticks()) <= 7 for ax in colorbar_axes)
    assert all(
        ax.yaxis.get_major_formatter().__class__.__name__ == "ScalarFormatter"
        for ax in colorbar_axes
    )
    for image_ax, colorbar_ax in zip(axes.flat, colorbar_axes, strict=True):
        vmin, vmax = image_ax.images[0].get_clim()
        ticks = colorbar_ax.get_yticks()
        assert np.all(ticks >= vmin)
        assert np.all(ticks <= vmax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for colorbar_ax in colorbar_axes:
        labels = [
            label.get_window_extent(renderer)
            for label in colorbar_ax.get_yticklabels()
            if label.get_visible() and label.get_text()
        ]
        assert all(
            lower.y1 <= upper.y0
            for lower, upper in zip(labels, labels[1:], strict=False)
        )
    fig.clf()


def test_detector_image_grid_uses_log_colorbar_locator_for_shared_scale():
    images = [
        np.geomspace(1.0, 1.0e6, 64).reshape(8, 8)
        for _idx in range(6)
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B", "G", "R", "YJ", "H", "K"],
        clip=None,
        shared_scale=True,
        colorbar_mode="shared",
        image_scale="log",
    )

    colorbar_axes = [
        ax for ax in fig.axes
        if ax not in set(axes.flat)
    ]
    assert len(colorbar_axes) == 1
    assert colorbar_axes[0].yaxis.get_major_locator().__class__.__name__ == "FixedLocator"
    assert colorbar_axes[0].yaxis.get_major_formatter().__class__.__name__ == "LogFormatterSciNotation"
    ticks = colorbar_axes[0].get_yticks()
    assert np.all(ticks >= 1.0)
    assert np.all(ticks <= 1.0e6)
    fig.clf()


def test_detector_image_grid_bounds_symlog_colorbar_ticks():
    images = [
        np.linspace(-1.0e4, 1.0e4, 81).reshape(9, 9)
        for _idx in range(6)
    ]

    fig, axes = plots.plot_detector_image_grid(
        images,
        titles=["B", "G", "R", "YJ", "H", "K"],
        clip=None,
        shared_scale=True,
        colorbar_mode="shared",
        symmetric=True,
        image_scale="symlog",
    )

    colorbar_axes = [
        ax for ax in fig.axes
        if ax not in set(axes.flat)
    ]
    assert len(colorbar_axes) == 1
    ticks = colorbar_axes[0].get_yticks()
    assert np.all(ticks >= -1.0e4)
    assert np.all(ticks <= 1.0e4)
    assert colorbar_axes[0].yaxis.get_major_formatter().__class__.__name__ == "LogFormatterSciNotation"
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
        annotate_delta=True,
    )

    assert axes[0, 0].images[0].get_clim() == (0.0, 5.0)
    assert "min -10" in axes[0, 0].texts[0].get_text()
    fig.clf()


def test_show_and_save_hdul_saves_hdul_reference_and_delta(tmp_path):
    from astropy.io import fits
    import matplotlib.pyplot as plt

    plt.close("all")

    signal = [
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((4, 4), value, dtype=float)),
        ])
        for value in (10.0, 20.0)
    ]
    reference = [
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((4, 4), value, dtype=float)),
        ])
        for value in (3.0, 5.0)
    ]

    result = plots.show_and_save_hdul(
        signal,
        label="case",
        titles=["B", "G"],
        reference_hdul=reference,
        output_dir=tmp_path,
        show_hdul=False,
        show_delta=True,
        show_cross_dispersion=True,
        save=True,
        figure_title="Case Delta",
        delta_clip=None,
    )

    assert result["figures"]["hdul"] is None
    assert result["figures"]["delta"] is not None
    assert result["figures"]["cross_dispersion"] is not None
    assert result["figures"]["delta"]._suptitle.get_text() == "Case Delta"
    assert result["figures"]["cross_dispersion"]._suptitle.get_text() == "Case Delta"
    assert len(result["files"]["hdul"]) == 2
    assert len(result["files"]["reference"]) == 2
    assert len(result["files"]["delta"]) == 2
    assert len(result["files"]["figures"]) == 2
    for paths in result["files"].values():
        for path in paths:
            assert path.exists()
    with fits.open(result["files"]["delta"][0]) as hdul:
        np.testing.assert_allclose(hdul[1].data, 7.0)
    for figure in result["figures"].values():
        if figure is not None:
            plt.close(figure)


def test_show_and_save_hdul_passes_display_scale_and_panel_limits():
    from astropy.io import fits
    import matplotlib.pyplot as plt

    signal = [
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((4, 4), value, dtype=float)),
        ])
        for value in (10.0, 20.0)
    ]

    result = plots.show_and_save_hdul(
        signal,
        label="case",
        titles=["B", "G"],
        show_hdul=True,
        save=False,
        figure_title="Panel Limits",
        hdul_clip=None,
        hdul_vmax=[12.0, 25.0],
        hdul_image_scale="sqrt",
        hdul_image_interpolation="hanning",
    )

    assert result["directory"] is None
    assert result["files"] == {"hdul": [], "reference": [], "delta": [], "figures": []}
    assert result["figures"]["hdul"]._suptitle.get_text() == "Panel Limits"
    axes = result["figures"]["hdul"].axes
    plotted = [ax.images[0] for ax in axes if ax.images]
    assert [image.get_clim() for image in plotted] == [(10.0, 12.0), (20.0, 25.0)]
    assert {image.norm.__class__.__name__ for image in plotted} == {"PowerNorm"}
    assert {image.get_interpolation() for image in plotted} == {"hanning"}
    plt.close(result["figures"]["hdul"])


def test_show_and_save_hdul_save_false_overrides_specific_save_flags(tmp_path):
    from astropy.io import fits

    signal = [
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((4, 4), 10.0, dtype=float)),
        ])
    ]
    reference = [
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((4, 4), 3.0, dtype=float)),
        ])
    ]

    result = plots.show_and_save_hdul(
        signal,
        label="case",
        titles=["B"],
        reference_hdul=reference,
        output_dir=tmp_path,
        show_hdul=True,
        show_delta=True,
        save=False,
        save_hdul=True,
        save_reference=True,
        save_delta=True,
        save_figures=True,
    )

    assert result["directory"] is None
    assert result["files"] == {"hdul": [], "reference": [], "delta": [], "figures": []}
    assert not (tmp_path / "readouts").exists()


def test_show_and_save_hdul_saves_array_inputs(tmp_path):
    from astropy.io import fits

    result = plots.show_and_save_hdul(
        [np.full((4, 4), 5.0, dtype=float)],
        label="array_case",
        titles=["B"],
        output_dir=tmp_path,
        show_hdul=False,
        save=True,
        save_figures=False,
    )

    assert len(result["files"]["hdul"]) == 1
    with fits.open(result["files"]["hdul"][0]) as hdul:
        np.testing.assert_allclose(hdul[0].data, 5.0)


def test_resolving_power_echellogram_smoke():
    table = Table(rows=[
        {
            "channel": "B",
            "image_plane_id": 0,
            "trace_id": "B_1",
            "sample_index": 0,
            "detector_x_mm": -1.0,
            "detector_y_mm": 0.2,
            "detector_pixel_size_mm": 0.01,
            "detector_naxis1": 200,
            "detector_naxis2": 50,
            "wave_nm": 1000.0,
            "dispersion_nm_pix": 0.05,
            "resolving_power_R": 20000.0,
            "spectral_fwhm_pix": 4.2,
            "seeing_fwhm_arcsec": 0.6,
            "spatial_fwhm_pix": 4.0,
        },
        {
            "channel": "B",
            "image_plane_id": 0,
            "trace_id": "B_1",
            "sample_index": 1,
            "detector_x_mm": 1.0,
            "detector_y_mm": 0.2,
            "detector_pixel_size_mm": 0.01,
            "detector_naxis1": 200,
            "detector_naxis2": 50,
            "wave_nm": 1005.0,
            "dispersion_nm_pix": 0.05,
            "resolving_power_R": 20100.0,
            "spectral_fwhm_pix": 4.2,
            "seeing_fwhm_arcsec": 0.6,
            "spatial_fwhm_pix": 4.0,
        },
    ])

    fig, axes = plots.plot_resolving_power_echellogram(
        table, trace_width_fraction=0.25)

    assert axes[0, 0].collections
    assert len(axes[0, 0].collections[0].get_segments()) == 2
    assert "FWHM" in axes[0, 0].get_title()
    assert "spat" in axes[0, 0].get_title()
    assert axes[0, 0].get_xlabel() == "Pixel"
    assert axes[0, 0].get_ylabel() == "Pixel"
    np.testing.assert_allclose(axes[0, 0].get_xlim(), [0, 200])
    np.testing.assert_allclose(axes[0, 0].get_ylim(), [0, 50])
    np.testing.assert_allclose(axes[0, 0].collections[0].get_segments()[0][1], [0, 45])
    fig.clf()
