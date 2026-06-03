from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import Table

from zs_scopesim_tools import plots
from zs_scopesim_tools import validation


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
    assert validation.plot_slit_pair_geometry is plots.plot_slit_pair_geometry
    assert validation.plot_readout_overview is plots.plot_readout_overview


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
                    "zenith": {
                        "label": "zenith",
                        "loss": np.linspace(0.1, 0.2, wave.size),
                    },
                    "elevation_60_ad_only": {
                        "label": "60 deg elevation, AD only",
                        "loss": np.linspace(0.2, 0.4, wave.size),
                    },
                },
            },
        },
    }

    fig, axes = plots.plot_slit_loss_by_arm(data)

    assert axes.shape == (1, 1)
    fig.clf()
