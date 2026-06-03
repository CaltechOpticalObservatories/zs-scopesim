from __future__ import annotations

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
    assert validation.plot_slit_pair_geometry is plots.plot_slit_pair_geometry


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
