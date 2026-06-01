from __future__ import annotations

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
