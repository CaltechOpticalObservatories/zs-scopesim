"""FSR plots retain the actual model's complete wavelength and slit domain."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from astropy import units as u

from scopesim.optics.echelle import spectrograph_factory
from validation.zshooter_xdisp.geometry import fsr_footprints, plot_footprints


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
        np.testing.assert_allclose((wave[-1] - wave[0]).to_value(u.nm), model.fsr(order).to_value(u.nm), rtol=1e-12)
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
