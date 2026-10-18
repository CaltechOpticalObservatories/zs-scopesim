"""Physical checks for the prescription's separate paraxial camera calculation."""

import numpy as np
import pytest

from validation.zshooter_xdisp.zemax_prescription import camera_focal_length


@pytest.mark.parametrize('curvature', [.01, -.01])
def test_thin_biconvex_lens_in_either_propagation_direction(curvature):
    surfaces = {
        1: {'TYPE': ['STANDARD'], 'CURV': [curvature], 'DISZ': [0], 'GLAS': ['constant']},
        2: {'TYPE': ['STANDARD'], 'CURV': [-curvature], 'DISZ': [0]},
    }
    # Sellmeier coefficient pair (1.25, 0) gives n=1.5 at every wavelength.
    glasses = {'constant': (2, np.array([1.25, 0, 0, 0, 0, 0]))}
    assert camera_focal_length(surfaces, glasses, 1, 2, .5) == pytest.approx(100)


def test_missing_camera_glass_is_not_replaced_with_a_nominal_index():
    surfaces = {1: {'TYPE': ['STANDARD'], 'CURV': [.01], 'DISZ': [0], 'GLAS': ['absent']}}
    with pytest.raises(KeyError, match='absent'):
        camera_focal_length(surfaces, {}, 1, 1, .5)
