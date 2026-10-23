"""Check the folded-coordinate conversion and separate paraxial camera calculation."""

from pathlib import Path
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from validation.zshooter_xdisp.zemax_prescription import camera_focal_length
from scopesim.optics.echelle import spectrograph_factory


CHANNELS = Table.read(Path(__file__).parents[1] / 'validation/zshooter_xdisp/data/zemax_channels.csv')


@pytest.mark.parametrize('row', list(CHANNELS), ids=list(CHANNELS['channel']))
@pytest.mark.parametrize('height', [2048, 4096])
def test_folded_cross_disperser_angles_match_unfolded_planar_camera(row, height):
    direction = np.sign(row['camera_space_mm'])
    theta = np.deg2rad(row['cx_coordinate_tilt_x_deg'])
    alpha = -direction*row['cx_coordinate_tilt_x_deg']
    model = spectrograph_factory(300, 2500, 225, 18000, 65.6, row['min_order'], row['max_order'],
                                1/row['echelle_grooves_per_mm'], 4.46, 2048, height, .015,
                                xdisp_groove_length=1/row['cx_grooves_per_mm'], xdisp_beta_center=alpha, dispersion_focal_len=270)
    wave = model.edge_wave(fsr=True).ravel()
    # Zemax's two CX coordinate breaks rotate the local axes by theta.
    # The signed camera distance follows the local propagation direction.
    incidence_y = direction*np.sin(theta)
    outgoing_y = incidence_y + (wave/model.cross_disperser.d).to_value(u.dimensionless_unscaled)
    outgoing_z = direction*np.sqrt(1 - outgoing_y**2)
    camera_y = np.cos(theta)*outgoing_y + np.sin(theta)*outgoing_z
    camera_z = -np.sin(theta)*outgoing_y + np.cos(theta)*outgoing_z
    folded_height = direction*model.focal_length*camera_y/camera_z
    unfolded_height = (model.wavelength_to_y_pixel(wave) - model.detector.n_pix_y/2)*model.detector.pixel_size
    np.testing.assert_allclose(unfolded_height.to_value(u.mm), folded_height.to_value(u.mm), atol=1e-9, rtol=0)


@pytest.mark.parametrize('row', list(CHANNELS), ids=list(CHANNELS['channel']))
def test_detector_coordinate_rotation_is_inverse_point_rotation(row):
    theta = np.deg2rad(row['detector_coordinate_rotation_z_deg'])
    local_x_axis = np.array([np.cos(theta), np.sin(theta)])
    local_y_axis = np.array([-np.sin(theta), np.cos(theta)])
    point = np.array([3., 7.])  # Physical focal-plane coordinates in mm.
    detector_coordinates = np.array([point @ local_x_axis, point @ local_y_axis])
    angle = -theta
    active_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(active_rotation @ point, detector_coordinates, atol=1e-12, rtol=0)


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
