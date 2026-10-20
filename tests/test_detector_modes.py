import numpy as np
import pytest
import astropy.units as u
import scopesim as sim

from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level

pytestmark = pytest.mark.filterwarnings('ignore::astropy.coordinates.NonRotationTransformationWarning')


@pytest.fixture(scope='module')
def detector_trains():
    configure_irdb_path()
    set_scopesim_log_level('critical')
    properties = {'!OBS.mjdobs': 60310.25, '!OBS.airmass': 1.1, '!SIM.random.seed': 736}
    return [sim.OpticalTrain(sim.UserCommands(use_instrument='ZShooter_v2', set_modes=[mode], properties=properties.copy())) for mode in ('SPEC', 'SPEC_ALTDETECT')]


def test_detector_package_changes_geometry_qe_and_operating_point_together(detector_trains):
    current, alternate = detector_trains
    for index, (manager, alt_manager) in enumerate(zip(current.detector_managers, alternate.detector_managers, strict=True)):
        native, alt_native = manager[0]._native_header, alt_manager[0]._native_header
        assert native['DETMODE'] == 'SPEC'
        assert alt_native['DETMODE'] == 'SPEC_ALTDETECT'
        assert native['WELLDEP'] == (60000 if index < 3 else 15000)
        assert alt_native['WELLDEP'] == (15000 if index == 3 else 60000)
        assert native['EXNOISE'] == (1 if index < 3 else 1.2)
        assert alt_native['EXNOISE'] == (1.2 if index == 3 else 1)
        assert alt_manager._detector_list.pixel_size == (0.018 if index > 3 else 0.015)*u.mm
        assert (native['NAXIS1'], native['NAXIS2']) == (alt_native['NAXIS1'], alt_native['NAXIS2'])
        effect = alternate['post_echelle_diffuse_background_selector'].get_effect(index)
        qe = alternate['detector_qe_selector'].get_effect(index)
        assert type(effect._detector_qe) is type(qe)
        wave = np.array([1.1, 1.6, 2.2])*u.um
        np.testing.assert_allclose(effect._detector_qe.throughput(wave), qe.throughput(wave))


def test_pitch_changes_sampling_without_moving_the_physical_prescription(detector_trains):
    current, alternate = detector_trains
    old_traces = current['trace_list_analytical']
    new_traces = alternate['trace_list_analytical']
    assert set(old_traces.spectral_traces) == set(new_traces.spectral_traces)
    for trace_id in old_traces.spectral_traces:
        old = old_traces.spectral_traces[trace_id]
        new = new_traces.spectral_traces[trace_id]
        wave = np.linspace(max(old.wave_min, new.wave_min), min(old.wave_max, new.wave_max), 9)
        for slit_position in (-4., 0., 4.):
            slit = np.full(wave.shape, slit_position)
            np.testing.assert_allclose(old.xilam2x(slit, wave), new.xilam2x(slit, wave), atol=0.00015, rtol=0)
            np.testing.assert_allclose(old.xilam2y(slit, wave), new.xilam2y(slit, wave), atol=0.00015, rtol=0)
