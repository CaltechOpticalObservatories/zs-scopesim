import numpy as np
import pytest
import astropy.units as u
import scopesim as sim
from scopesim.utils import from_currsys

from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level

pytestmark = pytest.mark.filterwarnings('ignore::astropy.coordinates.NonRotationTransformationWarning')


@pytest.fixture(scope='module')
def detector_trains():
    configure_irdb_path()
    set_scopesim_log_level('critical')
    properties = {'!OBS.mjdobs': 60310.25, '!OBS.airmass': 1.1, '!SIM.random.seed': 736}
    return [sim.OpticalTrain(sim.UserCommands(use_instrument='ZShooter_v2', set_modes=[mode], properties=properties.copy())) for mode in ('SPEC', 'SPEC_ALTDETECT')]


def test_configured_detector_properties_reach_effects_and_headers(detector_trains):
    for train in detector_trains:
        for manager in train.detector_managers:
            detector, = manager
            native = detector._native_header
            detector_list = manager._detector_list
            arm = detector_list.meta['name'].removeprefix('detector_')
            expected = {
                'DETMODE': 'detector_mode', 'DETTYPE': f'detector_type_{arm}',
                'WELLDEP': f'usable_well_{arm}', 'EXNOISE': f'excess_noise_factor_{arm}', 'AVGAIN': f'avalanche_gain_{arm}',
                'PIXEL_SIZE': f'pixel_size_{arm}',
            }
            for keyword, setting in expected.items():
                assert native[keyword] == train.cmds[f'!DET.{setting}']
            assert detector_list.pixel_size == train.cmds[f'!DET.pixel_size_{arm}']*u.mm
            assert np.isclose(abs(native['CDELT1D']), detector_list.pixel_size.to_value(u.mm))

            dark = train['dark_current_selector'].get_effect(detector.det_id)
            read = train['readout_noise_selector'].get_effect(detector.det_id)
            shot = train['shot_noise'].get_effect(detector.det_id)
            assert dark._get_dark_level(detector.det_id) == train.cmds[f'!DET.dark_{arm}']
            assert from_currsys(read.meta['noise_std'], train.cmds) == train.cmds[f'!DET.ron_{arm}']
            if isinstance(shot, sim.effects.AvalancheShotNoise):
                assert shot.excess_noise_factor == train.cmds[f'!DET.excess_noise_factor_{arm}']
            else:
                assert train.cmds[f'!DET.excess_noise_factor_{arm}'] == 1  # Ordinary Poisson variance equals its mean.

            effect = train['post_echelle_diffuse_background_selector'].get_effect(detector.det_id)
            qe = train['detector_qe_selector'].get_effect(detector.det_id)
            assert type(qe).__name__ == train.cmds[f'!DET.qe_class_{arm}']
            assert type(effect._detector_qe) is type(qe)
            wave = np.array([1.1, 1.6, 2.2])*u.um
            np.testing.assert_allclose(effect._detector_qe.throughput(wave), qe.throughput(wave))


def test_pitch_changes_sampling_without_moving_the_physical_prescription(detector_trains):
    current, alternate = detector_trains
    old_traces = current['trace_list_analytical']
    new_traces = alternate['trace_list_analytical']
    tolerance_mm = .01 * min(manager._detector_list.pixel_size.to_value(u.mm) for train in detector_trains for manager in train.detector_managers)
    assert set(old_traces.spectral_traces) == set(new_traces.spectral_traces)
    for trace_id in old_traces.spectral_traces:
        old = old_traces.spectral_traces[trace_id]
        new = new_traces.spectral_traces[trace_id]
        wave = np.linspace(max(old.wave_min, new.wave_min), min(old.wave_max, new.wave_max), 9)
        for slit_position in (-4., 0., 4.):
            slit = np.full(wave.shape, slit_position)
            np.testing.assert_allclose(old.xilam2x(slit, wave), new.xilam2x(slit, wave), atol=tolerance_mm, rtol=0)
            np.testing.assert_allclose(old.xilam2y(slit, wave), new.xilam2y(slit, wave), atol=tolerance_mm, rtol=0)
