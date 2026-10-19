from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scopesim.effects.electronic import ADConversion, ExposureOutput, ExposureIntegration, DarkCurrent, BasicReadoutNoise

from zs_scopesim_tools.helpers import electron_readouts, combine_readouts, save_zshooter_readout, subtract_hduls


def readout(value, variance=None, exptime=300., ndit=1):
    header = fits.Header({'BUNIT': 'electron', 'GAIN': 1., 'DIT': exptime/ndit, 'NDIT': ndit, 'EXPTIME': exptime, 'RON': 2., 'DARK': .01})
    hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(np.full((2, 3), value), header=header)])
    if variance is not None:
        hdul.append(fits.ImageHDU(np.full((2, 3), variance), name='VARIANCE'))
    return hdul


@pytest.mark.parametrize('average', [False, True])
@pytest.mark.parametrize('dit', [0., 30.])
def test_electron_conversion_preserves_mean_and_accumulated_variance(monkeypatch, average, dit):
    from zs_scopesim_tools import comparison
    ndit, gain, rate, dark, ron = 4, 2.3, 7., .02, 3.
    effects = [ExposureIntegration(dit=dit, ndit=ndit), DarkCurrent(value=dark, dit=dit, ndit=ndit), BasicReadoutNoise(noise_std=ron, ndit=ndit), ADConversion(gain=gain, dtype='float64')]
    if average:
        effects.append(ExposureOutput(mode='average', dit=dit, ndit=ndit))
    train = SimpleNamespace(detector_managers=[[SimpleNamespace(det_id=0)]], optics_manager=SimpleNamespace(detector_effects=effects), cmds={})
    monkeypatch.setattr(comparison, 'detector_rates', lambda train: [np.full((2, 3), rate)])
    charge = (rate + dark) * dit * ndit
    original = readout(charge / gain / (ndit if average else 1), exptime=dit*ndit, ndit=ndit)
    result, = electron_readouts(train, [original])
    np.testing.assert_allclose(result[1].data, charge)
    np.testing.assert_allclose(result['VARIANCE'].data, charge + ndit*ron**2)
    assert result[1].header['EXPTIME'] == dit*ndit
    assert result[1].header['RON'] == np.sqrt(ndit)*ron
    assert result[1].header['GAIN'] == 1
    assert result[1].header['COMBINE'] == 'sum'
    assert result[1].header['DARK'] == dark
    assert 'VARIANCE' not in original


def test_shared_sky_coefficient_is_squared_before_extraction():
    groups = [[readout(20., 25.)], [readout(30., 35.)], [readout(8., 12.)]]
    combined, = combine_readouts(groups, [1, 1, -2], source_indices=[0, 1])
    np.testing.assert_array_equal(combined[1].data, 34.)
    np.testing.assert_array_equal(combined['VARIANCE'].data, 25+35+4*12)
    assert combined[1].header['EXPTIME'] == 600.
    assert combined[1].header['NDIT'] == 2
    assert combined[1].header['RON']**2 == pytest.approx(24.)


def test_missing_variance_does_not_manufacture_an_uncertainty():
    with pytest.raises(KeyError):
        combine_readouts([[readout(10.)], [readout(4.)]], [1, -1], source_indices=[0])
    difference, = subtract_hduls([readout(10.)], [readout(4.)])
    assert len(difference) == 2
    np.testing.assert_array_equal(difference[1].data, 6.)


def test_source_average_is_not_mislabeled_as_accumulated_exposure():
    with pytest.raises(ValueError, match='coefficient'):
        combine_readouts([[readout(10., 12.)]], [.5], source_indices=[0])


def test_source_cadence_is_retained_when_shorter_sky_is_listed_first():
    groups = [[readout(8., 12., exptime=150.)], [readout(20., 25., exptime=300.)]]
    combined, = combine_readouts(groups, [-2, 1], source_indices=[1])
    np.testing.assert_array_equal(combined[1].data, 4.)
    np.testing.assert_array_equal(combined['VARIANCE'].data, 73.)
    for header in (combined[0].header, combined[1].header):
        assert header['DIT'] == header['EXPTIME'] == 300.
        assert header['NDIT'] == 1


def test_mixed_source_cadences_do_not_manufacture_a_single_dit():
    groups = [[readout(20., 25., exptime=300.)], [readout(40., 50., exptime=600.)]]
    with pytest.raises(ValueError, match='same source integration time'):
        combine_readouts(groups, [1, 1], source_indices=[0, 1])


def test_save_preserves_prepared_total_exposure_and_noise(tmp_path):
    science = readout(10., 12., exptime=3600., ndit=12)
    science[0].header['RON'] = science[1].header['RON'] = 6.*np.sqrt(12)
    path, = save_zshooter_readout([science], tmp_path, object_name='test', train=None)
    with fits.open(path) as saved:
        assert saved[1].header['EXPTIME'] == 3600.
        assert saved[0].header['RON'] == pytest.approx(6.*np.sqrt(12))
        assert saved[0].header['DATE-OBS'] != '2026-09-01T00:00:00.000'
        np.testing.assert_array_equal(saved['VARIANCE'].data, 12.)
