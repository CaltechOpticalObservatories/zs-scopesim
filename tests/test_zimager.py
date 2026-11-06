"""Exercise the configured imaging branches and per-arm detector readouts."""

import numpy as np
import pytest
import astropy.units as u
from synphot import SourceSpectrum, ConstFlux1D
import scopesim as sim
from scopesim.utils import from_currsys

from zs_scopesim_tools.helpers import configure_irdb_path, disable_scopesim_progress_bars


def imaging_train(filter_name="g", small=True):
    configure_irdb_path()
    commands = sim.UserCommands(use_instrument="ZShooter_v2", set_modes=["IMAGER"])
    commands["!INST.imager_filter_v"] = filter_name
    commands["!SIM.random.seed"] = 948
    # The branch and detector checks do not need a network sky or a seeing convolution.
    for document in commands.yaml_dicts:
        if "effects" in document:
            document["effects"] = [effect for effect in document["effects"] if effect["class"] not in ["TAPASTERCurve", "SkyBackgroundTERCurve", "PalaceAirglowEmission", "AOEnhanceablePSF", "ADCShift"]]
            if small:
                for effect in document["effects"]:
                    if effect["class"] == "DetectorList":
                        effect["kwargs"]["array_dict"]["x_size"] = [128]
                        effect["kwargs"]["array_dict"]["y_size"] = [128]
    return sim.OpticalTrain(commands)


def test_imager_detector_geometry_comes_from_irdb():
    train = imaging_train(small=False)
    for arm, manager in zip(("u", "v", "z"), train.detector_managers, strict=True):
        detector, = manager
        configured = train[f"detector_{arm}"].table[0]
        header = detector.header
        assert (header["NAXIS2"], header["NAXIS1"]) == (configured["y_size"], configured["x_size"])
        assert header["CDELT1D"] == pytest.approx(configured["pixel_size"])
        assert header["CDELT2D"] == pytest.approx(configured["pixel_size"])
        assert header["BUNIT"] == "electron"
        assert header["DETMODE"] == "IMAGER"


def test_independent_exposure_noise_and_repeated_readouts():
    train = imaging_train()
    rates = (0., .5, 4.)  # Pure dark, then two uniform source/sky rates in electron/s/pixel.
    for plane, rate in zip(train.image_planes, rates, strict=True):
        plane.data.fill(rate)
    exposures = {"imager_dit_u": 100, "imager_ndit_u": 3, "imager_dit_v": 200, "imager_ndit_v": 2, "imager_dit_z": 300, "imager_ndit_z": 1}
    first = train.readout(roid=0, **exposures)
    second = train.readout(roid=1, **exposures)
    replay = train.readout(roid=0, **exposures)
    for i, arm in enumerate(("u", "v", "z")):
        dit, ndit = exposures[f"imager_dit_{arm}"], exposures[f"imager_ndit_{arm}"]
        dark = from_currsys(train["dark_current_selector"].get_effect(i).meta["value"], train.cmds)
        rn = from_currsys(train["readout_noise_selector"].get_effect(i).meta["noise_std"], train.cmds)
        bias = train["bias_selector"].get_effect(i).bias_level
        expected_mean = (rates[i] + dark) * dit * ndit + bias
        expected_variance = (rates[i] + dark) * dit * ndit + rn**2 * ndit
        data = first[i][1].data
        sampling_error = np.sqrt(expected_variance / data.size)
        assert abs(data.mean() - expected_mean) < 6 * sampling_error
        assert abs(data.var() - expected_variance) < .08 * expected_variance
        assert first[i][1].header["DIT"] == dit
        assert first[i][1].header["NDIT"] == ndit
        assert first[i][1].header["EXPTIME"] == dit * ndit
        assert first[i][1].header["PEAKDIT"] == pytest.approx((rates[i] + dark) * dit)
        np.testing.assert_array_equal(data, replay[i][1].data)
        assert not np.array_equal(data, second[i][1].data)


def test_exposure_limit_warns_without_clipping_or_changing_exposure():
    train = imaging_train()
    well = train.cmds["!DET.full_well"]
    for plane in train.image_planes:
        plane.data.fill(well)
    train["shot_noise"].include = False
    train["readout_noise_selector"].include = False
    exposures = {f"imager_{quantity}_{arm}": value for arm in ("u", "v", "z") for quantity, value in (("dit", 2), ("ndit", 3))}
    with pytest.warns(UserWarning, match="exceeds usable well") as warnings:
        readouts = train.readout(**exposures)
    assert len(warnings) == 3
    for readout in readouts:
        assert readout[1].header["DIT"] == 2
        assert readout[1].header["NDIT"] == 3
        assert readout[1].data.min() > 6 * well


@pytest.mark.parametrize("filter_name", ["r", "i"])
def test_filter_round_trip_matches_fresh_train(filter_name):
    train = imaging_train()
    source = sim.Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[SourceSpectrum(ConstFlux1D, amplitude=20*u.ABmag)])
    with disable_scopesim_progress_bars():
        train.observe(source, update=False)
        initial = [plane.data.copy() for plane in train.image_planes]
        train.cmds["!INST.imager_filter_v"] = filter_name
        train.update(migrate_cmds=True)
        train.observe(source, update=False)
        fresh = imaging_train(filter_name)
        fresh.observe(source, update=False)
        for reused_plane, fresh_plane in zip(train.image_planes, fresh.image_planes, strict=True):
            np.testing.assert_array_equal(reused_plane.data, fresh_plane.data)
        for i in (0, 2):
            np.testing.assert_array_equal(train.image_planes[i].data, initial[i])
        assert not np.allclose(train.image_planes[1].data, initial[1])
        train.cmds["!INST.imager_filter_v"] = "g"
        train.update(migrate_cmds=True)
        train.observe(source, update=False)
        for plane, reference in zip(train.image_planes, initial, strict=True):
            np.testing.assert_array_equal(plane.data, reference)
