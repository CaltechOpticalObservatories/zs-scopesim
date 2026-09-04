from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from scopesim.utils import airmass2zendist

from zs_scopesim_tools import helpers


def test_configure_irdb_path_sets_scopesim_search_path(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "resolve_irdb_path", lambda fallback=None: tmp_path)
    monkeypatch.setitem(helpers.sim.rc.__config__, "!SIM.file.search_path", [])

    resolved = helpers.configure_irdb_path()

    assert resolved == str(tmp_path)
    assert helpers.sim.rc.__config__["!SIM.file.local_packages_path"] == str(tmp_path)
    assert helpers.sim.rc.__config__["!SIM.file.search_path"] == [str(tmp_path)]


def test_instrument_package_dir_uses_configured_irdb(tmp_path, monkeypatch):
    instrument = tmp_path / "ZShooter_v2"
    instrument.mkdir()
    monkeypatch.setitem(helpers.sim.rc.__config__, "!SIM.file.local_packages_path", str(tmp_path))

    assert helpers.instrument_package_dir("ZShooter_v2") == instrument


def test_warning_prevent_sync_alt_ra_dec_uses_airmass_without_radec(monkeypatch):
    def fail_get_observation_info_from_cmds(_cmd):
        raise AssertionError("airmass path should not reconstruct observation info")

    import scopesim.utils as scopesim_utils

    monkeypatch.setattr(
        scopesim_utils,
        "get_observation_info_from_cmds",
        fail_get_observation_info_from_cmds,
    )
    cmd = {"!OBS.airmass": 1.3}

    helpers.warning_prevent_sync_alt_ra_dec(cmd)

    np.testing.assert_allclose(cmd["!OBS.alt"], 90.0 - airmass2zendist(1.3))
    assert cmd["!OBS.az"] == 0.0
    assert cmd["!OBS.ra"] is None
    assert cmd["!OBS.dec"] is None


def test_warning_prevent_sync_alt_ra_dec_can_fall_back_to_current_target(monkeypatch):
    class FakeTarget:
        alt = 61.0 * u.deg
        az = 23.0 * u.deg

    def fake_get_observation_info_from_cmds(_cmd):
        return FakeTarget(), None, None

    import scopesim.utils as scopesim_utils

    monkeypatch.setattr(
        scopesim_utils,
        "get_observation_info_from_cmds",
        fake_get_observation_info_from_cmds,
    )
    cmd = {}

    helpers.warning_prevent_sync_alt_ra_dec(cmd)

    assert cmd["!OBS.alt"] == 61.0
    assert cmd["!OBS.az"] == 23.0
    assert cmd["!OBS.ra"] is None
    assert cmd["!OBS.dec"] is None
