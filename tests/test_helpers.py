from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from scopesim.utils import airmass2zendist

from zs_scopesim_tools import helpers


def test_zshooter_package_dir_points_inside_irdb_checkout(tmp_path):
    assert helpers.zshooter_package_dir(tmp_path) == tmp_path / "ZShooter_v2"


def test_validation_work_dir_uses_base_dir_outside_irdb(tmp_path):
    output_dir = helpers.validation_work_dir(
        irdb_path=tmp_path / "irdb",
        name="out",
        base_dir=tmp_path / "work",
    )

    assert output_dir == tmp_path / "work" / "out"
    assert output_dir.is_dir()


def test_validation_work_dir_avoids_writing_inside_irdb(tmp_path, monkeypatch):
    irdb_path = tmp_path / "irdb"
    package_dir = irdb_path / "ZShooter_v2"
    package_dir.mkdir(parents=True)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(helpers, "repo_root", lambda start=None: repo_dir)

    output_dir = helpers.validation_work_dir(
        irdb_path=irdb_path,
        name="out",
        base_dir=package_dir,
    )

    assert output_dir.name == "out"
    assert output_dir == repo_dir / "out"
    assert Path("irdb") not in output_dir.relative_to(repo_dir).parents


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
