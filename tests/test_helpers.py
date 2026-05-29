from __future__ import annotations

from pathlib import Path

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
