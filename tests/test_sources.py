from __future__ import annotations

import numpy as np
from astropy import units as u
import h5py

from zs_scopesim_tools import sources


def test_gaussian_line_flux_density_preserves_integrated_flux():
    wave = np.linspace(4990.0, 5010.0, 101)
    line_centers = np.array([5000.03, 5005.2])
    line_fluxes = np.array([3.0, 2.0])
    line_fwhm = np.array([0.05, 0.1])

    flux_density = sources.gaussian_line_flux_density(
        wave, line_centers, line_fluxes, line_fwhm,
    )
    widths = np.diff(sources.bin_edges_from_centers(wave))

    np.testing.assert_allclose(
        np.sum(flux_density * widths),
        np.sum(line_fluxes),
        rtol=1e-5,
    )


def test_read_lamp_lines_combines_pipe_delimited_tables(tmp_path):
    filenames = ["a.dat", "b.dat", "c.dat"]
    for idx, filename in enumerate(filenames):
        (tmp_path / filename).write_text(
            f"wave|amplitude\n{5000 + idx}|{idx + 1}\n",
        )

    table = sources._read_lamp_lines(tmp_path, filenames=tuple(filenames))

    assert len(table) == 3
    np.testing.assert_allclose(table["wave"], [5000, 5001, 5002])
    np.testing.assert_allclose(table["amplitude"], [1, 2, 3])


def test_numeric_wave_step_is_interpreted_as_angstrom(tmp_path, monkeypatch):
    captured = {}

    class FakeTemplates:
        Empirical1D = object()

        @staticmethod
        def SourceSpectrum(_model, *, points, lookup_table, z_type):
            captured["points"] = points
            captured["lookup_table"] = lookup_table
            captured["z_type"] = z_type
            return "spectrum"

        @staticmethod
        def uniform_source(spectrum, *, extent):
            return {"spectrum": spectrum, "extent": extent}

    monkeypatch.setattr(sources, "source_templates", FakeTemplates)

    filename = "lines.dat"
    (tmp_path / filename).write_text("wave|amplitude\n5000|1\n5010|1\n")

    source = sources.lamp_flat(
        line_dir=tmp_path,
        filenames=[filename],
        wave_step=5,
    )

    assert source["spectrum"] == "spectrum"
    assert np.allclose(np.diff(captured["points"]), 5)


def test_slit_frame_offsets_uses_zero_degrees_along_slit():
    x, y = sources.slit_frame_offsets(2 * u.arcsec, 0 * u.deg)

    np.testing.assert_allclose(x.to_value(u.arcsec), [-1.0, 1.0])
    np.testing.assert_allclose(y.to_value(u.arcsec), [0.0, 0.0], atol=1e-12)


def test_angle_from_cmds_reads_pupil_angle_with_default():
    assert sources.angle_from_cmds({"!OBS.pupil_angle": 37.5}) == 37.5 * u.deg
    assert sources.angle_from_cmds({}, default=3 * u.deg) == 3 * u.deg


def test_field_angle_demo_sources_names_slit_frame_scenarios():
    scenarios = sources.field_angle_demo_sources(angle_on_slit=30 * u.deg)

    assert list(scenarios) == ["along_slit_centered", "across_slit_one_off"]
    assert scenarios["along_slit_centered"].meta["scene_angle_on_slit"] == 30
    assert scenarios["across_slit_one_off"].fields[0].field.meta["frame"] == "slit"


def write_kilonova_model(path):
    with h5py.File(path, "w") as model:
        model["nu"] = [1e14, 2e14]
        model["time"] = np.array([1.0, 3.0]) * 86400
        model["Lnu"] = [[10.0, 20.0], [30.0, 60.0]]


def test_kilonova_spectra_interpolates_phase(tmp_path, monkeypatch):
    model_file = tmp_path / "model.h5"
    write_kilonova_model(model_file)
    captured = []
    monkeypatch.setattr(sources, "empirical_spectrum", lambda wavelength, flux: captured.append((wavelength, flux)) or flux)

    spectra = sources.kilonova_spectra([model_file], phases=[2.0], redshift=0.02, combine_models=False)

    assert list(spectra) == [2.0]
    assert len(spectra[2.0]) == 1
    flux = captured[0][1]
    np.testing.assert_allclose((flux[1] / flux[0]).value, 8.0)


def test_kilonova_spectra_rejects_zero_redshift(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "redshift must be positive"):
        sources.kilonova_spectra([tmp_path / "missing.h5"], redshift=0)


def test_kilonova_spectra_rejects_phase_outside_model(tmp_path):
    model_file = tmp_path / "model.h5"
    write_kilonova_model(model_file)
    with np.testing.assert_raises_regex(ValueError, "outside the model range"):
        sources.kilonova_spectra([model_file], phases=[4.0], redshift=0.02)
