from __future__ import annotations

import numpy as np
from astropy import units as u

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

    table = sources.read_lamp_lines(tmp_path, filenames=filenames)

    assert len(table) == 3
    np.testing.assert_allclose(table["wave"], [5000, 5001, 5002])
    np.testing.assert_allclose(table["amplitude"], [1, 2, 3])


def test_lamp_flat_requires_explicit_line_source():
    with np.testing.assert_raises_regex(ValueError, "line_dir or zshooter_dir"):
        sources.lamp_flat()


def test_numeric_wave_step_is_interpreted_as_angstrom(tmp_path, monkeypatch):
    captured = {}

    class FakeTemplates:
        Empirical1D = object()

        @staticmethod
        def SourceSpectrum(_model, *, points, lookup_table):
            captured["points"] = points
            captured["lookup_table"] = lookup_table
            return "spectrum"

        @staticmethod
        def uniform_source(spectrum, *, extent):
            return {"spectrum": spectrum, "extent": extent}

    import sys
    import types

    scopesim_module = types.ModuleType("scopesim")
    source_module = types.ModuleType("scopesim.source")
    templates_module = types.ModuleType("scopesim.source.source_templates")
    templates_module.Empirical1D = FakeTemplates.Empirical1D
    templates_module.SourceSpectrum = FakeTemplates.SourceSpectrum
    templates_module.uniform_source = FakeTemplates.uniform_source
    source_module.source_templates = templates_module
    scopesim_module.source = source_module
    monkeypatch.setitem(sys.modules, "scopesim", scopesim_module)
    monkeypatch.setitem(sys.modules, "scopesim.source", source_module)
    monkeypatch.setitem(sys.modules, "scopesim.source.source_templates", templates_module)

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


def test_slit_frame_offsets_can_anchor_first_source_on_axis():
    x, y = sources.slit_frame_offsets(
        2 * u.arcsec, 90 * u.deg, centered=False)

    np.testing.assert_allclose(x.to_value(u.arcsec), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(y.to_value(u.arcsec), [0.0, 2.0])


def test_angle_from_cmds_reads_pupil_angle_with_default():
    assert sources.angle_from_cmds({"!OBS.pupil_angle": 37.5}) == 37.5 * u.deg
    assert sources.angle_from_cmds({}, default=3 * u.deg) == 3 * u.deg


def test_two_point_source_builds_slit_frame_table(monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, *, spectra, table):
            captured["spectra"] = spectra
            captured["table"] = table
            self.meta = {}

    import sys
    import types

    scopesim_module = types.ModuleType("scopesim")
    source_pkg = types.ModuleType("scopesim.source")
    source_module = types.ModuleType("scopesim.source.source")
    templates_module = types.ModuleType("scopesim.source.source_templates")
    source_module.Source = FakeSource
    templates_module.ab_spectrum = lambda mag: f"ab:{mag}"
    source_pkg.source = source_module
    source_pkg.source_templates = templates_module
    scopesim_module.source = source_pkg
    monkeypatch.setitem(sys.modules, "scopesim", scopesim_module)
    monkeypatch.setitem(sys.modules, "scopesim.source", source_pkg)
    monkeypatch.setitem(sys.modules, "scopesim.source.source", source_module)
    monkeypatch.setitem(
        sys.modules, "scopesim.source.source_templates", templates_module)

    source = sources.two_point_source(
        separation=2 * u.arcsec,
        angle_on_slit=90 * u.deg,
        centered=False,
        mag=19,
    )

    assert source.meta["angle_on_slit"] == 90
    assert captured["spectra"] == ["ab:19"]
    np.testing.assert_allclose(captured["table"]["x"], [0, 0], atol=1e-12)
    np.testing.assert_allclose(captured["table"]["y"], [0, 2])
    assert captured["table"].meta["frame"] == "slit"


def test_field_angle_demo_sources_names_slit_frame_scenarios(monkeypatch):
    captured = []

    class FakeSource:
        def __init__(self, *, spectra, table):
            self.spectra = spectra
            self.table = table
            self.meta = {}
            captured.append(self)

    import sys
    import types

    scopesim_module = types.ModuleType("scopesim")
    source_pkg = types.ModuleType("scopesim.source")
    source_module = types.ModuleType("scopesim.source.source")
    templates_module = types.ModuleType("scopesim.source.source_templates")
    source_module.Source = FakeSource
    templates_module.ab_spectrum = lambda mag: f"ab:{mag}"
    source_pkg.source = source_module
    source_pkg.source_templates = templates_module
    scopesim_module.source = source_pkg
    monkeypatch.setitem(sys.modules, "scopesim", scopesim_module)
    monkeypatch.setitem(sys.modules, "scopesim.source", source_pkg)
    monkeypatch.setitem(sys.modules, "scopesim.source.source", source_module)
    monkeypatch.setitem(
        sys.modules, "scopesim.source.source_templates", templates_module)

    scenarios = sources.field_angle_demo_sources(
        along_separation=5 * u.arcsec,
        across_separation=0.9 * u.arcsec,
        angle_on_slit=30 * u.deg,
        along_mag=15,
        across_mag=17,
    )

    assert list(scenarios) == ["along_slit_centered", "across_slit_one_off"]
    along = scenarios["along_slit_centered"]
    across = scenarios["across_slit_one_off"]
    assert along.meta["name"] == "along_slit_centered"
    assert across.meta["name"] == "across_slit_one_off"
    assert along.meta["function_call"] == "field_angle_demo_sources"
    assert along.meta["scene_angle_on_slit"] == 30
    assert across.meta["workflow_note"].startswith("Set angle_on_slit")
    assert captured[0].spectra == ["ab:15"]
    assert captured[1].spectra == ["ab:17"]
    np.testing.assert_allclose(
        along.table["x"], [-2.5 * np.cos(np.deg2rad(30)),
                           2.5 * np.cos(np.deg2rad(30))],
    )
    np.testing.assert_allclose(
        along.table["y"], [-1.25, 1.25],
    )
    np.testing.assert_allclose(
        across.table["x"], [0.0, 0.9 * np.cos(np.deg2rad(120))],
    )
    np.testing.assert_allclose(
        across.table["y"], [0.0, 0.9 * np.sin(np.deg2rad(120))],
        atol=1e-12,
    )
