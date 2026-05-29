from __future__ import annotations

import numpy as np

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
