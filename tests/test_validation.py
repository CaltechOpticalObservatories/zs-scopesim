from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM

from zs_scopesim_tools import validation as val


class ConstantCurve:
    def __init__(self, value):
        self.value = value

    def __call__(self, wave):
        return np.full(wave.size, self.value)


class ConstantEmission:
    def __init__(self, value):
        self.value = value

    def __call__(self, wave):
        return np.full(wave.size, self.value) * PHOTLAM


class FakeSurface:
    def __init__(
        self,
        transmission=1.0,
        reflection=0.0,
        emissivity=0.0,
        emission=0.0,
    ):
        self.transmission = ConstantCurve(transmission)
        self.reflection = ConstantCurve(reflection)
        self.emissivity = ConstantCurve(emissivity)
        self.emission = ConstantEmission(emission)
        self.throughput = self.transmission
        self.meta = {"temperature": 120}
        self.table = Table({"wavelength": [1.0], "transmission": [transmission]})


class FakeSurfaceList:
    def __init__(self):
        self.table = Table({
            "name": ["Pre", "Camera"],
            "action": ["transmission", "transmission"],
            "throughput_group": ["preoptics", "camera"],
            "emission_phase": ["pre_disperser", "post_disperser"],
        })
        self.surfaces = {
            "Pre": FakeSurface(transmission=0.5, emissivity=0.1, emission=100.0),
            "Camera": FakeSurface(transmission=0.8, emissivity=0.2, emission=2.0),
        }


class FakeDetectorQE:
    throughput = ConstantCurve(0.5)


class FakeDichroic:
    def __init__(self, transmission, reflection):
        self.surface = FakeSurface(transmission=transmission, reflection=reflection)


class FakeDichroicTree:
    def __init__(self):
        self.table = Table({
            "aperture_id": [0],
            "d1": ["T"],
            "d2": ["R"],
            "unused": ["X"],
        })
        self.dichroics = {
            "d1": FakeDichroic(transmission=0.8, reflection=0.1),
            "d2": FakeDichroic(transmission=0.7, reflection=0.5),
        }


class FakeTrace:
    def __init__(self, trace_id, aperture_id, image_plane_id, wave_min, wave_max):
        self.trace_id = trace_id
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.meta = {
            "trace_id": trace_id,
            "aperture_id": aperture_id,
            "image_plane_id": image_plane_id,
            "extension_id": 2,
        }


class FakeTraceList:
    def __init__(self):
        self.spectral_traces = {
            "R_2": FakeTrace("R_2", 1, 3, 0.5, 0.6),
            "B_1": FakeTrace("B_1", 0, 2, 0.3, 0.4),
        }


def test_effect_name_handles_objects_without_meta():
    obj = object()
    assert val.effect_name(obj).startswith("<object object at ")


def test_surface_group_for_row_prefers_explicit_metadata():
    row = FakeSurfaceList().table[0]
    assert val.surface_group_for_row(row) == "preoptics"


def test_emission_phase_for_row_prefers_explicit_metadata():
    row = FakeSurfaceList().table[1]
    assert val.emission_phase_for_row(row, "camera") == "post_disperser"


def test_emission_phase_for_row_falls_back_to_camera_post_disperser():
    row = Table({"name": ["VIS_Camera_1"]})[0]
    assert val.emission_phase_for_row(row, "camera") == "post_disperser"


def test_effective_diffuse_qe_uses_average_positional_qe():
    wave = np.linspace(1, 2, 4) * u.um
    spatial_map = np.array([[0.8, 1.0], [0.6, 1.0]])
    qe = val.effective_diffuse_qe(FakeDetectorQE(), wave, spatial_map)

    np.testing.assert_allclose(qe, np.full(wave.size, 0.425))


def test_surface_list_emissivity_terms_splits_phases_and_applies_qe():
    wave = np.linspace(1, 2, 4) * u.um
    qe_values = np.full(wave.size, 0.5)

    terms, counts, details = val.surface_list_emissivity_terms(
        FakeSurfaceList(), wave, qe_values=qe_values)

    np.testing.assert_allclose(
        terms["pre_disperser"]["preoptics"],
        np.full(wave.size, 0.08),
    )
    np.testing.assert_allclose(
        terms["post_disperser"]["camera"],
        np.full(wave.size, 0.1),
    )
    assert counts["pre_disperser:preoptics"] == 1
    assert counts["post_disperser:camera"] == 1
    assert details[0]["qe_applied"] is False
    assert details[1]["qe_applied"] is True


def test_surface_list_emissivity_terms_rejects_unknown_phase():
    surface_list = FakeSurfaceList()
    surface_list.table["emission_phase"][0] = "typo"
    wave = np.linspace(1, 2, 4) * u.um

    with np.testing.assert_raises_regex(ValueError, "Unknown emission_phase"):
        val.surface_list_emissivity_terms(surface_list, wave)


def test_surface_list_post_disperser_diffuse_terms_uses_post_phase_and_qe():
    wave = np.linspace(1, 2, 4) * u.um
    qe_values = np.full(wave.size, 0.5)

    spectra, details = val.surface_list_post_disperser_diffuse_terms(
        FakeSurfaceList(), wave, qe_values=qe_values,
    )

    assert list(spectra) == ["camera"]
    np.testing.assert_allclose(spectra["camera"].value, np.full(wave.size, 1.0))
    included = [row for row in details if row["included_as_diffuse"]]
    assert [row["surface"] for row in included] == ["Camera"]


def test_surface_list_post_disperser_diffuse_terms_rejects_unknown_phase():
    surface_list = FakeSurfaceList()
    surface_list.table["emission_phase"][1] = "typo"
    wave = np.linspace(1, 2, 4) * u.um

    with np.testing.assert_raises_regex(ValueError, "Unknown emission_phase"):
        val.surface_list_post_disperser_diffuse_terms(surface_list, wave)


def test_dichroic_path_throughput_uses_tree_actions():
    wave = np.linspace(1, 2, 4) * u.um

    total, components = val.dichroic_path_throughput(
        FakeDichroicTree(), aperture_id=0, wave=wave,
    )

    np.testing.assert_allclose(total, np.full(wave.size, 0.4))
    assert list(components) == ["d1:T", "d2:R"]


def test_trace_catalog_table_uses_in_memory_traces():
    table = val.trace_catalog_table(FakeTraceList())

    assert list(table["trace_id"]) == ["B_1", "R_2"]
    assert list(table["aperture_id"]) == [0, 1]
    assert list(table["image_plane_id"]) == [2, 3]
    np.testing.assert_allclose(table["wave_min_um"], [0.3, 0.5])
