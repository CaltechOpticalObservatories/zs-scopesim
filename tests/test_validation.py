from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.table import Table

from zs_scopesim_tools import validation as val


class ConstantCurve:
    def __init__(self, value):
        self.value = value

    def __call__(self, wave):
        return np.full(wave.size, self.value)


class FakeSurface:
    def __init__(self, transmission=1.0, reflection=0.0, emissivity=0.0):
        self.transmission = ConstantCurve(transmission)
        self.reflection = ConstantCurve(reflection)
        self.emissivity = ConstantCurve(emissivity)
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
            "Pre": FakeSurface(transmission=0.5, emissivity=0.1),
            "Camera": FakeSurface(transmission=0.8, emissivity=0.2),
        }


class FakeDetectorQE:
    throughput = ConstantCurve(0.5)


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
