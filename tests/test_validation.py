from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from synphot import Empirical1D, SourceSpectrum
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


class FakeUntaggedSurfaceList:
    def __init__(self):
        self.table = Table({
            "name": ["M1", "M2"],
            "action": ["reflection", "reflection"],
        })
        self.surfaces = {
            "M1": FakeSurface(reflection=0.9, emissivity=0.1, emission=1.0),
            "M2": FakeSurface(reflection=0.8, emissivity=0.2, emission=2.0),
        }
        self.meta = {"name": "telescope_reflection"}


class FakeADCSurfaceList(FakeSurfaceList):
    def __init__(self):
        super().__init__()
        self.table = Table({
            "name": ["VIS_ADC_12", "VIS_ADC_34"],
            "action": ["transmission", "transmission"],
            "throughput_group": ["preoptics", "preoptics"],
            "emission_phase": ["pre_disperser", "pre_disperser"],
        })


class FakeEffectWithMissingTable:
    table = None


class FakeDetectorQE:
    meta = {"name": "fake_detector_qe", "filename": "QE_fake.dat"}
    throughput = ConstantCurve(0.5)


class FakeTaperedQuantumEfficiency:
    uses_detector_footprint = True

    def __init__(self):
        self.meta = {
            "name": "fake_tapered_qe",
            "center_wave_min": 0.3,
            "center_wave_max": 0.4,
            "flat_width": 0.06,
            "transition_width": 0.18,
            "peak": 0.95,
        }
        self.footprints = []

    def throughput(self, wave):
        return np.full(wave.size, 0.6)

    def effective_diffuse_throughput(self, wave, footprint=None):
        self.footprints.append(footprint)
        return np.full(wave.size, 0.7)


class FakeTEREffect:
    include = True

    def __init__(
        self,
        transmission=1.0,
        reflection=0.0,
        emissivity=0.0,
        emission=0.0,
        **meta,
    ):
        self.meta = {
            "name": "fake_ter",
            "action": "transmission",
            "throughput_group": "fake_ter",
            "emission_phase": "none",
            **meta,
        }
        self.surface = FakeSurface(
            transmission=transmission,
            reflection=reflection,
            emissivity=emissivity,
            emission=emission,
        )


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


class ConstantGradient:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def gradient(self):
        return (
            lambda x, y: np.full(np.asarray(x).shape, self.dx, dtype=float),
            lambda x, y: np.full(np.asarray(x).shape, self.dy, dtype=float),
        )


class FakeGeometryTrace:
    trace_id = "B_1"
    wave_min = 1.0
    wave_max = 1.1

    def __init__(self):
        self.meta = {
            "trace_id": self.trace_id,
            "aperture_id": 0,
            "image_plane_id": 0,
            "extension_id": 2,
            "nominal_fwhm_pix": 4.0,
            "nominal_slit_width": 0.7,
            "design_res": 20000.0,
        }
        self.xy2lam = ConstantGradient(1.0, 0.0)

    def xilam2x(self, xi, lam):
        return np.asarray(lam, dtype=float)

    def xilam2y(self, xi, lam):
        return np.asarray(xi, dtype=float) / 10.0


class FakeGeometryTraceList:
    def __init__(self):
        self.spectral_traces = {"B_1": FakeGeometryTrace()}


class FakeTraceEfficiency:
    include = True
    display_name = "trace_eff_analytical"
    meta = {"name": display_name}

    def efficiency_generator(self, trace_id, wave):
        return np.full(wave.size, 0.9)


class FakeImagePlane:
    def __init__(self, header):
        self.header = header


class FakeTrainWithImagePlane:
    cmds = {"!INST.plate_scale": 10.0}

    def __init__(self):
        header = fits.Header({
            "CDELT1D": 0.015,
            "CUNIT1D": "mm",
            "CDELT2D": 0.015,
            "CUNIT2D": "mm",
        })
        self.image_planes = [FakeImagePlane(header)]


class FakeDiffuseEffect:
    include = True
    meta = {
        "filename": "optics/LIST_fake.dat",
        "detector_qe_filename": "detector_specs/QE_fake.dat",
    }

    def __init__(self, rate):
        self.rate = rate

    def background_value(self, image_plane):
        return self.rate


class FakeConfiguredPostDiffuseEffect:
    include = True
    meta = {
        "name": "configured_post_diffuse",
        "filename": "optics/LIST_fake.dat",
        "emission_phase": "post_disperser",
    }

    def __init__(self, downstream=0.25):
        self._surface_list = FakeSurfaceList()
        self._detector_qe = FakeDetectorQE()
        self._positional_qe = None
        self.downstream = downstream

    def _downstream_throughput_values(self, wave):
        return np.full(wave.size, self.downstream)


class FakeSelector:
    display_name = "post_echelle_diffuse_background_selector"
    include = True
    meta = {"name": display_name}

    def __init__(self):
        self.wheel_effects = {2: FakeDiffuseEffect(1.0)}


class FakeOpticsManager:
    def __init__(self):
        self.all_effects = [FakeSelector()]


class FakeTrainWithDiffuseEffect(FakeTrainWithImagePlane):
    def __init__(self):
        super().__init__()
        self.image_planes = [None, None, *self.image_planes]
        self.optics_manager = FakeOpticsManager()


class DetectorList:
    include = True

    def __init__(self):
        self.meta = {
            "name": "detector_b",
            "image_plane_id": 0,
            "detector": "CCD_B",
        }
        self.table = Table({
            "id": [0],
            "x_size": [2],
            "y_size": [2],
            "pixel_size": [0.015],
            "gain": [1.0],
        })


class FakeSelectedEffect:
    include = True

    def __init__(self, **meta):
        self.meta = meta


class FakeNamedSelector:
    def __init__(self, name, selector_key, effects, *, include=True):
        self.include = include
        self.display_name = name
        self.meta = {"name": name, "selector_key": selector_key}
        self.wheel_effects = effects


class FakePSFEffect:
    include = True
    display_name = "seeing_psf"
    meta = {"name": display_name, "fwhm": "!OBS.seeing"}


class AOEnhanceablePSF:
    include = True
    display_name = "seeing_psf"
    alpha = 3.25

    def __init__(self, scale=0.2, *, is_absolute=True, fwhm=0.6):
        self.scale = scale
        self.fwhm_arcsec = fwhm
        self.seen_wave_units = []
        self.seen_fwhm_units = []
        self.meta = {
            "name": self.display_name,
            "is_absolute": is_absolute,
        }

    def fwhm(self, wave):
        self.seen_fwhm_units.append(u.Quantity(wave).unit)
        return np.full(wave.size, self.fwhm_arcsec) * u.arcsec

    def ao_scale(self, wave):
        self.seen_wave_units.append(u.Quantity(wave).unit)
        return np.full(wave.size, self.scale)


def named_effect(effect, name, *, include=True):
    effect.include = include
    effect.display_name = name
    meta = getattr(effect, "meta", {}) or {}
    meta.setdefault("name", name)
    effect.meta = meta
    return effect


class FakeScienceOpticsManager:
    def __init__(self, qe_selectors, optical_selectors):
        self.all_effects = [
            named_effect(FakeDichroicTree(), "dichroic_tree"),
            FakeNamedSelector(
                "channel_optics_selector",
                "aperture_id",
                {0: FakeSurfaceList()},
            ),
            *optical_selectors,
            *qe_selectors,
            named_effect(FakeTraceList(), "trace_list_analytical"),
            FakeTraceEfficiency(),
        ]


class FakeScienceTrain:
    def __init__(self, qe_selectors, optical_selectors=()):
        self.cmds = {
            "!TEL.area": "1 m2",
            "!INST.plate_scale": 10.0,
        }
        header = fits.Header({
            "CDELT1D": 0.015,
            "CUNIT1D": "mm",
            "CDELT2D": 0.015,
            "CUNIT2D": "mm",
        })
        self.image_planes = [None, None, FakeImagePlane(header)]
        self.optics_manager = FakeScienceOpticsManager(
            qe_selectors, optical_selectors,
        )


class FakeBudgetOpticsManager:
    def __init__(self):
        self.all_effects = [
            DetectorList(),
            FakeNamedSelector(
                "exposure_integration_selector",
                "detector_id",
                {0: FakeSelectedEffect(dit=10.0, ndit=3)},
            ),
            FakeNamedSelector(
                "dark_current_selector",
                "detector_id",
                {0: FakeSelectedEffect(value=0.1, dit=10.0, ndit=3)},
            ),
            FakeNamedSelector(
                "readout_noise_selector",
                "detector_id",
                {0: FakeSelectedEffect(noise_std=5.0, ndit=3)},
            ),
            FakeNamedSelector(
                "bias_selector",
                "detector_id",
                {0: FakeSelectedEffect(bias=1040.0)},
            ),
        ]


class FakeBudgetTrain(FakeTrainWithImagePlane):
    cmds = {}

    def __init__(self):
        super().__init__()
        self.optics_manager = FakeBudgetOpticsManager()


class FakeAOOpticsManager:
    def __init__(self, effect=None):
        self.all_effects = [
            FakeNamedSelector(
                "slitwheel_selector",
                "aperture_id",
                {
                    0: FakeSelectedEffect(
                        current_slit="!INST.vis_curr_slit",
                        slit_names=[1.25, 0.7, 0.5, 0.33],
                    ),
                    3: FakeSelectedEffect(
                        current_slit="!INST.nir_curr_slit",
                        slit_names=[1.25, 0.7, 0.5, 0.33],
                    ),
                },
            ),
            effect or AOEnhanceablePSF(),
        ]


class FakeAOTrain:
    def __init__(self, effect=None):
        self.cmds = {
            "!OBS.airmass": 1.3,
            "!OBS.seeing": 0.6,
            "!ATMO.temperature": 9.0,
            "!ATMO.pressure": 0.75,
            "!ATMO.humidity": 0.15,
            "!ATMO.x_co2": 450.0,
            "!INST.vis_curr_slit": 0.7,
        }
        self.optics_manager = FakeAOOpticsManager(effect)


def test_effect_name_handles_objects_without_meta():
    obj = object()
    assert val.effect_name(obj).startswith("<object object at ")


def test_surface_group_for_row_prefers_explicit_metadata():
    row = FakeSurfaceList().table[0]
    assert val.surface_group_for_row(row) == "preoptics"


def test_emission_phase_for_row_prefers_explicit_metadata():
    row = FakeSurfaceList().table[1]
    assert val.emission_phase_for_row(row) == "post_disperser"


def test_emission_phase_for_row_normalizes_common_aliases():
    row = Table({"name": ["Filter"], "emission_phase": ["postdisperser"]})[0]
    assert val.emission_phase_for_row(row) == "post_disperser"


def test_surface_group_for_row_rejects_missing_metadata():
    row = Table({"name": ["VIS_Camera_1"], "emission_phase": ["post_disperser"]})[0]

    with np.testing.assert_raises_regex(ValueError, "throughput_group"):
        val.surface_group_for_row(row)


def test_emission_phase_for_row_rejects_missing_metadata():
    row = Table({"name": ["VIS_Camera_1"], "throughput_group": ["camera"]})[0]

    with np.testing.assert_raises_regex(ValueError, "emission_phase"):
        val.emission_phase_for_row(row)


def test_optical_surface_rows_accepts_explicit_missing_component_group():
    wave = np.linspace(1, 2, 4) * u.um
    effect = FakeTEREffect(transmission=0.5, emission_phase="post_disperser")
    del effect.meta["throughput_group"]
    selector = FakeNamedSelector("untagged_filter_selector", "aperture_id", {0: effect})
    components = [{
        "selector": selector,
        "selector_name": "untagged_filter_selector",
        "effect": effect,
    }]

    rows = val.optical_surface_rows(
        components,
        wave,
        component_metadata={
            "untagged_filter_selector": {"throughput_group": "blocking_filter"},
        },
    )

    assert rows[0]["group"] == "blocking_filter"
    assert rows[0]["emission_phase"] == "post_disperser"


def test_optical_surface_rows_rejects_conflicting_component_group():
    wave = np.linspace(1, 2, 4) * u.um
    effect = FakeTEREffect(transmission=0.5, throughput_group="configured")
    selector = FakeNamedSelector("tagged_filter_selector", "aperture_id", {0: effect})
    components = [{
        "selector": selector,
        "selector_name": "tagged_filter_selector",
        "effect": effect,
    }]

    with np.testing.assert_raises_regex(ValueError, "conflicts"):
        val.optical_surface_rows(
            components,
            wave,
            component_metadata={
                "tagged_filter_selector": {"throughput_group": "override"},
            },
        )


def test_channel_optical_components_includes_static_surface_lists():
    static_surface_list = named_effect(
        FakeUntaggedSurfaceList(), "telescope_reflection",
    )
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[static_surface_list],
    )

    components = val.channel_optical_components(train, aperture_id=0)

    names = [component["selector_name"] for component in components]
    assert "telescope_reflection" in names
    assert "channel_optics_selector" in names


def test_optical_surface_rows_accepts_static_surface_list_metadata():
    wave = np.linspace(1, 2, 4) * u.um
    surface_list = named_effect(
        FakeUntaggedSurfaceList(), "telescope_reflection",
    )
    components = [{
        "selector": surface_list,
        "selector_name": "telescope_reflection",
        "effect": surface_list,
    }]

    rows = val.optical_surface_rows(
        components,
        wave,
        component_metadata={
            "telescope_reflection": {
                "throughput_group": "telescope",
                "emission_phase": "pre_disperser",
            },
        },
    )

    assert [row["surface_name"] for row in rows] == ["M1", "M2"]
    assert {row["group"] for row in rows} == {"telescope"}
    assert {row["emission_phase"] for row in rows} == {"pre_disperser"}


def test_effective_diffuse_qe_uses_average_positional_qe():
    wave = np.linspace(1, 2, 4) * u.um
    spatial_map = np.array([[0.8, 1.0], [0.6, 1.0]])
    qe = val.effective_diffuse_qe(FakeDetectorQE(), wave, spatial_map)

    np.testing.assert_allclose(qe, np.full(wave.size, 0.425))


def test_effective_diffuse_qe_uses_effect_footprint_average():
    wave = np.linspace(1, 2, 4) * u.um
    footprint = object()
    detector_qe = FakeTaperedQuantumEfficiency()

    qe = val.effective_diffuse_qe(
        detector_qe, wave, footprint=footprint,
    )

    np.testing.assert_allclose(qe, np.full(wave.size, 0.7))
    assert detector_qe.footprints == [footprint]


def test_build_transmission_sanity_data_can_auto_select_enabled_qe():
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain([
        FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
            include=False,
        ),
        FakeNamedSelector(
            "tapered_detector_qe_selector",
            "aperture_id",
            {0: FakeTaperedQuantumEfficiency()},
        ),
    ])

    data = val.build_transmission_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    channel = data["channels"][0]
    np.testing.assert_allclose(channel["detector_qe"], [0.7, 0.7])
    np.testing.assert_allclose(channel["detector_qe_midpoint"], [0.6, 0.6])
    np.testing.assert_allclose(
        channel["pre_disperser_instrument_total"], [0.16, 0.16],
    )
    np.testing.assert_allclose(channel["pre_disperser_total"], [0.16, 0.16])
    order = next(iter(channel["orders"].values()))
    np.testing.assert_allclose(order["instrument"], order["total"])
    np.testing.assert_allclose(
        order["total_with_telescope_no_slit"],
        order["total"],
    )
    assert channel["order_detector_qe_methods"] == ["spectral throughput"]


def test_build_transmission_sanity_data_skips_orders_outside_wave_grid():
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain([
        FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        ),
    ])
    trace_list = val.get_effect(train, "trace_list_analytical")
    trace_list.spectral_traces["B_0"] = FakeTrace("B_0", 0, 2, 0.30, 0.31)

    data = val.build_transmission_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    channel = data["channels"][0]
    assert "B_0" not in channel["orders"]
    assert "B_1" in channel["orders"]
    val.validate_transmission_sanity_data(data)


def test_build_transmission_sanity_data_includes_extra_optical_selector():
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[FakeNamedSelector(
            "ir_blocking_filter_selector",
            "aperture_id",
            {0: FakeTEREffect(
                transmission=0.25,
                throughput_group="ir_blocking_filter",
            )},
        )],
    )

    data = val.build_transmission_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    channel = data["channels"][0]
    np.testing.assert_allclose(channel["optics_groups"]["preoptics"], [0.5, 0.5])
    np.testing.assert_allclose(
        channel["optics_groups"]["ir_blocking_filter"], [0.25, 0.25],
    )
    np.testing.assert_allclose(channel["pre_disperser_total"], [0.04, 0.04])


def test_build_emissivity_sanity_data_accepts_non_surface_qe():
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain([
        FakeNamedSelector(
            "tapered_detector_qe_selector",
            "aperture_id",
            {0: FakeTaperedQuantumEfficiency()},
        ),
    ])

    data = val.build_emissivity_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    channel = data["channels"][0]
    np.testing.assert_allclose(channel["detector_qe"], [0.7, 0.7])
    np.testing.assert_allclose(
        channel["post_disperser_after_qe"], [1.4, 1.4],
    )
    details = data["details"]
    qe_rows = details[np.asarray(details["group"], dtype=str) == "detector_qe"]
    assert len(qe_rows) == 1
    assert qe_rows[0]["effect_class"] == "FakeTaperedQuantumEfficiency"
    assert qe_rows[0]["transmission_source"] == "configured"


def test_build_emissivity_sanity_data_applies_downstream_extra_selector(monkeypatch):
    wave_nm = np.array([350.0, 360.0]) * u.nm
    extra_selector = FakeNamedSelector(
        "ir_blocking_filter_selector",
        "aperture_id",
        {0: FakeTEREffect(
            transmission=0.25,
            throughput_group="ir_blocking_filter",
            emission_phase="post_disperser",
        )},
    )
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[extra_selector],
    )
    monkeypatch.setattr(val, "_telescope_area", lambda _ztrain: 1 * u.m**2)
    monkeypatch.setattr(
        val, "_image_plane_pixel_area", lambda _ztrain, _id: 1 * u.arcsec**2,
    )

    data = val.build_emissivity_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    channel = data["channels"][0]
    np.testing.assert_allclose(
        channel["post_disperser_after_qe"], [0.25, 0.25],
    )
    np.testing.assert_allclose(
        channel["post_disperser_without_blocking_after_qe"], [1.0, 1.0],
    )
    np.testing.assert_allclose(channel["post_disperser_blocked_delta"], [0.75, 0.75])
    np.testing.assert_allclose(
        channel["post_disperser_extract_equiv_rate_ph_s"],
        [2.5e5, 2.5e5],
    )
    details = data["details"]
    assert "ir_blocking_filter" in set(details["group"])
    filter_rows = details[
        np.asarray(details["group"], dtype=str) == "ir_blocking_filter"
    ]
    assert filter_rows[0]["emission_phase"] == "post_disperser"
    assert filter_rows[0]["peak_output_thermal_emission"] == 0.0


def test_post_disperser_diffuse_data_applies_downstream_extra_selector(monkeypatch):
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[FakeNamedSelector(
            "ir_blocking_filter_selector",
            "aperture_id",
            {0: FakeTEREffect(
                transmission=0.25,
                throughput_group="ir_blocking_filter",
                emission_phase="post_disperser",
            )},
        )],
    )
    train.image_planes = [None, None, FakeImagePlane(fits.Header())]
    monkeypatch.setattr(val, "_image_plane_pixel_area", lambda _ztrain, _id: 1 * u.arcsec**2)
    monkeypatch.setattr(val, "_telescope_area", lambda _ztrain: 1 * u.m**2)

    data = val.build_post_disperser_diffuse_background_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    spectrum = data["channels"][0]["spectra"]["camera"]
    np.testing.assert_allclose(spectrum.value, [0.25, 0.25])
    unblocked = data["channels"][0]["spectra_without_blocking"]["camera"]
    np.testing.assert_allclose(unblocked.value, [1.0, 1.0])
    assert data["channels"][0]["total_rate_without_blocking_ph_s_pix"] > (
        data["channels"][0]["total_rate_ph_s_pix"]
    )


def test_post_disperser_diffuse_data_prefers_configured_effect_downstream(
    monkeypatch,
):
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[FakeNamedSelector(
            "post_echelle_diffuse_background_selector",
            "image_plane_id",
            {2: FakeConfiguredPostDiffuseEffect(downstream=0.25)},
        )],
    )
    monkeypatch.setattr(
        val,
        "_image_plane_pixel_area",
        lambda _ztrain, _id: 1 * u.arcsec**2,
    )
    monkeypatch.setattr(val, "_telescope_area", lambda _ztrain: 1 * u.m**2)

    data = val.build_post_disperser_diffuse_background_data(
        train,
        wave_nm=wave_nm,
        qe_selector_name=None,
    )

    spectrum = data["channels"][0]["spectra"]["camera"]
    unblocked = data["channels"][0]["spectra_without_blocking"]["camera"]
    np.testing.assert_allclose(unblocked.value, [1.0, 1.0])
    np.testing.assert_allclose(spectrum.value, [0.25, 0.25])
    assert data["channels"][0]["total_rate_without_blocking_ph_s_pix"] > (
        data["channels"][0]["total_rate_ph_s_pix"]
    )


def test_detector_qe_accounting_table_reports_paths(monkeypatch):
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain([
        FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        ),
    ])
    train.image_planes = [None, None, FakeImagePlane(fits.Header())]
    monkeypatch.setattr(
        val, "_image_plane_pixel_area", lambda _ztrain, _id: 1 * u.arcsec**2,
    )
    monkeypatch.setattr(val, "_telescope_area", lambda _ztrain: 1 * u.m**2)

    transmission = val.build_transmission_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )
    emissivity = val.build_emissivity_sanity_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )
    post_diffuse = val.build_post_disperser_diffuse_background_data(
        train, wave_nm=wave_nm, qe_selector_name=None,
    )

    table = val.detector_qe_accounting_table(
        transmission, emissivity, post_diffuse,
    )

    assert list(table["path"]) == [
        "transmission_trace_mapped",
        "emissivity_sanity",
        "post_disperser_diffuse",
    ]
    assert list(table["qe_model"]) == ["FakeDetectorQE"] * 3


def test_auto_qe_selection_rejects_ambiguous_enabled_selectors():
    wave_nm = np.array([350.0, 360.0]) * u.nm
    train = FakeScienceTrain([
        FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        ),
        FakeNamedSelector(
            "tapered_detector_qe_selector",
            "aperture_id",
            {0: FakeTaperedQuantumEfficiency()},
        ),
    ])

    with np.testing.assert_raises_regex(ValueError, "qe_selector_name"):
        val.build_transmission_sanity_data(
            train, wave_nm=wave_nm, qe_selector_name=None,
        )


def test_slit_pair_status_table_marks_across_slit_source_outside():
    table = Table({
        "x": [0.0, 0.0],
        "y": [0.0, 1.0],
        "label": ["on", "off"],
    })
    table["x"].unit = u.arcsec
    table["y"].unit = u.arcsec

    status = val.slit_pair_status_table(
        table, slit_width=0.7 * u.arcsec, slit_length=10 * u.arcsec)

    assert list(status["label"]) == ["on", "off"]
    assert list(status["in_slit"]) == [True, False]


def test_slit_adc_psf_status_table_reports_context_without_simulating():
    class FakeSlitPsfTrain:
        cmds = {
            "!INST.vis_curr_slit": 0.7,
            "!INST.nir_curr_slit": 1.25,
            "!OBS.seeing": 0.6,
        }

        def __init__(self):
            self.optics_manager = type("Manager", (), {
                "all_effects": [
                    FakeNamedSelector(
                        "slitwheel_selector",
                        "aperture_id",
                        {
                            0: FakeSelectedEffect(
                                current_slit="!INST.vis_curr_slit",
                                filename_format="slits/slit_{:.2f}_as.dat",
                            ),
                            3: FakeSelectedEffect(
                                current_slit="!INST.nir_curr_slit",
                                filename_format="slits/slit_{:.2f}_as.dat",
                            ),
                        },
                    ),
                    FakeNamedSelector(
                        "channel_optics_selector",
                        "aperture_id",
                        {
                            0: FakeEffectWithMissingTable(),
                            1: FakeADCSurfaceList(),
                        },
                    ),
                    FakePSFEffect(),
                ],
            })()

    table = val.slit_adc_psf_status_table(FakeSlitPsfTrain())

    assert "slit" in set(table["category"])
    assert "adc optics" in set(table["category"])
    assert "psf" in set(table["category"])
    assert "atmospheric dispersion" in set(table["category"])
    assert any("current_slit=0.7" in value for value in table["setting"])
    assert any("ADC" in value for value in table["setting"])
    assert any("fwhm=0.6" in value for value in table["setting"])
    assert any("No active atmospheric-dispersion" in value for value in table["note"])


def _slit_adc_cmds():
    return {
        "!OBS.airmass": 1.3,
        "!OBS.seeing": 0.6,
        "!ATMO.temperature": 9.0,
        "!ATMO.pressure": 0.75,
        "!ATMO.humidity": 0.15,
        "!ATMO.x_co2": 450.0,
        "!INST.vis_curr_slit": 0.7,
        "!INST.nir_curr_slit": 0.7,
    }


def test_build_slit_adc_psf_scene_data_uses_physical_scene_images():
    source = Table({
        "x": [0.0, 0.0],
        "y": [-1.0, 1.0],
        "weight": [1.0, 0.5],
        "label": ["a", "b"],
    }, units=[u.arcsec, u.arcsec, None, None])

    data = val.build_slit_adc_psf_scene_data(
        _slit_adc_cmds(),
        {"along": source},
        slit_width=0.7 * u.arcsec,
        slit_length=4.0 * u.arcsec,
        wave_nm=np.linspace(400, 700, 5) * u.nm,
        grid_step=0.2 * u.arcsec,
        allow_diagnostic_psf=True,
    )

    assert list(data["variants"]) == ["ad_only", "adc_residual"]
    assert "along" in data["scenarios"]
    images = data["scenarios"]["along"]["images"]
    assert images["ad_only"].shape == images["adc_residual"].shape
    assert np.isfinite(images["ad_only"]).all()
    assert np.nanmax(images["ad_only"]) > 0
    assert any("before slit clipping" in note for note in data["notes"])


def test_build_slit_loss_data_returns_losses_between_zero_and_one():
    data = val.build_slit_loss_data(
        _slit_adc_cmds(),
        arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
        n_wave=5,
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
        allow_diagnostic_psf=True,
    )

    curves = data["arms"]["VIS"]["curves"]
    assert "no_ao_current_adc_residual" in curves
    assert "no_ao_zenith" in curves
    assert "no_ao_elevation_60_ad_only" in curves
    assert np.isclose(data["airmass"], 1.3)
    for curve in curves.values():
        assert np.all(curve["loss"] >= 0)
        assert np.all(curve["loss"] <= 1)


def test_build_slit_loss_data_requires_configured_psf_by_default():
    with np.testing.assert_raises_regex(ValueError, "No active ScopeSim"):
        val.build_slit_loss_data(
            _slit_adc_cmds(),
            arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
            n_wave=5,
            slit_length=4.0 * u.arcsec,
            grid_step=0.25 * u.arcsec,
        )


def test_build_slit_loss_data_includes_ao_mode_when_psf_supports_it():
    effect = AOEnhanceablePSF()
    data = val.build_slit_loss_data(
        FakeAOTrain(effect),
        arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
        n_wave=5,
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
    )

    curves = data["arms"]["VIS"]["curves"]
    assert "no_ao_zenith" in curves
    assert "ao_zenith" in curves
    assert curves["ao_zenith"]["linestyle"] == "--"
    assert np.all(curves["ao_zenith"]["loss"] < curves["no_ao_zenith"]["loss"])
    assert set(effect.seen_wave_units) == {u.um}
    assert set(effect.seen_fwhm_units) == {u.um}


def test_build_slit_loss_data_handles_relative_dimensionless_ao_scale():
    effect = AOEnhanceablePSF(scale=0.5, is_absolute=False)
    data = val.build_slit_loss_data(
        FakeAOTrain(effect),
        arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
        n_wave=5,
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
    )

    curves = data["arms"]["VIS"]["curves"]
    assert "ao_zenith" in curves
    assert np.isfinite(curves["ao_zenith"]["loss"]).all()
    assert np.all(curves["ao_zenith"]["loss"] < curves["no_ao_zenith"]["loss"])
    assert set(effect.seen_wave_units) == {u.um}
    assert set(effect.seen_fwhm_units) == {u.um}


def test_slit_loss_notebook_call_chain_handles_dimensionless_ao_tables():
    data = val.build_slit_loss_data(
        FakeAOTrain(),
        arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
        n_wave=5,
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
    )
    fig, axes = val.plot_slit_loss_by_arm(data)

    assert axes.shape == (1, 1)
    assert "no_ao_current_adc_residual" in data["arms"]["VIS"]["curves"]
    assert "ao_current_adc_residual" in data["arms"]["VIS"]["curves"]
    assert len(axes[0, 0].lines) == 8
    fig.clf()


def test_slit_adc_scene_uses_configured_scopesim_psf_fwhm():
    effect = AOEnhanceablePSF()
    source = Table({
        "x": [0.0],
        "y": [0.0],
        "weight": [1.0],
        "label": ["center"],
    }, units=[u.arcsec, u.arcsec, None, None])

    data = val.build_slit_adc_psf_scene_data(
        FakeAOTrain(effect),
        {"center": source},
        slit_width=0.7 * u.arcsec,
        slit_length=4.0 * u.arcsec,
        wave_nm=np.linspace(400, 700, 5) * u.nm,
        grid_step=0.25 * u.arcsec,
    )

    assert "center" in data["scenarios"]
    assert set(effect.seen_fwhm_units) == {u.um}
    assert any("active ScopeSim" in note for note in data["notes"])


def test_build_slit_width_loss_data_uses_configured_scopesim_psf_fwhm():
    effect = AOEnhanceablePSF()
    slit_widths = np.array([0.3, 0.7, 1.2]) * u.arcsec

    data = val.build_slit_width_loss_data(
        FakeAOTrain(effect),
        slit_widths=slit_widths,
        arms={"VIS": (np.array([500, 750]) * u.nm, "!INST.vis_curr_slit")},
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
    )

    arm = data["arms"]["VIS"]
    assert data["psf_modes"]["no_ao"]["label"] == '0.6" NS'
    np.testing.assert_allclose(
        arm["selector_slit_widths_arcsec"].to_value(u.arcsec),
        [0.33, 0.5, 0.7, 1.25],
    )
    assert set(arm["curves"]) == {
        "no_ao_500nm",
        "no_ao_750nm",
        "ao_500nm",
        "ao_750nm",
    }
    for curve in arm["curves"].values():
        assert curve["loss"].shape == slit_widths.shape
        assert np.all(curve["loss"] >= 0)
        assert np.all(curve["loss"] <= 1)
    assert np.all(
        arm["curves"]["no_ao_500nm"]["loss"][1:]
        <= arm["curves"]["no_ao_500nm"]["loss"][:-1]
    )
    assert set(effect.seen_fwhm_units) == {u.um}
    assert set(effect.seen_wave_units) == {u.um}


def test_image_plane_pixel_area_handles_detector_wcs_headers():
    area = val._image_plane_pixel_area(FakeTrainWithImagePlane(), 0)

    np.testing.assert_allclose(area.to_value(u.arcsec**2), 0.0225)


def test_post_disperser_diffuse_effect_consistency_table_compares_rates():
    helper_data = {
        "channels": {
            0: {
                "label": "B",
                "image_plane_id": 2,
                "pixel_area": 0.01 * u.arcsec**2,
                "total_rate_ph_s_pix": 1.01,
            },
        },
    }

    table = val.post_disperser_diffuse_effect_consistency_table(
        FakeTrainWithDiffuseEffect(), helper_data, match_effect_grid=False,
    )

    assert list(table["channel"]) == ["B"]
    np.testing.assert_allclose(table["effect_rate_ph_s_pix"], [1.0])
    np.testing.assert_allclose(table["helper_rel_delta"], [0.01])


def test_post_disperser_diffuse_effect_consistency_passes_component_metadata(monkeypatch):
    helper_data = {
        "channels": {
            0: {
                "label": "B",
                "image_plane_id": 2,
                "pixel_area": 0.01 * u.arcsec**2,
                "total_rate_ph_s_pix": 1.0,
            },
        },
    }
    component_metadata = {
        "ir_blocking_filter_selector": {"throughput_group": "ir_blocking_filter"},
    }
    train = FakeTrainWithDiffuseEffect()
    diffuse_effect = train.optics_manager.all_effects[0].wheel_effects[2]
    diffuse_effect._waveset = lambda: np.array([350.0, 360.0]) * u.nm
    seen_metadata = []

    def fake_build_post_diffuse(
        _ztrain,
        *,
        wave_nm,
        component_metadata,
        qe_selector_name,
        active_only,
    ):
        seen_metadata.append(component_metadata)
        np.testing.assert_allclose(wave_nm.to_value(u.nm), [350.0, 360.0])
        assert qe_selector_name == "detector_qe_selector"
        assert active_only is True
        return helper_data

    monkeypatch.setattr(
        val,
        "build_post_disperser_diffuse_background_data",
        fake_build_post_diffuse,
    )

    table = val.post_disperser_diffuse_effect_consistency_table(
        train,
        helper_data,
        component_metadata=component_metadata,
    )

    assert seen_metadata == [component_metadata]
    np.testing.assert_allclose(table["matched_rel_delta"], [0.0])


def test_validate_post_disperser_diffuse_effect_consistency_rejects_mismatch():
    table = Table({
        "matched_rel_delta": [0.0, 1e-3],
    })

    with np.testing.assert_raises_regex(ValueError, "mismatch"):
        val.validate_post_disperser_diffuse_effect_consistency(table, rtol=1e-6)


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


def test_trace_resolution_diagnostic_table_samples_trace_geometry(monkeypatch):
    header = fits.Header({
        "NAXIS": 2,
        "NAXIS1": 200,
        "NAXIS2": 50,
        "CTYPE1D": "LINEAR",
        "CTYPE2D": "LINEAR",
        "CUNIT1D": "mm",
        "CUNIT2D": "mm",
        "CRVAL1D": 0.0,
        "CRVAL2D": 0.0,
        "CRPIX1D": 1.0,
        "CRPIX2D": 1.0,
        "CDELT1D": 0.01,
        "CDELT2D": 0.01,
    })
    train = type("FakeTraceResolutionTrain", (), {})()
    train.cmds = {"!OBS.airmass": 1.0, "!OBS.seeing": 0.5}
    train.image_planes = [FakeImagePlane(header)]
    train.optics_manager = type("FakeOptics", (), {
        "all_effects": [
            named_effect(FakeGeometryTraceList(), "trace_list_analytical"),
        ],
    })()
    monkeypatch.setattr(
        val,
        "_image_plane_pixel_area",
        lambda _ztrain, _image_plane_id: 0.01 * u.arcsec**2,
    )
    monkeypatch.setattr(
        val,
        "_configured_psf_fwhm_func",
        lambda _ztrain, allow_diagnostic_fallback=False: (
            lambda wave, _zenith_angle, _seeing: np.full(wave.size, 0.5) * u.arcsec,
            None,
        ),
    )

    table = val.trace_resolution_diagnostic_table(train, samples_per_trace=3)

    assert len(table) == 3
    np.testing.assert_allclose(table["dispersion_nm_pix"], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(table["resolving_power_R"], [25.0, 26.25, 27.5])
    np.testing.assert_allclose(
        table["spectral_element_width_pix"],
        [4.0, 4.0, 4.0],
    )
    np.testing.assert_allclose(table["spatial_fwhm_pix"], [5.0, 5.0, 5.0])

    summary = val.trace_resolution_summary_table(table)
    assert list(summary["channel"]) == ["B"]
    np.testing.assert_allclose(summary["trace_R_median"], [26.25])
    np.testing.assert_allclose(summary["spectral_element_width_median_pix"], [4.0])
    np.testing.assert_allclose(
        summary["typical_spectral_element_width_median_pix"],
        [4.0],
    )
    np.testing.assert_allclose(summary["resel_footprint_min_pix"], [19.09090909])
    np.testing.assert_allclose(summary["resel_footprint_max_pix"], [21.0])

    display_table = val.trace_resolution_summary_display_table(summary)
    assert list(display_table.columns) == [
        "ch", "id", "orders", "wave nm", "disp nm/pix", "R k",
        "nyq R k", "pix/resel", "seeing",
    ]
    assert display_table.loc[0, "R k"] == "0.0"
    assert display_table.loc[0, "pix/resel"] == "19.1-21.0"


def test_trace_resolution_diagnostic_table_requires_full_slit_on_detector(
    monkeypatch,
):
    class EdgeTrace(FakeGeometryTrace):
        trace_id = "B_edge"
        wave_min = 0.0
        wave_max = 2.0
        table = Table({"s": [-1.0, 0.0, 1.0]})

        def xilam2x(self, xi, lam):
            return np.zeros(np.asarray(lam, dtype=float).shape)

        def xilam2y(self, xi, lam):
            return 0.2 + 0.01 * np.asarray(xi, dtype=float) - 0.1 * np.asarray(
                lam,
                dtype=float,
            )

    class EdgeTraceList:
        spectral_traces = {"B_edge": EdgeTrace()}

    header = fits.Header({
        "NAXIS": 2,
        "NAXIS1": 20,
        "NAXIS2": 50,
        "CTYPE1D": "LINEAR",
        "CTYPE2D": "LINEAR",
        "CUNIT1D": "mm",
        "CUNIT2D": "mm",
        "CRVAL1D": 0.0,
        "CRVAL2D": 0.0,
        "CRPIX1D": 1.0,
        "CRPIX2D": 1.0,
        "CDELT1D": 0.01,
        "CDELT2D": 0.01,
    })
    train = type("FakeTraceResolutionTrain", (), {})()
    train.cmds = {"!OBS.airmass": 1.0, "!OBS.seeing": 0.5}
    train.image_planes = [FakeImagePlane(header)]
    train.optics_manager = type("FakeOptics", (), {
        "all_effects": [
            named_effect(EdgeTraceList(), "trace_list_analytical"),
        ],
    })()
    monkeypatch.setattr(
        val,
        "_image_plane_pixel_area",
        lambda _ztrain, _image_plane_id: 0.01 * u.arcsec**2,
    )
    monkeypatch.setattr(
        val,
        "_configured_psf_fwhm_func",
        lambda _ztrain, allow_diagnostic_fallback=False: (
            lambda wave, _zenith_angle, _seeing: np.full(wave.size, 0.5) * u.arcsec,
            None,
        ),
    )

    table = val.trace_resolution_diagnostic_table(train, samples_per_trace=3)

    assert list(table["sample_index"]) == [0, 1]
    assert list(table["footprint_samples_on_detector"]) == [3, 3]


def test_detector_background_budget_table_combines_detector_terms():
    diffuse_data = {
        "channels": {
            0: {
                "image_plane_id": 0,
                "total_rate_ph_s_pix": 2.0,
            },
        },
    }

    ztrain = FakeBudgetTrain()
    ztrain.cmds = {"!DET.full_well": 100.0}

    table = val.detector_background_budget_table(ztrain, post_diffuse_data=diffuse_data)

    assert list(table["channel"]) == ["B"]
    np.testing.assert_allclose(table["exposure_time_s"], [30.0])
    np.testing.assert_allclose(table["post_diffuse_e_pix"], [60.0])
    np.testing.assert_allclose(table["dark_current_e_pix"], [3.0])
    np.testing.assert_allclose(table["additive_signal_e_pix"], [63.0])
    np.testing.assert_allclose(table["full_well_e"], [100.0])
    np.testing.assert_allclose(table["signal_fraction_of_full_well"], [0.63])
    assert list(table["saturation_status"]) == ["ok"]
    np.testing.assert_allclose(table["read_noise_e_rms"], [5.0 * np.sqrt(3)])
    expected_total_noise = np.sqrt(60.0 + 3.0 + 75.0)
    np.testing.assert_allclose(table["total_noise_e_rms"], [expected_total_noise])


def test_detector_background_budget_table_flags_saturation():
    diffuse_data = {
        "channels": {
            0: {
                "image_plane_id": 0,
                "total_rate_ph_s_pix": 2.0,
            },
        },
    }
    ztrain = FakeBudgetTrain()
    ztrain.cmds = {"!DET.full_well": 50.0}

    table = val.detector_background_budget_table(ztrain, post_diffuse_data=diffuse_data)

    np.testing.assert_allclose(table["signal_fraction_of_full_well"], [1.26])
    assert list(table["saturation_status"]) == ["saturated"]


def test_detector_background_budget_table_requires_full_well_command():
    ztrain = FakeBudgetTrain()

    with np.testing.assert_raises_regex(ValueError, "!DET.full_well"):
        val.detector_background_budget_table(
            ztrain,
            post_diffuse_data={
                "channels": {
                    0: {
                        "image_plane_id": 0,
                        "total_rate_ph_s_pix": 2.0,
                    },
                },
            },
        )


def test_science_truth_crosscheck_table_reports_physical_anchors():
    diffuse_data = {
        "channels": {
            0: {
                "image_plane_id": 0,
                "total_rate_ph_s_pix": 2.0,
                "pixel_area": 0.25 * u.arcsec**2,
                "trace_wave_min_nm": 300.0,
                "trace_wave_max_nm": 500.0,
            },
        },
    }
    ztrain = FakeBudgetTrain()
    ztrain.cmds = {
        "!DET.full_well": 100.0,
        "!SIM.spectral.spectral_resolution": 40000.0,
        "!INST.vis_curr_slit": 0.7,
    }
    budget = val.detector_background_budget_table(
        ztrain, post_diffuse_data=diffuse_data,
    )

    table = val.science_truth_crosscheck_table(
        ztrain,
        post_diffuse_data=diffuse_data,
        detector_budget=budget,
        extraction_pixels=4.0,
    )

    assert list(table["channel"]) == ["B"]
    np.testing.assert_allclose(table["current_slit_arcsec"], [0.7])
    np.testing.assert_allclose(table["wavelength_mid_nm"], [400.0])
    np.testing.assert_allclose(table["resolution_element_nm"], [0.01])
    np.testing.assert_allclose(table["post_diffuse_ph_s_arcsec2"], [8.0])
    np.testing.assert_allclose(table["post_diffuse_ph_s_extraction"], [8.0])
    np.testing.assert_allclose(table["post_diffuse_e_extraction"], [240.0])
    assert "integrated image-plane background" in table["note"][0]


def test_readout_delta_summary_table_reports_source_minus_reference():
    class FakeHDU:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)

    table = val.readout_delta_summary_table(
        [FakeHDU([[2.0, 3.0], [4.0, 5.0]])],
        [FakeHDU([[1.0, 1.0], [1.0, 1.0]])],
        titles=["B"],
    )

    assert list(table["channel"]) == ["B"]
    assert list(table["shape"]) == ["2x2"]
    np.testing.assert_allclose(table["sum_delta_e"], [10.0])
    np.testing.assert_allclose(table["max_abs_delta_e"], [4.0])
    assert list(table["nonzero_pixels"]) == [4]


def test_source_photon_crosscheck_table_uses_resolution_element_and_throughput():
    class FakeField:
        def __init__(self, spectrum):
            self.field = Table({
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "ref": [0, 0],
                "weight": [1.0, 2.0],
            })
            self.spectra = {0: spectrum}

    class FakeSource:
        def __init__(self, spectrum):
            self.fields = [FakeField(spectrum)]

    spectrum = SourceSpectrum(
        Empirical1D,
        points=[3900.0, 4100.0],
        lookup_table=[1.0, 1.0],
    )
    source = FakeSource(spectrum)
    ztrain = FakeBudgetTrain()
    ztrain.cmds = {
        "!TEL.area": "1 m2",
        "!SIM.spectral.spectral_resolution": 40000.0,
    }
    transmission = {
        "wave_nm": np.array([399.99, 400.0, 400.01]) * u.nm,
        "channels": {
            0: {
                "label": "B",
                "orders": {
                    "B_1": {
                        "total": np.array([0.5, 0.5, 0.5]),
                    },
                },
            },
        },
    }
    budget = Table({
        "channel": ["B"],
        "exposure_time_s": [10.0],
    })

    table = val.source_photon_crosscheck_table(
        source,
        ztrain,
        transmission,
        detector_budget=budget,
    )

    assert list(table["channel"]) == ["B"]
    np.testing.assert_allclose(table["resolution_element_nm"], [0.01])
    # 1 PHOTLAM over 0.1 A, 1 m2 telescope, summed weight 3, throughput 0.5.
    np.testing.assert_allclose(
        table["source_ph_s_resel_at_telescope"], [3000.0],
    )
    np.testing.assert_allclose(
        table["source_ph_s_resel_at_detector"], [1500.0],
    )
    np.testing.assert_allclose(table["source_e_resel"], [15000.0])


def test_resolution_element_footprint_table_derives_channel_scales(monkeypatch):
    train = FakeScienceTrain([
        FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE(), 1: FakeDetectorQE()},
        ),
    ])
    train.cmds.update({
        "!OBS.airmass": 1.3,
        "!OBS.seeing": 0.6,
        "!INST.vis_curr_slit": 0.7,
        "!INST.nir_curr_slit": 0.7,
        "!SIM.spectral.spectral_resolution": 10000.0,
    })
    train.image_planes.append(train.image_planes[-1])

    monkeypatch.setattr(
        val,
        "_trace_dispersion_nm_per_pixel",
        lambda _trace, _image_plane, _wave_mid_nm: 0.01,
    )
    monkeypatch.setattr(
        val,
        "_image_plane_pixel_area",
        lambda _ztrain, _image_plane_id: 0.04 * u.arcsec**2,
    )
    monkeypatch.setattr(
        val,
        "_configured_psf_fwhm_func",
        lambda _ztrain, allow_diagnostic_fallback=False: (
            lambda wave, _zenith_angle, _seeing: np.full(wave.size, 0.6) * u.arcsec,
            None,
        ),
    )

    table = val.resolution_element_footprint_table(train)

    by_channel = {str(row["channel"]): row for row in table}
    assert set(by_channel) == {"B", "R"}
    np.testing.assert_allclose(by_channel["B"]["wavelength_median_nm"], 350.0)
    np.testing.assert_allclose(by_channel["B"]["spectral_fwhm_pix"], 3.5)
    np.testing.assert_allclose(by_channel["B"]["spatial_fwhm_pix"], 3.0)
    np.testing.assert_allclose(by_channel["B"]["resel_pixels_fwhm"], 10.5)
    np.testing.assert_allclose(by_channel["B"]["snr_resel_scale"], np.sqrt(10.5))


def test_resolution_element_snr_summary_table_scales_positive_median():
    footprint = Table(rows=[{
        "channel": "B",
        "resel_pixels_fwhm": 9.0,
        "snr_resel_scale": 3.0,
        "spectral_fwhm_pix": 2.0,
        "spatial_fwhm_pix": 4.5,
    }])

    table = val.resolution_element_snr_summary_table(
        [np.array([[0.0, 1.0], [2.0, 4.0]])],
        footprint,
        titles=["B"],
    )

    np.testing.assert_allclose(table["median_positive_pixel_snr"], [2.0])
    np.testing.assert_allclose(table["median_resel_snr"], [6.0])
    assert list(table["positive_snr_pixels"]) == [3]


def test_source_photon_crosscheck_table_rejects_missing_transmission():
    class FakeSource:
        fields = []

    ztrain = FakeBudgetTrain()
    ztrain.cmds = {
        "!TEL.area": "1 m2",
        "!SIM.spectral.spectral_resolution": 40000.0,
    }
    transmission = {
        "wave_nm": np.array([400.0, 401.0]) * u.nm,
        "channels": {
            0: {
                "label": "B",
                "orders": {"B_1": {"total": np.array([np.nan, np.nan])}},
            },
        },
    }

    with np.testing.assert_raises_regex(ValueError, "no finite transmission"):
        val.source_photon_crosscheck_table(FakeSource(), ztrain, transmission)
