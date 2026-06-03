from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.io import fits
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
            "fwhm": 0.06,
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
        self.meta = {"name": "fake_ter", "action": "transmission", **meta}
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

    def __init__(self, scale=0.2, *, is_absolute=True):
        self.scale = scale
        self.seen_wave_units = []
        self.meta = {
            "name": self.display_name,
            "is_absolute": is_absolute,
        }

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
    cmds = {}

    def __init__(self, qe_selectors, optical_selectors=()):
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
        self.all_effects = [effect or AOEnhanceablePSF()]


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
    assert val.emission_phase_for_row(row, "camera") == "post_disperser"


def test_emission_phase_for_row_normalizes_common_aliases():
    row = Table({"name": ["Filter"], "emission_phase": ["postdisperser"]})[0]
    assert val.emission_phase_for_row(row) == "post_disperser"


def test_emission_phase_for_row_falls_back_to_camera_post_disperser():
    row = Table({"name": ["VIS_Camera_1"]})[0]
    assert val.emission_phase_for_row(row, "camera") == "post_disperser"


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
    np.testing.assert_allclose(channel["pre_disperser_total"], [0.16, 0.16])
    assert channel["order_detector_qe_methods"] == ["spectral throughput"]


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
            {0: FakeTEREffect(transmission=0.25)},
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
        {0: FakeTEREffect(transmission=0.25)},
    )
    extra_selector.meta["emission_phase"] = "postdisperser"
    train = FakeScienceTrain(
        [FakeNamedSelector(
            "detector_qe_selector",
            "aperture_id",
            {0: FakeDetectorQE()},
        )],
        optical_selectors=[extra_selector],
    )
    monkeypatch.setattr(val, "_telescope_area", lambda _ztrain: 1 * u.m**2)

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
            {0: FakeTEREffect(transmission=0.25)},
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
        "x": [0.0, 1.0],
        "y": [0.0, 0.0],
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
    )

    curves = data["arms"]["VIS"]["curves"]
    assert "no_ao_zenith" in curves
    assert "no_ao_elevation_60_ad_only" in curves
    for curve in curves.values():
        assert np.all(curve["loss"] >= 0)
        assert np.all(curve["loss"] <= 1)


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


def test_build_slit_loss_data_handles_relative_dimensionless_ao_scale():
    data = val.build_slit_loss_data(
        FakeAOTrain(AOEnhanceablePSF(scale=0.5, is_absolute=False)),
        arms={"VIS": (400 * u.nm, 700 * u.nm, "!INST.vis_curr_slit")},
        n_wave=5,
        slit_length=4.0 * u.arcsec,
        grid_step=0.25 * u.arcsec,
    )

    curves = data["arms"]["VIS"]["curves"]
    assert "ao_zenith" in curves
    assert np.isfinite(curves["ao_zenith"]["loss"]).all()
    assert np.all(curves["ao_zenith"]["loss"] < curves["no_ao_zenith"]["loss"])


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
    assert len(axes[0, 0].lines) == 6
    fig.clf()


def test_slit_loss_summary_table_computes_throughput():
    table = val.slit_loss_summary_table([
        {"scenario": "adc_on", "source": "source_0", "input_signal": 10, "output_signal": 7},
        {"scenario": "adc_on", "source": "source_1", "input_signal": 10, "output_signal": 0},
    ])

    np.testing.assert_allclose(table["throughput"], [0.7, 0.0])


def test_surface_list_emissivity_terms_splits_phases_and_applies_qe():
    wave = np.linspace(1, 2, 4) * u.um
    qe_values = np.full(wave.size, 0.5)

    terms, counts, details = val.surface_list_emissivity_terms(
        FakeSurfaceList(), wave, qe_values=qe_values)

    np.testing.assert_allclose(
        terms["pre_disperser"]["preoptics"],
        np.full(wave.size, 80.0),
    )
    np.testing.assert_allclose(
        terms["post_disperser"]["camera"],
        np.full(wave.size, 1.0),
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


def test_validate_post_disperser_diffuse_effect_consistency_rejects_mismatch():
    table = Table({
        "matched_rel_delta": [0.0, 1e-3],
    })

    with np.testing.assert_raises_regex(ValueError, "mismatch"):
        val.validate_post_disperser_diffuse_effect_consistency(table, rtol=1e-6)


def test_copy_image_plane_data_returns_detached_arrays():
    class DataPlane:
        def __init__(self):
            self.hdu = fits.ImageHDU(data=np.ones((2, 2)))

    ztrain = type("Train", (), {"image_planes": [DataPlane()]})()

    copied = val.copy_image_plane_data(ztrain)
    ztrain.image_planes[0].hdu.data[0, 0] = 9

    assert list(copied) == [0]
    np.testing.assert_allclose(copied[0], np.ones((2, 2)))


def test_image_plane_delta_summary_compares_uniform_expected_rates():
    with_effect = {
        2: np.full((2, 3), 4.25),
        3: np.full((2, 2), 8.5),
    }
    without_effect = {
        2: np.full((2, 3), 1.25),
        3: np.full((2, 2), 4.5),
    }
    expected = Table({
        "image_plane_id": [2, 3],
        "effect_rate_ph_s_pix": [3.0, 4.0],
    })

    table = val.image_plane_delta_summary(
        with_effect, without_effect, expected_rates=expected,
    )

    assert list(table["image_plane_id"]) == [2, 3]
    assert list(table["shape"]) == ["2x3", "2x2"]
    np.testing.assert_allclose(table["mean_delta_ph_s_pix"], [3.0, 4.0])
    np.testing.assert_allclose(table["std_delta_ph_s_pix"], [0.0, 0.0])
    np.testing.assert_allclose(table["expected_rel_delta"], [0.0, 0.0])
    val.validate_image_plane_delta_summary(table)


def test_validate_image_plane_delta_summary_rejects_nonuniform_delta():
    table = Table({
        "expected_rel_delta": [0.0],
        "std_delta_ph_s_pix": [0.2],
        "mean_delta_ph_s_pix": [1.0],
    })

    with np.testing.assert_raises_regex(ValueError, "not spatially uniform"):
        val.validate_image_plane_delta_summary(table)


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
