"""Validation helpers for ZShooter ScopeSim notebooks."""

from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM

from .plots import (
    plot_detector_background_budget,
    plot_emissivity_sanity,
    plot_post_disperser_diffuse_background,
    plot_readout_cross_dispersion_cut,
    plot_readout_overview,
    plot_slit_adc_psf_scenes,
    plot_slit_loss_by_arm,
    plot_slit_pair_geometry,
    plot_source,
    plot_transmission_sanity,
)


def effect_name(effect: Any) -> str:
    """Return the notebook-facing name of a ScopeSim effect."""
    display_name = getattr(effect, "display_name", None)
    if display_name is not None:
        return display_name
    meta = getattr(effect, "meta", {})
    return meta.get("name", repr(effect))


def active_effects(ztrain: Any) -> list[Any]:
    """Return included effects from an optical train."""
    return [
        eff for eff in ztrain.optics_manager.all_effects
        if getattr(eff, "include", True)
    ]


def get_effect(ztrain: Any, display_name: str, *, active_only: bool = True) -> Any:
    """Fetch one active optical-train effect by display name."""
    effects = active_effects(ztrain) if active_only else ztrain.optics_manager.all_effects
    matches = [eff for eff in effects if effect_name(eff) == display_name]
    if len(matches) != 1:
        available = sorted(effect_name(eff) for eff in effects)
        raise ValueError(
            f"Expected one active effect named {display_name!r}; found "
            f"{len(matches)}. Available: {available}"
        )
    return matches[0]


def resolve_effect(effect: Any, selector_value: Any = None) -> Any:
    """Return a plain effect, or one selected entry from a SelectorWheel."""
    if not hasattr(effect, "wheel_effects"):
        return effect
    if selector_value is None:
        keys = sorted(effect.wheel_effects)
        raise ValueError(
            f"{effect_name(effect)!r} is selector-based; choose one of {keys}"
        )
    try:
        return effect.wheel_effects[selector_value]
    except KeyError as exc:
        keys = sorted(effect.wheel_effects)
        raise KeyError(
            f"{effect_name(effect)!r} has no selector value "
            f"{selector_value!r}; available {keys}"
        ) from exc


def get_arm_effect(ztrain: Any, display_name: str, aperture_id: int) -> Any:
    """Fetch one selected arm/channel effect from an optical train."""
    return resolve_effect(get_effect(ztrain, display_name), aperture_id)


def _as_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def evaluate_curve(curve: Any, wave: u.Quantity) -> np.ndarray:
    """Evaluate a synphot/ScopeSim curve on a wavelength quantity."""
    with u.set_enabled_equivalencies(u.spectral()):
        return _as_float_array(curve(wave))


def evaluate_throughput(effect: Any, wave: u.Quantity) -> np.ndarray:
    """Evaluate an effect or SpectralSurface-like object as throughput."""
    if hasattr(effect, "throughput"):
        return evaluate_curve(effect.throughput, wave)
    if hasattr(effect, "surface") and hasattr(effect.surface, "throughput"):
        return evaluate_curve(effect.surface.throughput, wave)
    raise TypeError(f"Cannot evaluate throughput for {effect!r}")


def _evaluate_diffuse_throughput(
    effect: Any,
    wave: u.Quantity,
    *,
    footprint: Any = None,
) -> np.ndarray:
    if hasattr(effect, "effective_diffuse_throughput"):
        with u.set_enabled_equivalencies(u.spectral()):
            values = effect.effective_diffuse_throughput(
                wave, footprint=footprint,
            )
        return _as_float_array(values)
    return evaluate_throughput(effect, wave)


def _candidate_qe_effects(effect: Any) -> list[Any]:
    if hasattr(effect, "wheel_effects"):
        return list(effect.wheel_effects.values())
    return [effect]


def _looks_like_qe_effect(effect: Any) -> bool:
    name = effect_name(effect).lower()
    class_name = effect.__class__.__name__.lower()
    if "qe" in name or "quantumefficiency" in class_name:
        return True
    return any(
        "quantumefficiency" in candidate.__class__.__name__.lower()
        for candidate in _candidate_qe_effects(effect)
    )


def _get_qe_selector(
    ztrain: Any,
    qe_selector_name: str | None,
    *,
    active_only: bool = True,
) -> Any:
    if qe_selector_name is not None:
        return get_effect(
            ztrain, qe_selector_name, active_only=active_only,
        )

    effects = active_effects(ztrain) if active_only else ztrain.optics_manager.all_effects
    candidates = [effect for effect in effects if _looks_like_qe_effect(effect)]
    if len(candidates) != 1:
        available = [
            effect_name(effect) for effect in candidates
        ] or sorted(effect_name(effect) for effect in effects)
        raise ValueError(
            "Expected exactly one enabled QE selector when "
            "qe_selector_name=None; found "
            f"{len(candidates)}. Pass qe_selector_name explicitly. "
            f"Candidates/available effects: {available}"
        )
    return candidates[0]


def _qe_detail_summary(effect: Any) -> dict[str, Any]:
    meta = getattr(effect, "meta", {}) or {}
    surface = getattr(effect, "surface", None)
    summary = {
        "effect_class": effect.__class__.__name__,
        "qe_model": effect.__class__.__name__,
        "qe_source": meta.get("filename", "configured"),
        "action": meta.get("action", "throughput"),
        "temperature": meta.get("temperature"),
        "transmission_source": "configured",
    }
    if surface is not None:
        surface_meta = getattr(surface, "meta", {}) or {}
        summary.update({
            "qe_source": meta.get("filename", surface_meta.get("filename", "surface")),
            "action": surface_meta.get("action", meta.get("action", "transmission")),
            "temperature": surface_meta.get("temperature", meta.get("temperature")),
            "transmission_source": ter_property_sources(surface)["transmission"],
        })
    return summary


def _is_surface_list_effect(effect: Any) -> bool:
    table = getattr(effect, "table", None)
    return hasattr(effect, "surfaces") and hasattr(table, "colnames")


def _is_surface_effect(effect: Any) -> bool:
    surface = getattr(effect, "surface", None)
    return surface is not None and any(
        hasattr(surface, attr)
        for attr in ("throughput", "transmission", "reflection", "emissivity")
    )


def _selected_aperture_effect(effect: Any, aperture_id: int) -> Any | None:
    if not hasattr(effect, "wheel_effects"):
        return None
    if getattr(effect, "meta", {}).get("selector_key") != "aperture_id":
        return None
    try:
        return resolve_effect(effect, aperture_id)
    except KeyError:
        return None


def _surface_effect_group_name(parent: Any, effect: Any) -> str:
    for meta in (getattr(effect, "meta", {}), getattr(parent, "meta", {})):
        group_name = _clean_metadata_value(meta.get("throughput_group"))
        if group_name is not None:
            return group_name

    name = effect_name(parent)
    return name[:-9] if name.endswith("_selector") else name


def _surface_effect_phase_name(parent: Any, effect: Any) -> str:
    for meta in (getattr(effect, "meta", {}), getattr(parent, "meta", {})):
        phase_name = _clean_emission_phase(meta.get("emission_phase"))
        if phase_name is not None:
            return phase_name
    return "none"


def _surface_effect_action_name(effect: Any) -> str:
    surface = getattr(effect, "surface", None)
    for meta in (
        getattr(surface, "meta", {}) if surface is not None else {},
        getattr(effect, "meta", {}),
    ):
        action_name = _clean_metadata_value(meta.get("action"))
        if action_name is not None:
            return action_name
    return "transmission"


def channel_optical_components(
    ztrain: Any,
    aperture_id: int,
    *,
    qe_selector: Any | None = None,
) -> list[dict[str, Any]]:
    """Return active per-aperture optical components before trace mapping."""
    excluded_ids = {id(qe_selector)} if qe_selector is not None else set()
    components: list[dict[str, Any]] = []
    for parent in active_effects(ztrain):
        if id(parent) in excluded_ids or _looks_like_qe_effect(parent):
            continue
        effect = _selected_aperture_effect(parent, aperture_id)
        if effect is None or _looks_like_qe_effect(effect):
            continue
        if not (_is_surface_list_effect(effect) or _is_surface_effect(effect)):
            continue
        components.append({
            "selector": parent,
            "selector_name": effect_name(parent),
            "effect": effect,
        })
    return components


def optical_surface_rows(
    components: list[dict[str, Any]],
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten discovered optical components into ordered surface rows."""
    rows: list[dict[str, Any]] = []
    for component in components:
        parent = component["selector"]
        effect = component["effect"]
        selector_name = component["selector_name"]
        if _is_surface_list_effect(effect):
            name_col = _real_colname("name", effect.table.colnames)
            action_col = _real_colname("action", effect.table.colnames)
            if name_col is None or action_col is None:
                raise ValueError(
                    "SurfaceList table must contain name and action columns."
                )
            for row in effect.table:
                surface_name = str(_row_scalar(row, name_col))
                action_name = str(_row_scalar(row, action_col))
                group_name = surface_group_for_row(row, groups)
                phase_name = emission_phase_for_row(row, group_name)
                surface = effect.surfaces[surface_name]
                rows.append({
                    "component": selector_name,
                    "surface_name": surface_name,
                    "group": group_name,
                    "emission_phase": phase_name,
                    "action": action_name,
                    "surface": surface,
                    "action_values": evaluate_curve(
                        getattr(surface, action_name), wave,
                    ),
                    "emissivity": evaluate_ter_property(
                        surface, "emissivity", wave,
                    ),
                    "emission_values": evaluate_emission_density(surface, wave),
                    "temperature": surface.meta.get("temperature"),
                    "ter_sources": ter_property_sources(surface),
                })
            continue

        surface = effect.surface
        action_name = _surface_effect_action_name(effect)
        group_name = _surface_effect_group_name(parent, effect)
        phase_name = _surface_effect_phase_name(parent, effect)
        rows.append({
            "component": selector_name,
            "surface_name": selector_name,
            "group": group_name,
            "emission_phase": phase_name,
            "action": action_name,
            "surface": surface,
            "action_values": evaluate_curve(getattr(surface, action_name), wave),
            "emissivity": evaluate_ter_property(surface, "emissivity", wave),
            "emission_values": evaluate_emission_density(surface, wave),
            "temperature": surface.meta.get("temperature"),
            "ter_sources": ter_property_sources(surface),
        })
    return rows


def fetch_effect_spectrum_or_transmission(
    ztrain: Any,
    display_name: str,
    attribute: str | None = "throughput",
    wave: u.Quantity | None = None,
    aperture_id: int | None = None,
) -> tuple[u.Quantity, Any]:
    """Fetch a curve-like attribute from an active effect.

    ``attribute`` may be dotted, e.g. ``"line_TER.emission"``.
    """
    effect = get_effect(ztrain, display_name)
    target = resolve_effect(effect, aperture_id) if aperture_id is not None else effect
    if attribute is not None:
        for attr in attribute.split("."):
            target = getattr(target, attr)

    if wave is not None:
        return wave.to(u.um), evaluate_curve(target, wave)
    if hasattr(target, "_get_arrays"):
        wave_out, values = target._get_arrays(wavelengths=None)
        return wave_out.to(u.um), values
    if hasattr(target, "waveset"):
        wave_out = target.waveset.to(u.um)
        return wave_out, evaluate_curve(target, wave_out)
    raise TypeError(f"Cannot fetch spectrum/transmission from {target!r}")


def default_surface_groups() -> OrderedDict[str, tuple[str, ...]]:
    """Fallback group rules for older optics lists without group metadata."""
    return OrderedDict([
        ("preoptics", ("Window", "ADC", "Derotator", "PreOpt", "Fold")),
        ("collimator", ("Col", "Mangin")),
        ("camera", ("Camera",)),
    ])


def _real_colname(name: str, colnames: list[str] | tuple[str, ...]) -> str | None:
    name_lower = name.lower()
    for colname in colnames:
        if colname.lower() == name_lower:
            return colname
    return None


def _row_scalar(row: Any, name: str) -> Any:
    value = row[name]
    return value.item() if hasattr(value, "item") else value


def _clean_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "None", "nan"}:
        return None
    return text


def _clean_emission_phase(value: Any) -> str | None:
    phase_name = _clean_metadata_value(value)
    if phase_name is None:
        return None
    key = phase_name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "predisperser": "pre_disperser",
        "pre_disperser": "pre_disperser",
        "postdisperser": "post_disperser",
        "post_disperser": "post_disperser",
        "none": "none",
    }
    return aliases.get(key, phase_name)


def surface_group_for_name(
    surface_name: str,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    """Return fallback throughput group from a surface name."""
    groups = groups or default_surface_groups()
    return next(
        (name for name, patterns in groups.items()
         if any(pattern in surface_name for pattern in patterns)),
        "other",
    )


def surface_group_for_row(
    row: Any,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    """Return explicit row throughput group, falling back to name patterns."""
    group_col = _real_colname("throughput_group", row.colnames)
    if group_col is not None:
        group_name = _clean_metadata_value(_row_scalar(row, group_col))
        if group_name is not None:
            return group_name

    name_col = _real_colname("name", row.colnames)
    surface_name = str(_row_scalar(row, name_col))
    return surface_group_for_name(surface_name, groups)


def emission_phase_for_row(row: Any, group_name: str | None = None) -> str:
    """Return explicit row emission phase, falling back conservatively."""
    phase_col = _real_colname("emission_phase", row.colnames)
    if phase_col is not None:
        phase_name = _clean_emission_phase(_row_scalar(row, phase_col))
        if phase_name is not None:
            return phase_name

    if group_name == "camera":
        return "post_disperser"
    return "pre_disperser"


def ter_property_sources(surface: Any) -> dict[str, str]:
    """Report whether TER properties are explicit or ScopeSim-inferred."""
    colnames = set(getattr(surface.table, "colnames", []))
    explicit = {
        name: name in surface.meta or name in colnames
        for name in ("transmission", "reflection", "emissivity")
    }
    sources = {}
    for name, is_explicit in explicit.items():
        if is_explicit:
            sources[name] = "explicit"
        else:
            other_names = [
                other for other, exists in explicit.items()
                if other != name and exists
            ]
            if other_names:
                sources[name] = "inferred from " + "+".join(other_names)
            else:
                sources[name] = "missing"
    return sources


def evaluate_ter_property(
    effect_or_surface: Any,
    property_name: str,
    wave: u.Quantity,
) -> np.ndarray:
    """Evaluate one TER property from a TERCurve or SpectralSurface."""
    surface = getattr(effect_or_surface, "surface", effect_or_surface)
    curve = getattr(surface, property_name)
    if curve is None:
        return np.full(wave.size, np.nan)
    return evaluate_curve(curve, wave)


def surface_list_group_throughputs(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, int]]:
    """Return grouped throughputs for one SurfaceList."""
    name_col = _real_colname("name", surface_list.table.colnames)
    action_col = _real_colname("action", surface_list.table.colnames)

    grouped: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: dict[str, int] = {}

    for row in surface_list.table:
        surface_name = str(_row_scalar(row, name_col))
        action_name = str(_row_scalar(row, action_col))
        group_name = surface_group_for_row(row, groups)
        surface = surface_list.surfaces[surface_name]

        if group_name not in grouped:
            grouped[group_name] = np.ones(wave.size, dtype=float)
            counts[group_name] = 0
        grouped[group_name] *= evaluate_curve(getattr(surface, action_name), wave)
        counts[group_name] += 1

    return grouped, counts


def optical_surface_group_throughputs(
    rows: list[dict[str, Any]],
    wave: u.Quantity,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, int]]:
    """Return grouped throughputs from flattened optical surface rows."""
    grouped: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: dict[str, int] = {}
    for row in rows:
        group_name = row["group"]
        if group_name not in grouped:
            grouped[group_name] = np.ones(wave.size, dtype=float)
            counts[group_name] = 0
        grouped[group_name] *= row["action_values"]
        counts[group_name] += 1
    return grouped, counts


def surface_list_emissivity_terms(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    qe_values: np.ndarray | None = None,
) -> tuple[dict[str, OrderedDict[str, np.ndarray]], dict[str, int], list[dict[str, Any]]]:
    """Return grouped thermal-emission terms split by optical phase.

    The per-surface base contribution follows ScopeSim's thermal-emission
    path: a surface emission-density term is weighted by the downstream action
    throughputs that ScopeSim applies after that surface's own emission is
    added.

    For ``post_disperser`` terms, ``after_qe`` also applies the detector QE
    because this diffuse light is injected at the image plane instead of being
    trace-mapped as a spectral source.
    """
    rows = optical_surface_rows(
        [{
            "selector": surface_list,
            "selector_name": effect_name(surface_list),
            "effect": surface_list,
        }],
        wave,
        groups=groups,
    )
    return optical_surface_emissivity_terms(rows, wave, qe_values=qe_values)


def _emission_density_plot_values(values: u.Quantity | None) -> np.ndarray | None:
    """Return thermal emission density in PHOTLAM-equivalent plot units."""
    if values is None:
        return None
    try:
        return values.to_value(PHOTLAM)
    except Exception:
        return _as_float_array(values)


def optical_surface_emissivity_terms(
    rows: list[dict[str, Any]],
    wave: u.Quantity,
    qe_values: np.ndarray | None = None,
) -> tuple[dict[str, OrderedDict[str, np.ndarray]], dict[str, int], list[dict[str, Any]]]:
    """Return grouped thermal-emission terms from flattened optical rows."""
    downstream = [np.ones(wave.size, dtype=float) for _ in range(len(rows) + 1)]
    for idx in range(len(rows) - 1, -1, -1):
        downstream[idx] = downstream[idx + 1] * rows[idx]["action_values"]

    phase_terms: dict[str, OrderedDict[str, np.ndarray]] = {
        "pre_disperser": OrderedDict(),
        "post_disperser": OrderedDict(),
    }
    counts: dict[str, int] = defaultdict(int)
    details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if row["emission_phase"] == "none":
            details.append({
                "component": row["component"],
                "surface": row["surface_name"],
                "group": row["group"],
                "emission_phase": "none",
                "action": row["action"],
                "temperature": row["temperature"],
                "emissivity_source": row["ter_sources"]["emissivity"],
                "transmission_source": row["ter_sources"]["transmission"],
                "reflection_source": row["ter_sources"]["reflection"],
                "qe_applied": False,
                "peak_output_emissivity": np.nan,
                "peak_after_qe": np.nan,
            })
            continue

        emission_values = _emission_density_plot_values(row["emission_values"])
        if emission_values is None:
            contribution = np.zeros(wave.size, dtype=float)
        else:
            contribution = emission_values * downstream[idx + 1]
            contribution = np.nan_to_num(
                contribution, nan=0.0, posinf=0.0, neginf=0.0,
            )
        after_qe = contribution
        qe_applied = False
        if row["emission_phase"] == "post_disperser" and qe_values is not None:
            after_qe = contribution * qe_values
            qe_applied = True

        phase_name = row["emission_phase"]
        if phase_name not in phase_terms:
            raise ValueError(
                f"Unknown emission_phase {phase_name!r} for "
                f"{row['surface_name']!r}; expected one of "
                f"{sorted(phase_terms)} or 'none'."
            )
        group_name = row["group"]
        if group_name not in phase_terms[phase_name]:
            phase_terms[phase_name][group_name] = np.zeros(wave.size, dtype=float)
        phase_terms[phase_name][group_name] += after_qe
        counts[f"{phase_name}:{group_name}"] += 1

        details.append({
            "component": row["component"],
            "surface": row["surface_name"],
            "group": group_name,
            "emission_phase": phase_name,
            "action": row["action"],
            "temperature": row["temperature"],
            "emissivity_source": row["ter_sources"]["emissivity"],
            "transmission_source": row["ter_sources"]["transmission"],
            "reflection_source": row["ter_sources"]["reflection"],
            "qe_applied": qe_applied,
            "peak_output_emissivity": (
                float(np.nanmax(contribution)) if contribution.size else np.nan
            ),
            "peak_output_thermal_emission": (
                float(np.nanmax(contribution)) if contribution.size else np.nan
            ),
            "peak_after_qe": float(np.nanmax(after_qe)) if after_qe.size else np.nan,
        })

    return phase_terms, dict(counts), details


def representative_positional_qe(positional_qe: Any, wave: u.Quantity) -> float | np.ndarray:
    """Return an average positional QE factor for diffuse, non-trace light."""
    if positional_qe is None:
        return 1.0
    values = positional_qe(wave) if callable(positional_qe) else positional_qe
    values = _as_float_array(values)
    if values.shape == wave.shape:
        return values
    return float(np.nanmean(values))


def effective_diffuse_qe(
    detector_qe: Any,
    wave: u.Quantity,
    positional_qe: Callable[[u.Quantity], Any] | np.ndarray | None = None,
    footprint: Any = None,
) -> np.ndarray:
    """Return detector QE for diffuse image-plane backgrounds.

    For ordinary detectors this is the spectral QE. For future tapered coatings,
    pass a positional QE map or callable; this function applies its average
    positional response instead of skipping QE for non-dispersed light.
    """
    spectral_qe = _evaluate_diffuse_throughput(
        detector_qe, wave, footprint=footprint,
    )
    return spectral_qe * representative_positional_qe(positional_qe, wave)


def detector_qe_accounting_summary(detector_qe: Any) -> dict[str, Any]:
    """Return notebook-facing detector QE accounting metadata."""
    qe_detail = _qe_detail_summary(detector_qe)
    is_position_aware = hasattr(detector_qe, "throughput_at")
    uses_footprint = bool(getattr(detector_qe, "uses_detector_footprint", False))
    meta = getattr(detector_qe, "meta", {}) or {}
    return {
        **qe_detail,
        "trace_mapped_qe": (
            "per-pixel detector coordinates from trace mapping"
            if is_position_aware
            else "wavelength-only spectral throughput"
        ),
        "transmission_plot_qe": (
            "footprint/detector-axis average shown for display; order totals "
            "use slit-center trace detector coordinates"
            if uses_footprint
            else "same wavelength-only spectral throughput used for all orders"
        ),
        "diffuse_qe": (
            "mean over uniformly sampled detector-axis positions"
            if uses_footprint
            else "wavelength-only spectral throughput"
        ),
        "diffuse_position_samples": meta.get("diffuse_position_samples"),
        "position_axis": meta.get("axis"),
        "position_min": meta.get("position_min"),
        "position_max": meta.get("position_max"),
    }


def _trace_center_detector_positions(
    trace: Any,
    wave: u.Quantity,
    image_plane: Any | None = None,
) -> dict[str, np.ndarray] | None:
    """Return slit-center detector coordinates for a trace and wave grid."""
    if not all(hasattr(trace, attr) for attr in ("xilam2x", "xilam2y")):
        return None

    wave_um = wave.to_value(u.um)
    xi = np.zeros(wave_um.shape, dtype=float)
    x_mm = np.asarray(trace.xilam2x(xi, wave_um), dtype=float)
    y_mm = np.asarray(trace.xilam2y(xi, wave_um), dtype=float)
    coords = {
        "detector_x_mm": x_mm,
        "detector_y_mm": y_mm,
    }

    if image_plane is None:
        return coords

    try:
        from astropy.wcs import WCS

        x_pix, y_pix = WCS(image_plane.header, key="D").all_world2pix(
            x_mm, y_mm, 0,
        )
        coords["detector_x"] = np.asarray(x_pix, dtype=float)
        coords["detector_y"] = np.asarray(y_pix, dtype=float)
    except Exception:
        pass

    return coords


def evaluate_trace_detector_qe(
    detector_qe: Any,
    trace: Any,
    wave: u.Quantity,
    image_plane: Any | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate detector QE for a trace sanity curve."""
    if not hasattr(detector_qe, "throughput_at"):
        return evaluate_throughput(detector_qe, wave), "spectral throughput"

    coords = _trace_center_detector_positions(trace, wave, image_plane)
    if coords is None:
        return (
            evaluate_throughput(detector_qe, wave),
            "representative detector midpoint",
        )

    with u.set_enabled_equivalencies(u.spectral()):
        values = detector_qe.throughput_at(
            wave,
            detector_x=coords.get("detector_x"),
            detector_y=coords.get("detector_y"),
            detector_x_mm=coords["detector_x_mm"],
            detector_y_mm=coords["detector_y_mm"],
            trace=trace,
        )
    method = (
        "slit-center trace detector pixels"
        if "detector_y" in coords or "detector_x" in coords
        else "representative detector midpoint; detector-pixel WCS unavailable"
    )
    return _as_float_array(values), method


def traces_by_aperture(trace_list: Any) -> dict[int, list[Any]]:
    """Group spectral traces by aperture id."""
    grouped: dict[int, list[Any]] = defaultdict(list)
    for trace_id, trace in trace_list.spectral_traces.items():
        grouped[int(trace.meta["aperture_id"])].append(trace)
    return {
        key: sorted(value, key=lambda trace: trace.trace_id)
        for key, value in grouped.items()
    }


def channel_label(aperture_id: int, traces: list[Any]) -> str:
    """Return display label for one aperture/channel."""
    if traces:
        prefix = traces[0].trace_id.partition("_")[0]
        return prefix.upper()
    return f"aperture {aperture_id}"


def trace_catalog_table(trace_list: Any) -> Table:
    """Return the loaded trace catalog from an in-memory SpectralTraceList."""
    rows = []
    for trace_id, trace in trace_list.spectral_traces.items():
        rows.append({
            "trace_id": trace.trace_id,
            "aperture_id": int(trace.meta["aperture_id"]),
            "image_plane_id": int(trace.meta["image_plane_id"]),
            "extension_id": int(trace.meta.get("extension_id", -1)),
            "wave_min_um": float(trace.wave_min),
            "wave_max_um": float(trace.wave_max),
        })
    return Table(rows=sorted(rows, key=lambda row: row["trace_id"]))


def fov_image_plane_counts(ztrain: Any) -> Table:
    """Return image-plane counts for the current in-memory FOV manager."""
    image_plane_ids = [
        int(fov.meta["image_plane_id"])
        for fov in ztrain.fov_manager.fovs
    ]
    unique_ids, counts = np.unique(image_plane_ids, return_counts=True)
    return Table({
        "image_plane_id": unique_ids.astype(int),
        "n_fovs": counts.astype(int),
    })


def build_emissivity_sanity_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
    blocking_component_names: tuple[str, ...] = ("ir_blocking_filter_selector",),
    diffuse_extraction_pixels: float = 1.0,
) -> dict[str, Any]:
    """Build split pre/post-disperser emissivity sanity-check data."""
    from scopesim.effects.illumination import integrate_spectral_background

    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)
    positional_qe_by_aperture = positional_qe_by_aperture or {}
    blocked_components = set(blocking_component_names)
    telescope_area = _telescope_area(ztrain)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    qe_selector = _get_qe_selector(
        ztrain, qe_selector_name, active_only=active_only,
    )
    trace_list = get_effect(ztrain, "trace_list_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    details: list[dict[str, Any]] = []
    for aperture_id in aperture_ids:
        detector_qe = resolve_effect(qe_selector, aperture_id)
        traces = traces_for_aperture.get(aperture_id, [])
        image_plane_id = (
            int(traces[0].meta["image_plane_id"]) if traces else aperture_id
        )
        try:
            image_pixel_area = _image_plane_pixel_area(ztrain, image_plane_id)
        except (AttributeError, IndexError, KeyError):
            image_pixel_area = 1.0 * u.arcsec**2
        qe_values = effective_diffuse_qe(
            detector_qe,
            wave,
            positional_qe=positional_qe_by_aperture.get(aperture_id),
        )

        components = channel_optical_components(
            ztrain, aperture_id, qe_selector=qe_selector,
        )
        surface_rows = optical_surface_rows(components, wave, groups=groups)
        phase_terms, counts, surface_details = optical_surface_emissivity_terms(
            surface_rows, wave, qe_values=qe_values,
        )
        unblocked_rows = _rows_excluding_components(
            surface_rows, blocked_components,
        )
        unblocked_phase_terms, _unblocked_counts, _unblocked_details = (
            optical_surface_emissivity_terms(
                unblocked_rows, wave, qe_values=qe_values,
            )
        )
        pre_total = _sum_terms(phase_terms["pre_disperser"], wave.size)
        post_total = _sum_terms(phase_terms["post_disperser"], wave.size)
        post_unblocked_total = _sum_terms(
            unblocked_phase_terms["post_disperser"], wave.size,
        )
        post_blocked_delta = post_unblocked_total - post_total
        diffuse_rate_per_pix = integrate_spectral_background(
            post_total * PHOTLAM,
            wave,
            telescope_area=telescope_area,
            image_pixel_area=image_pixel_area,
        )
        diffuse_unblocked_rate_per_pix = integrate_spectral_background(
            post_unblocked_total * PHOTLAM,
            wave,
            telescope_area=telescope_area,
            image_pixel_area=image_pixel_area,
        )
        diffuse_extract_equiv = np.full(
            wave.size, diffuse_rate_per_pix * diffuse_extraction_pixels,
            dtype=float,
        )
        diffuse_unblocked_extract_equiv = np.full(
            wave.size,
            diffuse_unblocked_rate_per_pix * diffuse_extraction_pixels,
            dtype=float,
        )

        label = channel_label(aperture_id, traces)
        for row in surface_details:
            row["aperture_id"] = aperture_id
            row["channel"] = label
            details.append(row)

        qe_detail = _qe_detail_summary(detector_qe)
        details.append({
            "aperture_id": aperture_id,
            "channel": label,
            "surface": effect_name(detector_qe),
            "group": "detector_qe",
            "emission_phase": "throughput_only",
            "effect_class": qe_detail["effect_class"],
            "qe_model": qe_detail["qe_model"],
            "qe_source": qe_detail["qe_source"],
            "action": qe_detail["action"],
            "temperature": qe_detail["temperature"],
            "emissivity_source": "not used",
            "transmission_source": qe_detail["transmission_source"],
            "reflection_source": "not used",
            "qe_applied": True,
            "peak_output_emissivity": np.nan,
            "peak_after_qe": float(np.nanmax(qe_values)),
        })

        channels[aperture_id] = {
            "label": label,
            "pre_disperser_terms": phase_terms["pre_disperser"],
            "post_disperser_terms": phase_terms["post_disperser"],
            "post_disperser_terms_without_blocking": (
                unblocked_phase_terms["post_disperser"]
            ),
            "emissivity_group_counts": counts,
            "pre_disperser_output_equiv": pre_total,
            "post_disperser_after_qe": post_total,
            "post_disperser_without_blocking_after_qe": post_unblocked_total,
            "post_disperser_blocked_delta": post_blocked_delta,
            "post_disperser_extract_equiv_rate_ph_s": diffuse_extract_equiv,
            "post_disperser_extract_equiv_without_blocking_rate_ph_s": (
                diffuse_unblocked_extract_equiv
            ),
            "post_disperser_rate_ph_s_pix": diffuse_rate_per_pix,
            "post_disperser_without_blocking_rate_ph_s_pix": (
                diffuse_unblocked_rate_per_pix
            ),
            "diffuse_extraction_pixels": float(diffuse_extraction_pixels),
            "diffuse_extraction_note": (
                "Integrated post-disperser diffuse background scaled to "
                f"{diffuse_extraction_pixels:g} image-plane pixel(s) per "
                "wavelength sample. Adjust diffuse_extraction_pixels to match "
                "the extraction footprint."
            ),
            "detector_qe": qe_values,
            "detector_qe_accounting": detector_qe_accounting_summary(detector_qe),
            "detector_qe_note": "throughput only; detector emissivity is not modeled",
        }

    return {"wave_nm": wave_nm, "channels": channels, "details": Table(rows=details)}


def evaluate_emission_density(surface: Any, wave: u.Quantity) -> u.Quantity | None:
    """Evaluate a ScopeSim surface emission curve on ``wave``."""
    emission = getattr(surface, "emission", None)
    if emission is None:
        return None
    values = emission(wave)
    if not isinstance(values, u.Quantity):
        values = values * PHOTLAM
    return values


def surface_list_post_disperser_diffuse_terms(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    qe_values: np.ndarray | None = None,
    emission_phase: str = "post_disperser",
) -> tuple[OrderedDict[str, u.Quantity], list[dict[str, Any]]]:
    """Return post-disperser diffuse spectra grouped by optics metadata.

    This is the physical counterpart to ``surface_list_emissivity_terms``. It
    follows ScopeSim's downstream-emission bookkeeping, but only surfaces tagged
    with ``emission_phase == "post_disperser"`` are returned as image-plane
    diffuse background candidates.
    """
    rows = optical_surface_rows(
        [{
            "selector": surface_list,
            "selector_name": effect_name(surface_list),
            "effect": surface_list,
        }],
        wave,
        groups=groups,
    )
    return optical_surface_post_disperser_diffuse_terms(
        rows, wave, qe_values=qe_values, emission_phase=emission_phase,
    )


def optical_surface_post_disperser_diffuse_terms(
    rows: list[dict[str, Any]],
    wave: u.Quantity,
    qe_values: np.ndarray | None = None,
    emission_phase: str = "post_disperser",
) -> tuple[OrderedDict[str, u.Quantity], list[dict[str, Any]]]:
    """Return post-disperser diffuse spectra from flattened surface rows."""
    for row in rows:
        phase_name = row["emission_phase"]
        if phase_name not in {"pre_disperser", "post_disperser", "none"}:
            raise ValueError(
                f"Unknown emission_phase {phase_name!r} for "
                f"{row['surface_name']!r}; expected pre_disperser, "
                "post_disperser, or none."
            )

    downstream = [np.ones(wave.size, dtype=float) for _ in range(len(rows) + 1)]
    for idx in range(len(rows) - 1, -1, -1):
        downstream[idx] = downstream[idx + 1] * rows[idx]["action_values"]

    if qe_values is None:
        qe_values = np.ones(wave.size, dtype=float)

    grouped: OrderedDict[str, u.Quantity] = OrderedDict()
    details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        is_diffuse = row["emission_phase"] == emission_phase
        contribution = None
        if is_diffuse and row["emission_values"] is not None:
            contribution = row["emission_values"] * downstream[idx + 1] * qe_values
            group_name = row["group"]
            grouped[group_name] = (
                contribution if group_name not in grouped
                else grouped[group_name] + contribution
            )

        peak = np.nan
        if contribution is not None:
            peak = float(np.nanmax(_as_float_array(contribution)))
        details.append({
            "component": row["component"],
            "surface": row["surface_name"],
            "group": row["group"],
            "emission_phase": row["emission_phase"],
            "action": row["action"],
            "temperature": row["temperature"],
            "included_as_diffuse": is_diffuse,
            "qe_applied": is_diffuse,
            "peak_after_qe": peak,
        })

    return grouped, details


def _image_plane_pixel_area(ztrain: Any, image_plane_id: int) -> u.Quantity:
    from scopesim.effects.illumination import image_plane_pixel_area

    return image_plane_pixel_area(
        ztrain.image_planes[image_plane_id].header,
        ztrain.cmds,
    )


def _telescope_area(ztrain: Any) -> u.Quantity:
    from scopesim.utils import from_currsys, quantify

    return quantify(from_currsys("!TEL.area", ztrain.cmds), u.m**2)


def build_post_disperser_diffuse_background_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
    blocking_component_names: tuple[str, ...] = ("ir_blocking_filter_selector",),
) -> dict[str, Any]:
    """Build image-plane post-disperser diffuse background data."""
    from scopesim.effects.illumination import integrate_spectral_background

    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)
    positional_qe_by_aperture = positional_qe_by_aperture or {}
    blocked_components = set(blocking_component_names)
    telescope_area = _telescope_area(ztrain)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    qe_selector = _get_qe_selector(
        ztrain, qe_selector_name, active_only=active_only,
    )
    trace_list = get_effect(ztrain, "trace_list_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    details: list[dict[str, Any]] = []
    for aperture_id in aperture_ids:
        detector_qe = resolve_effect(qe_selector, aperture_id)
        traces = traces_for_aperture.get(aperture_id, [])
        image_plane_id = (
            int(traces[0].meta["image_plane_id"]) if traces else aperture_id
        )
        image_pixel_area = _image_plane_pixel_area(ztrain, image_plane_id)
        image_plane = (
            ztrain.image_planes[image_plane_id]
            if image_plane_id < len(ztrain.image_planes)
            else None
        )
        qe_values = effective_diffuse_qe(
            detector_qe,
            wave,
            positional_qe=positional_qe_by_aperture.get(aperture_id),
            footprint=image_plane,
        )
        components = channel_optical_components(
            ztrain, aperture_id, qe_selector=qe_selector,
        )
        surface_rows = optical_surface_rows(components, wave, groups=groups)
        spectra, surface_details = optical_surface_post_disperser_diffuse_terms(
            surface_rows,
            wave,
            qe_values=qe_values,
        )
        unblocked_rows = _rows_excluding_components(
            surface_rows, blocked_components,
        )
        spectra_unblocked, _surface_details_unblocked = (
            optical_surface_post_disperser_diffuse_terms(
                unblocked_rows,
                wave,
                qe_values=qe_values,
            )
        )
        rates = OrderedDict(
            (name, integrate_spectral_background(
                spectrum,
                wave,
                telescope_area=telescope_area,
                image_pixel_area=image_pixel_area,
            ))
            for name, spectrum in spectra.items()
        )
        rates_unblocked = OrderedDict(
            (name, integrate_spectral_background(
                spectrum,
                wave,
                telescope_area=telescope_area,
                image_pixel_area=image_pixel_area,
            ))
            for name, spectrum in spectra_unblocked.items()
        )
        total_spectrum = _sum_quantity_terms(spectra)
        total_spectrum_unblocked = _sum_quantity_terms(spectra_unblocked)
        total_rate = float(np.sum(list(rates.values()))) if rates else 0.0
        total_rate_unblocked = (
            float(np.sum(list(rates_unblocked.values())))
            if rates_unblocked else 0.0
        )
        label = channel_label(aperture_id, traces)

        for row in surface_details:
            row["aperture_id"] = aperture_id
            row["image_plane_id"] = image_plane_id
            row["channel"] = label
            details.append(row)

        channels[aperture_id] = {
            "label": label,
            "image_plane_id": image_plane_id,
            "trace_wave_min_nm": (
                min(_trace_wave_nm(trace.wave_min) for trace in traces)
                if traces else np.nan
            ),
            "trace_wave_max_nm": (
                max(_trace_wave_nm(trace.wave_max) for trace in traces)
                if traces else np.nan
            ),
            "spectra": spectra,
            "spectra_without_blocking": spectra_unblocked,
            "total_spectrum": total_spectrum,
            "total_spectrum_without_blocking": total_spectrum_unblocked,
            "rates_ph_s_pix": rates,
            "rates_without_blocking_ph_s_pix": rates_unblocked,
            "total_rate_ph_s_pix": total_rate,
            "total_rate_without_blocking_ph_s_pix": total_rate_unblocked,
            "blocking_delta_rate_ph_s_pix": total_rate_unblocked - total_rate,
            "detector_qe": qe_values,
            "detector_qe_accounting": detector_qe_accounting_summary(detector_qe),
            "pixel_area": image_pixel_area,
            "telescope_area": telescope_area,
            "detector_qe_note": "throughput only; detector emissivity is not modeled",
        }

    return {"wave_nm": wave_nm, "channels": channels, "details": Table(rows=details)}


def detector_qe_accounting_table(
    transmission_data: Mapping[str, Any] | None = None,
    emissivity_data: Mapping[str, Any] | None = None,
    post_diffuse_data: Mapping[str, Any] | None = None,
) -> Table:
    """Return detector-QE accounting rows for notebook validation paths."""
    datasets = [
        ("transmission_trace_mapped", transmission_data),
        ("emissivity_sanity", emissivity_data),
        ("post_disperser_diffuse", post_diffuse_data),
    ]
    rows: list[dict[str, Any]] = []
    for path_name, data in datasets:
        if data is None:
            continue
        for aperture_id, channel in data["channels"].items():
            accounting = channel.get("detector_qe_accounting", {})
            rows.append({
                "path": path_name,
                "aperture_id": int(aperture_id),
                "channel": channel["label"],
                "qe_model": accounting.get("qe_model", ""),
                "trace_mapped_qe": accounting.get("trace_mapped_qe", ""),
                "transmission_plot_qe": accounting.get("transmission_plot_qe", ""),
                "diffuse_qe": accounting.get("diffuse_qe", ""),
                "diffuse_position_samples": accounting.get(
                    "diffuse_position_samples",
                ),
                "order_qe_methods": ", ".join(
                    channel.get("order_detector_qe_methods", []),
                ),
                "position_axis": accounting.get("position_axis"),
                "position_min": accounting.get("position_min"),
                "position_max": accounting.get("position_max"),
            })
    return Table(rows=rows)


def post_disperser_diffuse_effect_consistency_table(
    ztrain: Any,
    helper_data: Mapping[str, Any],
    *,
    effect_display_name: str = "post_echelle_diffuse_background_selector",
    match_effect_grid: bool = True,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
) -> Table:
    """Compare notebook helper rates to the configured image-plane effect.

    ``helper_data`` is normally the output of
    :func:`build_post_disperser_diffuse_background_data`. The ``matched`` rate
    is recomputed on the effect's own wavelength grid when possible, so the
    table separates wavelength-sampling differences from real wiring problems.
    """
    selector = get_effect(ztrain, effect_display_name, active_only=False)
    matched_data_by_image_plane = {}
    if match_effect_grid:
        matched_data_by_image_plane = _matched_post_diffuse_data_by_image_plane(
            ztrain,
            helper_data,
            selector,
            qe_selector_name=qe_selector_name,
            active_only=active_only,
        )

    rows = []
    for aperture_id, channel in helper_data["channels"].items():
        image_plane_id = int(channel["image_plane_id"])
        effect = resolve_effect(selector, image_plane_id)
        effect_rate = float(effect.background_value(ztrain.image_planes[image_plane_id]))
        helper_rate = float(channel["total_rate_ph_s_pix"])
        matched_channel = matched_data_by_image_plane.get(image_plane_id, {}).get(
            "channels", {},
        ).get(aperture_id)
        matched_rate = (
            float(matched_channel["total_rate_ph_s_pix"])
            if matched_channel is not None
            else np.nan
        )
        rows.append({
            "aperture_id": int(aperture_id),
            "channel": channel["label"],
            "image_plane_id": image_plane_id,
            "effect_class": effect.__class__.__name__,
            "effect_included": bool(getattr(effect, "include", True)),
            "surface_file": effect.meta.get("filename", ""),
            "detector_qe_file": effect.meta.get("detector_qe_filename", ""),
            "pixel_area_arcsec2": channel["pixel_area"].to_value(u.arcsec**2),
            "helper_rate_ph_s_pix": helper_rate,
            "effect_rate_ph_s_pix": effect_rate,
            "helper_rel_delta": _relative_delta(helper_rate, effect_rate),
            "matched_helper_rate_ph_s_pix": matched_rate,
            "matched_rel_delta": _relative_delta(matched_rate, effect_rate),
        })

    return Table(rows=rows)


def validate_post_disperser_diffuse_effect_consistency(
    table: Table,
    *,
    rtol: float = 1e-6,
    use_matched: bool = True,
) -> None:
    """Validate helper/effect agreement for post-disperser diffuse background."""
    delta_col = "matched_rel_delta" if use_matched else "helper_rel_delta"
    finite = np.isfinite(table[delta_col])
    if not np.all(finite):
        bad = table[~finite]
        raise ValueError(f"Non-finite {delta_col} rows: {bad}")

    abs_delta = np.abs(np.asarray(table[delta_col], dtype=float))
    if np.nanmax(abs_delta) > rtol:
        bad = table[abs_delta > rtol]
        raise ValueError(
            f"Post-disperser diffuse helper/effect mismatch above {rtol}: {bad}"
        )


def _matched_post_diffuse_data_by_image_plane(
    ztrain: Any,
    helper_data: Mapping[str, Any],
    selector: Any,
    *,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
) -> dict[int, Mapping[str, Any]]:
    matched = {}
    cache = {}
    for channel in helper_data["channels"].values():
        image_plane_id = int(channel["image_plane_id"])
        effect = resolve_effect(selector, image_plane_id)
        if not hasattr(effect, "_waveset"):
            continue
        wave_nm = effect._waveset().to(u.nm)
        cache_key = tuple(np.round(wave_nm.to_value(u.nm), 12))
        if cache_key not in cache:
            cache[cache_key] = build_post_disperser_diffuse_background_data(
                ztrain,
                wave_nm=wave_nm,
                qe_selector_name=qe_selector_name,
                active_only=active_only,
            )
        matched[image_plane_id] = cache[cache_key]
    return matched


def _relative_delta(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    if reference == 0:
        return 0.0 if value == 0 else np.inf
    return (value - reference) / reference


def copy_image_plane_data(ztrain: Any) -> dict[int, np.ndarray]:
    """Return detached copies of populated optical-train image planes."""
    copies = {}
    for image_plane_id, image_plane in enumerate(ztrain.image_planes):
        if image_plane is None:
            continue
        try:
            data = _image_plane_array(image_plane)
        except ValueError:
            continue
        copies[int(image_plane_id)] = np.array(data, dtype=float, copy=True)
    return copies


def image_plane_delta_summary(
    with_effect: Any,
    without_effect: Any,
    *,
    expected_rates: Mapping[str, Any] | Mapping[int, float] | Table | None = None,
    expected_rate_column: str | None = None,
    image_plane_ids: list[int] | np.ndarray | None = None,
) -> Table:
    """Summarize image-plane differences from toggling a scalar background.

    ``with_effect`` and ``without_effect`` may be image-plane objects, HDUs,
    arrays, mappings keyed by image-plane id, or an optical train. The returned
    table is intended for checks such as post-disperser diffuse emissivity,
    where the expected image-plane signature is a uniform additive offset.
    """
    with_planes = _image_plane_items(with_effect, image_plane_ids)
    without_planes = _image_plane_items(without_effect, image_plane_ids)
    if set(with_planes) != set(without_planes):
        raise ValueError(
            "with_effect and without_effect have different image-plane ids: "
            f"{sorted(with_planes)} != {sorted(without_planes)}"
        )

    expected_by_plane = _expected_rates_by_image_plane(
        expected_rates, expected_rate_column,
    )
    rows = []
    for image_plane_id in sorted(with_planes):
        enabled = with_planes[image_plane_id]
        disabled = without_planes[image_plane_id]
        if enabled.shape != disabled.shape:
            raise ValueError(
                f"Image plane {image_plane_id} shape mismatch: "
                f"{enabled.shape} != {disabled.shape}"
            )

        delta = enabled - disabled
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size == 0:
            raise ValueError(f"Image plane {image_plane_id} has no finite pixels.")

        mean_delta = float(np.mean(finite_delta))
        std_delta = float(np.std(finite_delta))
        min_delta = float(np.min(finite_delta))
        max_delta = float(np.max(finite_delta))
        median_delta = float(np.median(finite_delta))
        expected_rate = expected_by_plane.get(image_plane_id, np.nan)
        rows.append({
            "image_plane_id": int(image_plane_id),
            "shape": "x".join(str(value) for value in enabled.shape),
            "n_pixels": int(finite_delta.size),
            "mean_delta_ph_s_pix": mean_delta,
            "median_delta_ph_s_pix": median_delta,
            "std_delta_ph_s_pix": std_delta,
            "min_delta_ph_s_pix": min_delta,
            "max_delta_ph_s_pix": max_delta,
            "peak_to_peak_delta_ph_s_pix": max_delta - min_delta,
            "std_over_abs_mean": _ratio_to_abs_reference(std_delta, mean_delta),
            "expected_rate_ph_s_pix": expected_rate,
            "mean_minus_expected_ph_s_pix": mean_delta - expected_rate,
            "expected_rel_delta": _relative_delta(mean_delta, expected_rate),
        })

    return Table(rows=rows)


def validate_image_plane_delta_summary(
    table: Table,
    *,
    expected_rtol: float = 1e-3,
    uniformity_rtol: float = 1e-4,
    uniformity_atol: float = 1e-6,
    require_expected: bool = True,
) -> None:
    """Validate that an image-plane delta is uniform and rate-consistent."""
    if len(table) == 0:
        raise ValueError("Image-plane delta summary is empty.")

    expected_rel_delta = np.asarray(table["expected_rel_delta"], dtype=float)
    if require_expected and not np.all(np.isfinite(expected_rel_delta)):
        bad = table[~np.isfinite(expected_rel_delta)]
        raise ValueError(f"Missing expected image-plane rates: {bad}")

    finite_expected = np.isfinite(expected_rel_delta)
    if np.any(np.abs(expected_rel_delta[finite_expected]) > expected_rtol):
        bad = table[
            finite_expected & (np.abs(expected_rel_delta) > expected_rtol)
        ]
        raise ValueError(
            f"Image-plane delta differs from expected rate above "
            f"{expected_rtol}: {bad}"
        )

    std_delta = np.asarray(table["std_delta_ph_s_pix"], dtype=float)
    mean_delta = np.asarray(table["mean_delta_ph_s_pix"], dtype=float)
    tolerance = uniformity_atol + uniformity_rtol * np.abs(mean_delta)
    if np.any(std_delta > tolerance):
        bad = table[std_delta > tolerance]
        raise ValueError(
            "Image-plane delta is not spatially uniform within "
            f"atol={uniformity_atol}, rtol={uniformity_rtol}: {bad}"
        )


def _image_plane_items(
    planes: Any,
    image_plane_ids: list[int] | np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    if hasattr(planes, "image_planes"):
        planes = planes.image_planes

    if isinstance(planes, Mapping):
        items = {
            int(image_plane_id): _image_plane_array(plane)
            for image_plane_id, plane in planes.items()
        }
        if image_plane_ids is None:
            return items
        missing = [
            int(image_plane_id) for image_plane_id in image_plane_ids
            if int(image_plane_id) not in items
        ]
        if missing:
            raise ValueError(f"Missing image-plane ids: {missing}")
        return {
            int(image_plane_id): items[int(image_plane_id)]
            for image_plane_id in image_plane_ids
        }

    if (
        isinstance(planes, np.ndarray)
        or hasattr(planes, "hdu")
        or hasattr(planes, "data")
    ):
        planes = [planes]

    values = list(planes)
    if image_plane_ids is None:
        image_plane_ids = np.arange(len(values), dtype=int)
    if len(image_plane_ids) != len(values):
        raise ValueError(
            f"Expected {len(values)} image-plane ids, got {len(image_plane_ids)}."
        )
    return {
        int(image_plane_id): _image_plane_array(plane)
        for image_plane_id, plane in zip(image_plane_ids, values, strict=True)
    }


def _image_plane_array(image_plane: Any) -> np.ndarray:
    if image_plane is None:
        raise ValueError("Image plane is None.")
    if hasattr(image_plane, "hdu") and image_plane.hdu is not None:
        data = image_plane.hdu.data
    elif hasattr(image_plane, "data"):
        data = image_plane.data
    else:
        data = image_plane
    if data is None:
        raise ValueError(f"Image plane {image_plane!r} has no data array.")
    return np.asarray(data, dtype=float)


def _expected_rates_by_image_plane(
    expected_rates: Mapping[str, Any] | Mapping[int, float] | Table | None,
    expected_rate_column: str | None,
) -> dict[int, float]:
    if expected_rates is None:
        return {}

    if isinstance(expected_rates, Table):
        if "image_plane_id" not in expected_rates.colnames:
            raise ValueError("Expected-rate table needs an image_plane_id column.")
        rate_col = expected_rate_column or _first_available_column(
            expected_rates,
            (
                "effect_rate_ph_s_pix",
                "matched_helper_rate_ph_s_pix",
                "helper_rate_ph_s_pix",
                "total_rate_ph_s_pix",
            ),
        )
        return {
            int(row["image_plane_id"]): float(row[rate_col])
            for row in expected_rates
        }

    if "channels" in expected_rates:
        return {
            int(channel["image_plane_id"]): float(channel["total_rate_ph_s_pix"])
            for channel in expected_rates["channels"].values()
        }

    return {
        int(image_plane_id): float(rate)
        for image_plane_id, rate in expected_rates.items()
    }


def _first_available_column(table: Table, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in table.colnames:
            return candidate
    raise ValueError(
        "Expected-rate table has none of these columns: "
        f"{', '.join(candidates)}"
    )


def _ratio_to_abs_reference(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    if reference == 0:
        return 0.0 if value == 0 else np.inf
    return value / abs(reference)


def _sum_quantity_terms(terms: Mapping[str, u.Quantity]) -> u.Quantity | None:
    total = None
    for values in terms.values():
        total = values if total is None else total + values
    return total


def _rows_excluding_components(
    rows: list[dict[str, Any]],
    excluded_components: set[str],
) -> list[dict[str, Any]]:
    if not excluded_components:
        return rows
    return [
        row for row in rows
        if str(row["component"]) not in excluded_components
    ]


def _trace_wave_nm(value: Any) -> float:
    wave = value if isinstance(value, u.Quantity) else value * u.um
    return float(wave.to_value(u.nm))


def validate_post_disperser_diffuse_background_data(data: Mapping[str, Any]) -> None:
    """Validate post-disperser diffuse background helper output."""
    for aperture_id, channel in data["channels"].items():
        if not channel["spectra"]:
            raise ValueError(
                f"No post-disperser diffuse spectra for aperture_id={aperture_id}"
            )
        if channel["total_rate_ph_s_pix"] < -1e-12:
            raise ValueError(
                f"Negative diffuse background for aperture_id={aperture_id}: "
                f"{channel['total_rate_ph_s_pix']}"
            )
        if "detector_emissivity" in channel:
            raise ValueError("Detector emissivity must not be included.")


def source_position_table(source_or_table: Any) -> Table:
    """Return the point-source position table from a Source or Table."""
    if isinstance(source_or_table, Table):
        table = source_or_table
    elif hasattr(source_or_table, "fields"):
        table_fields = [
            field.field for field in source_or_table.fields
            if hasattr(field, "field") and isinstance(field.field, Table)
        ]
        if len(table_fields) != 1:
            raise ValueError(
                "Expected exactly one table source field; found "
                f"{len(table_fields)}."
            )
        table = table_fields[0]
    else:
        raise TypeError("Expected a ScopeSim Source or astropy Table.")

    for column in ("x", "y"):
        if column not in table.colnames:
            raise ValueError(f"Source position table has no {column!r} column.")
    return table


def slit_pair_status_table(
    source_or_table: Any,
    *,
    slit_width: u.Quantity = 0.7 * u.arcsec,
    slit_length: u.Quantity = 10.0 * u.arcsec,
) -> Table:
    """Return point-source positions and whether each falls inside the slit."""
    table = source_position_table(source_or_table)
    x = u.Quantity(table["x"]).to(u.arcsec)
    y = u.Quantity(table["y"]).to(u.arcsec)
    width = u.Quantity(slit_width).to(u.arcsec)
    length = u.Quantity(slit_length).to(u.arcsec)
    in_slit = (np.abs(x) <= 0.5 * width) & (np.abs(y) <= 0.5 * length)

    labels = (
        [str(value) for value in table["label"]]
        if "label" in table.colnames
        else [f"source_{idx}" for idx in range(len(table))]
    )
    rows = []
    for idx, label in enumerate(labels):
        rows.append({
            "source_index": idx,
            "label": label,
            "x_arcsec": x[idx].to_value(u.arcsec),
            "y_arcsec": y[idx].to_value(u.arcsec),
            "in_slit": bool(in_slit[idx]),
        })
    return Table(rows=rows)


def slit_adc_psf_status_table(
    ztrain: Any,
    *,
    slit_selector_name: str = "slitwheel_selector",
) -> Table:
    """Return compact slit/ADC/PSF settings relevant to source placement.

    This is a notebook-facing context table. It intentionally reports active
    settings and effect names; it does not attempt to validate the underlying
    physical model.
    """
    rows: list[dict[str, Any]] = []
    rows.extend(_slit_setting_rows(ztrain, slit_selector_name))
    rows.extend(_adc_surface_rows(ztrain))
    rows.extend(_keyword_effect_rows(
        ztrain,
        category="atmospheric dispersion",
        keywords=("adcshift", "atmosphericdispersion", "dispersioncorrection"),
    ))
    rows.extend(_keyword_effect_rows(
        ztrain,
        category="psf",
        keywords=("psf", "seeing", "moffat", "diffraction", "aoenhanceable"),
    ))
    if not any(row["category"] == "psf" for row in rows):
        rows.append({
            "category": "psf",
            "scope": "global",
            "name": "none active",
            "class": "",
            "setting": "",
            "note": "No active PSF-like effect found in this optical train.",
        })
    if not any(row["category"] == "atmospheric dispersion" for row in rows):
        rows.append({
            "category": "atmospheric dispersion",
            "scope": "global",
            "name": "none active",
            "class": "",
            "setting": "",
            "note": (
                "No active atmospheric-dispersion correction effect found. "
                "ADC-named optical surfaces, if listed, are throughput/emissivity "
                "surfaces rather than a shift/correction model."
            ),
        })
    return Table(rows=rows)


def _slit_setting_rows(ztrain: Any, selector_name: str) -> list[dict[str, Any]]:
    try:
        selector = get_effect(ztrain, selector_name)
    except ValueError:
        return [{
            "category": "slit",
            "scope": "all apertures",
            "name": selector_name,
            "class": "",
            "setting": "",
            "note": "No active slit selector found.",
        }]

    groups: OrderedDict[tuple[str, str, str], list[Any]] = OrderedDict()
    for selector_value, effect in sorted(selector.wheel_effects.items()):
        current_slit = _resolved_for_display(
            getattr(effect, "meta", {}).get("current_slit", ""),
            ztrain.cmds,
        )
        filename_format = getattr(effect, "meta", {}).get("filename_format", "")
        key = (effect.__class__.__name__, str(current_slit), str(filename_format))
        groups.setdefault(key, []).append(selector_value)

    rows = []
    for (class_name, current_slit, filename_format), selector_values in groups.items():
        rows.append({
            "category": "slit",
            "scope": _selector_scope(
                selector.meta.get("selector_key", "selector"),
                selector_values,
            ),
            "name": selector_name,
            "class": class_name,
            "setting": f"current_slit={current_slit}",
            "note": f"filename_format={filename_format}",
        })
    return rows


def _adc_surface_rows(ztrain: Any) -> list[dict[str, Any]]:
    rows = []
    for parent in active_effects(ztrain):
        if not hasattr(parent, "wheel_effects"):
            continue
        selector_key = getattr(parent, "meta", {}).get("selector_key", "selector")
        grouped: OrderedDict[tuple[str, tuple[str, ...]], list[Any]] = OrderedDict()
        for selector_value, effect in sorted(parent.wheel_effects.items()):
            surface_names = tuple(_matching_surface_names(effect, ("ADC",)))
            if not surface_names:
                continue
            key = (effect.__class__.__name__, surface_names)
            grouped.setdefault(key, []).append(selector_value)

        for (class_name, surface_names), selector_values in grouped.items():
            rows.append({
                "category": "adc optics",
                "scope": _selector_scope(selector_key, selector_values),
                "name": effect_name(parent),
                "class": class_name,
                "setting": ", ".join(surface_names),
                "note": (
                    "ADC is represented here as optical surfaces; this row is "
                    "not an atmospheric-dispersion correction toggle."
                ),
            })
    return rows


def _keyword_effect_rows(
    ztrain: Any,
    *,
    category: str,
    keywords: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for effect in active_effects(ztrain):
        name = effect_name(effect)
        class_name = effect.__class__.__name__
        haystack = f"{name} {class_name}".lower()
        if not any(keyword.lower() in haystack for keyword in keywords):
            continue
        rows.append({
            "category": category,
            "scope": "global",
            "name": name,
            "class": class_name,
            "setting": _resolved_meta_summary(effect, ztrain.cmds),
            "note": "",
        })
    return rows


def _matching_surface_names(effect: Any, patterns: tuple[str, ...]) -> list[str]:
    table = getattr(effect, "table", None)
    if not hasattr(table, "colnames"):
        return []
    name_col = _real_colname("name", table.colnames)
    if name_col is None:
        return []
    pattern_upper = tuple(pattern.upper() for pattern in patterns)
    return [
        str(name)
        for name in table[name_col]
        if any(pattern in str(name).upper() for pattern in pattern_upper)
    ]


def _selector_scope(selector_key: str, values: list[Any]) -> str:
    values_text = ", ".join(str(value) for value in values)
    return f"{selector_key} {values_text}"


def _resolved_for_display(value: Any, cmds: Any) -> Any:
    from scopesim.utils import from_currsys

    try:
        return from_currsys(value, cmds=cmds)
    except Exception:
        return value


def slit_loss_summary_table(rows: list[Mapping[str, Any]]) -> Table:
    """Return slit-throughput summary rows from input/output signal pairs."""
    output_rows = []
    for row in rows:
        input_signal = float(row["input_signal"])
        output_signal = float(row["output_signal"])
        if input_signal == 0:
            throughput = np.nan if output_signal != 0 else 0.0
        else:
            throughput = output_signal / input_signal
        output = dict(row)
        output["throughput"] = throughput
        output_rows.append(output)
    return Table(rows=output_rows)


def _cmd_value(cmds: Any, key: str, default: Any) -> Any:
    value = _resolved_for_display(key, cmds)
    if isinstance(value, str) and value == key:
        return default
    return value


def _cmd_float(cmds: Any, key: str, default: float) -> float:
    value = _cmd_value(cmds, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cmd_quantity(
    cmds: Any,
    key: str,
    default: u.Quantity,
    unit: u.UnitBase,
) -> u.Quantity:
    value = _cmd_value(cmds, key, default)
    try:
        quantity = u.Quantity(value)
        if quantity.unit == u.dimensionless_unscaled:
            quantity = quantity.value * unit
    except Exception:
        quantity = u.Quantity(value, unit)
    return quantity.to(unit)


def _quantity_with_default_unit(value: Any, unit: u.UnitBase) -> u.Quantity:
    quantity = u.Quantity(value)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity.value * unit
    return quantity.to(unit)


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cmds_from_train_or_cmds(train_or_cmds: Any) -> Any:
    return getattr(train_or_cmds, "cmds", train_or_cmds)


def _zenith_angle_from_airmass(airmass: float) -> u.Quantity:
    airmass = max(float(airmass), 1.0)
    return np.arccos(np.clip(1.0 / airmass, 0.0, 1.0)) * u.rad


def _zenith_angle_from_cmds(cmds: Any) -> u.Quantity:
    return _zenith_angle_from_airmass(_cmd_float(cmds, "!OBS.airmass", 1.0))


def _natural_seeing_fwhm(
    wave: u.Quantity,
    seeing: u.Quantity,
    zenith_angle: u.Quantity,
    *,
    pivot: u.Quantity = 500 * u.nm,
) -> u.Quantity:
    z_rad = u.Quantity(zenith_angle).to_value(u.rad)
    return (
        u.Quantity(seeing).to(u.arcsec)
        * (u.Quantity(wave).to(u.nm) / u.Quantity(pivot).to(u.nm)) ** -0.2
        / np.cos(z_rad) ** 0.6
    ).to(u.arcsec)


def _atmospheric_refraction_shift(
    wave: u.Quantity,
    zenith_angle: u.Quantity,
    cmds: Any,
    *,
    wave_ref: u.Quantity = 500 * u.nm,
) -> u.Quantity:
    from scopesim.effects.atmo_dispersion import refractive_index

    temperature = _cmd_quantity(
        cmds, "!ATMO.temperature", 9.0 * u.deg_C, u.deg_C,
    )
    pressure = _cmd_quantity(cmds, "!ATMO.pressure", 0.75 * u.bar, u.bar)
    humidity = _cmd_float(cmds, "!ATMO.humidity", 0.15)
    x_co2 = _cmd_float(cmds, "!ATMO.x_co2", 450.0)
    wave_um = u.Quantity(wave).to(u.um)
    ref_um = u.Quantity(wave_ref).to(u.um)
    delta_n = (
        refractive_index(wave_um, temperature, pressure, humidity, x_co2)
        - refractive_index(ref_um, temperature, pressure, humidity, x_co2)
    )
    shift = 206265 * delta_n * np.tan(u.Quantity(zenith_angle).to_value(u.rad))
    return (shift * u.arcsec).to(u.arcsec)


def _adc_zenith_angle_error(train_or_cmds: Any, default: u.Quantity) -> tuple[u.Quantity, str]:
    if not hasattr(train_or_cmds, "optics_manager"):
        return u.Quantity(default).to(u.deg), "diagnostic ADC residual"

    for effect in active_effects(train_or_cmds):
        class_name = effect.__class__.__name__.lower()
        if class_name != "adcshift":
            continue
        value = getattr(effect, "meta", {}).get("zenith_angle_error")
        if value is None:
            continue
        return u.Quantity(float(_resolved_for_display(value, train_or_cmds.cmds)), u.deg), (
            f"active {effect_name(effect)} zenith-angle residual"
        )

    return u.Quantity(default).to(u.deg), (
        "diagnostic ADC residual; no active ADCShift effect found"
    )


def _moffat_image(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    x0: float,
    y0: float,
    fwhm: float,
    beta: float,
) -> np.ndarray:
    gamma = 0.5 * fwhm / np.sqrt(2 ** (1.0 / beta) - 1.0)
    amplitude = (beta - 1.0) / (np.pi * gamma**2)
    radius2 = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    return amplitude * (1.0 + radius2 / gamma**2) ** -beta


def _scene_image(
    positions: Table,
    wave: u.Quantity,
    shifts: u.Quantity,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    seeing: u.Quantity,
    zenith_angle: u.Quantity,
    beta: float,
) -> np.ndarray:
    x = u.Quantity(positions["x"]).to_value(u.arcsec)
    y = u.Quantity(positions["y"]).to_value(u.arcsec)
    weights = (
        np.asarray(positions["weight"], dtype=float)
        if "weight" in positions.colnames
        else np.ones(len(positions), dtype=float)
    )
    fwhm_values = _natural_seeing_fwhm(wave, seeing, zenith_angle).to_value(u.arcsec)
    shift_values = u.Quantity(shifts).to_value(u.arcsec)

    image = np.zeros_like(x_grid, dtype=float)
    for shift, fwhm in zip(shift_values, fwhm_values, strict=True):
        for xpos, ypos, weight in zip(x, y, weights, strict=True):
            image += weight * _moffat_image(
                x_grid,
                y_grid,
                x0=xpos + shift,
                y0=ypos,
                fwhm=float(fwhm),
                beta=beta,
            )
    return image / max(wave.size, 1)


def _slit_throughput_curve(
    wave: u.Quantity,
    shifts: u.Quantity,
    *,
    seeing: u.Quantity,
    zenith_angle: u.Quantity,
    slit_width: u.Quantity,
    slit_length: u.Quantity,
    beta: float,
    grid_step: u.Quantity,
    fwhm_func: (
        Callable[[u.Quantity, u.Quantity, u.Quantity], u.Quantity] | None
    ) = None,
) -> np.ndarray:
    wave = u.Quantity(wave).to(u.nm)
    shifts_arcsec = u.Quantity(shifts).to_value(u.arcsec)
    if fwhm_func is None:
        fwhm = _natural_seeing_fwhm(wave, seeing, zenith_angle)
    else:
        fwhm = fwhm_func(wave, zenith_angle, seeing)
    fwhm_values = u.Quantity(fwhm).to_value(u.arcsec)
    width = u.Quantity(slit_width).to_value(u.arcsec)
    length = u.Quantity(slit_length).to_value(u.arcsec)
    step = u.Quantity(grid_step).to_value(u.arcsec)

    max_fwhm = float(np.nanmax(fwhm_values))
    max_shift = float(np.nanmax(np.abs(shifts_arcsec)))
    x_extent = max(4.0, width + 8 * max_fwhm + 2 * max_shift)
    y_extent = max(length + 8 * max_fwhm, 1.2 * length)
    x = np.arange(-0.5 * x_extent, 0.5 * x_extent + step, step)
    y = np.arange(-0.5 * y_extent, 0.5 * y_extent + step, step)
    x_grid, y_grid = np.meshgrid(x, y)
    slit_mask = (
        (np.abs(x_grid) <= 0.5 * width)
        & (np.abs(y_grid) <= 0.5 * length)
    )
    pixel_area = step**2

    throughput = np.empty(wave.size, dtype=float)
    for idx, (shift, fwhm) in enumerate(zip(shifts_arcsec, fwhm_values, strict=True)):
        image = _moffat_image(
            x_grid, y_grid, x0=float(shift), y0=0.0,
            fwhm=float(fwhm), beta=beta,
        )
        total = np.sum(image) * pixel_area
        inside = np.sum(image[slit_mask]) * pixel_area
        throughput[idx] = inside / total if total > 0 else np.nan
    return throughput


def _active_ao_enhanceable_psf(train_or_cmds: Any) -> Any | None:
    if not hasattr(train_or_cmds, "optics_manager"):
        return None
    for effect in active_effects(train_or_cmds):
        if effect.__class__.__name__ == "AOEnhanceablePSF":
            return effect
    return None


def _slit_loss_psf_modes(
    train_or_cmds: Any,
    beta: float,
) -> OrderedDict[str, dict[str, Any]]:
    modes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    modes["no_ao"] = {
        "label": "no AO",
        "style": "-",
        "beta": beta,
        "fwhm_func": None,
        "note": "Natural seeing law from current observation settings.",
    }

    effect = _active_ao_enhanceable_psf(train_or_cmds)
    if effect is None or not hasattr(effect, "ao_scale"):
        return modes

    is_absolute = _metadata_bool(
        getattr(effect, "meta", {}).get("is_absolute"),
        default=True,
    )

    def ao_fwhm(
        wave: u.Quantity,
        zenith_angle: u.Quantity,
        seeing: u.Quantity,
    ) -> u.Quantity:
        values = effect.ao_scale(u.Quantity(wave).to(u.um))
        if is_absolute:
            return _quantity_with_default_unit(values, u.arcsec)

        scale = _as_float_array(values)
        return (
            _natural_seeing_fwhm(wave, seeing, zenith_angle)
            * scale
        ).to(u.arcsec)

    modes["ao"] = {
        "label": "AO",
        "style": "--",
        "beta": float(getattr(effect, "alpha", beta) or beta),
        "fwhm_func": ao_fwhm,
        "note": (
            "AO design FWHM from active AOEnhanceablePSF. "
            "Dimensionless absolute AO tables are interpreted as arcsec, "
            "matching ScopeSim's PSF quantification convention. "
            "This diagnostic curve does not change the optical train."
        ),
    }
    return modes


def build_slit_adc_psf_scene_data(
    train_or_cmds: Any,
    sources: Mapping[str, Any],
    *,
    slit_width: u.Quantity,
    slit_length: u.Quantity = 10.0 * u.arcsec,
    wave_nm: u.Quantity | None = None,
    wave_ref: u.Quantity = 500.0 * u.nm,
    adc_zenith_angle_error: u.Quantity = 0.9 * u.deg,
    grid_step: u.Quantity = 0.035 * u.arcsec,
    beta: float = 4.765,
) -> dict[str, Any]:
    """Build PSF/AD slit-scene images for notebook validation.

    The scene images deliberately convolve the point-source scene and apply
    chromatic atmospheric shifts before slit clipping. ScopeSim applies PSF
    convolution later in the detector path, so this helper is a validation
    diagnostic rather than a replacement for the simulated detector image.
    """
    cmds = _cmds_from_train_or_cmds(train_or_cmds)
    wave = (
        np.linspace(310, 980, 41) * u.nm
        if wave_nm is None
        else u.Quantity(wave_nm).to(u.nm)
    )
    seeing = _cmd_quantity(cmds, "!OBS.seeing", 0.6 * u.arcsec, u.arcsec)
    zenith_angle = _zenith_angle_from_cmds(cmds)
    slit_width = u.Quantity(slit_width).to(u.arcsec)
    slit_length = u.Quantity(slit_length).to(u.arcsec)
    adc_error, adc_note = _adc_zenith_angle_error(
        train_or_cmds, adc_zenith_angle_error,
    )

    ad_shift = _atmospheric_refraction_shift(
        wave, zenith_angle, cmds, wave_ref=wave_ref,
    )
    adc_shift = ad_shift - _atmospheric_refraction_shift(
        wave, zenith_angle + adc_error, cmds, wave_ref=wave_ref,
    )
    variants: OrderedDict[str, dict[str, Any]] = OrderedDict({
        "ad_only": {
            "label": "AD only",
            "shift_arcsec": ad_shift,
            "note": "Full atmospheric dispersion relative to wave_ref.",
        },
        "adc_residual": {
            "label": "ADC residual",
            "shift_arcsec": adc_shift,
            "note": adc_note,
        },
    })

    source_tables = OrderedDict(
        (name, source_position_table(source))
        for name, source in sources.items()
    )
    all_x = np.concatenate([
        u.Quantity(table["x"]).to_value(u.arcsec)
        for table in source_tables.values()
    ])
    all_y = np.concatenate([
        u.Quantity(table["y"]).to_value(u.arcsec)
        for table in source_tables.values()
    ])
    max_shift = max(
        float(np.nanmax(np.abs(variant["shift_arcsec"].to_value(u.arcsec))))
        for variant in variants.values()
    )
    max_fwhm = float(np.nanmax(
        _natural_seeing_fwhm(wave, seeing, zenith_angle).to_value(u.arcsec),
    ))
    width = slit_width.to_value(u.arcsec)
    length = slit_length.to_value(u.arcsec)
    x_extent = max(
        3.5,
        width + 2 * max_shift + 6 * max_fwhm,
        float(np.ptp(all_x)) + 2 * max_shift + 4 * max_fwhm,
    )
    y_extent = max(
        1.12 * length,
        float(np.ptp(all_y)) + 6 * max_fwhm,
    )
    step = grid_step.to_value(u.arcsec)
    x = np.arange(-0.5 * x_extent, 0.5 * x_extent + step, step)
    y = np.arange(-0.5 * y_extent, 0.5 * y_extent + step, step)
    x_grid, y_grid = np.meshgrid(x, y)

    scenarios: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for name, table in source_tables.items():
        status = slit_pair_status_table(
            table, slit_width=slit_width, slit_length=slit_length,
        )
        images = OrderedDict(
            (
                key,
                _scene_image(
                    table,
                    wave,
                    variant["shift_arcsec"],
                    x_grid,
                    y_grid,
                    seeing=seeing,
                    zenith_angle=zenith_angle,
                    beta=beta,
                ),
            )
            for key, variant in variants.items()
        )
        scenarios[name] = {
            "positions": table,
            "status": status,
            "images": images,
        }

    return {
        "wave_nm": wave,
        "wave_ref_nm": u.Quantity(wave_ref).to(u.nm),
        "seeing_arcsec": seeing,
        "zenith_angle_deg": zenith_angle.to(u.deg),
        "airmass": _cmd_float(cmds, "!OBS.airmass", 1.0),
        "slit_width_arcsec": slit_width,
        "slit_length_arcsec": slit_length,
        "x_arcsec": x * u.arcsec,
        "y_arcsec": y * u.arcsec,
        "variants": variants,
        "scenarios": scenarios,
        "notes": [
            "Images show PSF convolution and chromatic shift before slit clipping.",
            "ScopeSim detector images apply the PSF after aperture clipping.",
            "The residual ADC curve follows the ADCShift zenith-angle-error convention.",
        ],
    }


def build_slit_loss_data(
    train_or_cmds: Any,
    *,
    arms: Mapping[str, tuple[u.Quantity, u.Quantity, str]] | None = None,
    n_wave: int = 180,
    slit_length: u.Quantity = 10.0 * u.arcsec,
    wave_ref: u.Quantity = 500.0 * u.nm,
    adc_zenith_angle_error: u.Quantity = 0.9 * u.deg,
    grid_step: u.Quantity = 0.04 * u.arcsec,
    beta: float = 4.765,
) -> dict[str, Any]:
    """Return centered point-source slit-loss curves by arm."""
    cmds = _cmds_from_train_or_cmds(train_or_cmds)
    arms = arms or OrderedDict({
        "VIS": (310 * u.nm, 980 * u.nm, "!INST.vis_curr_slit"),
        "NIR": (980 * u.nm, 2450 * u.nm, "!INST.nir_curr_slit"),
    })
    seeing = _cmd_quantity(cmds, "!OBS.seeing", 0.6 * u.arcsec, u.arcsec)
    psf_modes = _slit_loss_psf_modes(train_or_cmds, beta)
    adc_error, adc_note = _adc_zenith_angle_error(
        train_or_cmds, adc_zenith_angle_error,
    )
    curve_specs = OrderedDict({
        "zenith": {
            "label": "zenith",
            "zenith_angle": 0.0 * u.deg,
            "mode": "ad",
            "color": "tab:blue",
            "note": "No chromatic displacement at zenith.",
        },
        "elevation_60_ad_only": {
            "label": "60 deg elevation, AD only",
            "zenith_angle": 30.0 * u.deg,
            "mode": "ad",
            "color": "tab:orange",
            "note": "Full atmospheric dispersion before slit clipping.",
        },
        "elevation_60_adc_residual": {
            "label": "60 deg elevation, ADC residual",
            "zenith_angle": 30.0 * u.deg,
            "mode": "adc",
            "color": "tab:green",
            "note": adc_note,
        },
    })

    arm_data: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for arm_name, (wave_min, wave_max, slit_key) in arms.items():
        wave = np.linspace(
            u.Quantity(wave_min).to_value(u.nm),
            u.Quantity(wave_max).to_value(u.nm),
            n_wave,
        ) * u.nm
        slit_width = _cmd_quantity(cmds, slit_key, 0.7 * u.arcsec, u.arcsec)
        curves: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for psf_name, psf_spec in psf_modes.items():
            for curve_name, spec in curve_specs.items():
                zenith_angle = spec["zenith_angle"]
                ad_shift = _atmospheric_refraction_shift(
                    wave, zenith_angle, cmds, wave_ref=wave_ref,
                )
                if spec["mode"] == "adc":
                    shifts = ad_shift - _atmospheric_refraction_shift(
                        wave,
                        zenith_angle + adc_error,
                        cmds,
                        wave_ref=wave_ref,
                    )
                else:
                    shifts = ad_shift
                throughput = _slit_throughput_curve(
                    wave,
                    shifts,
                    seeing=seeing,
                    zenith_angle=zenith_angle,
                    slit_width=slit_width,
                    slit_length=slit_length,
                    beta=psf_spec["beta"],
                    grid_step=grid_step,
                    fwhm_func=psf_spec["fwhm_func"],
                )
                curves[f"{psf_name}_{curve_name}"] = {
                    "label": f"{psf_spec['label']}, {spec['label']}",
                    "throughput": throughput,
                    "loss": 1.0 - throughput,
                    "shift_arcsec": shifts,
                    "color": spec["color"],
                    "linestyle": psf_spec["style"],
                    "psf_mode": psf_name,
                    "psf_note": psf_spec["note"],
                    "note": spec["note"],
                }
        arm_data[arm_name] = {
            "wave_nm": wave,
            "slit_width_arcsec": slit_width,
            "slit_length_arcsec": u.Quantity(slit_length).to(u.arcsec),
            "curves": curves,
        }

    return {
        "arms": arm_data,
        "seeing_arcsec": seeing,
        "psf_modes": psf_modes,
        "wave_ref_nm": u.Quantity(wave_ref).to(u.nm),
        "adc_zenith_angle_error_deg": adc_error,
        "notes": [
            "Centered point-source loss from a pre-slit Moffat PSF.",
            "This diagnostic exposes slit loss that ScopeSim does not measure directly.",
        ],
    }


def _sum_terms(terms: Mapping[str, np.ndarray], size: int) -> np.ndarray:
    if not terms:
        return np.zeros(size, dtype=float)
    return np.sum(list(terms.values()), axis=0)


def validate_emissivity_sanity_data(data: Mapping[str, Any]) -> None:
    """Validate split emissivity sanity-check data."""
    for aperture_id, channel in data["channels"].items():
        if not channel["pre_disperser_terms"]:
            raise ValueError(
                f"No pre-disperser emissivity terms for aperture_id={aperture_id}"
            )
        if not channel["post_disperser_terms"]:
            raise ValueError(
                f"No post-disperser emissivity terms for aperture_id={aperture_id}"
            )
        checks = [
            ("pre_disperser", channel["pre_disperser_output_equiv"]),
            ("post_disperser_after_qe", channel["post_disperser_after_qe"]),
            ("detector_qe", channel["detector_qe"]),
        ]
        for label, arr in checks:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise ValueError(
                    f"Empty/non-finite {label} curve for aperture_id={aperture_id}"
                )
            if np.nanmin(finite) < -1e-6:
                warnings.warn(
                    f"Negative {label} curve for aperture_id={aperture_id}: "
                    f"{np.nanmin(finite):.3g}",
                    stacklevel=2,
                )

    if "detector_emissivity" in next(iter(data["channels"].values())):
        raise ValueError("Detector emissivity should not be part of QE validation data.")


def detector_geometry_table(ztrain: Any) -> Table:
    """Return detector geometry/settings rows from active DetectorList effects."""
    rows: list[dict[str, Any]] = []
    for effect in active_effects(ztrain):
        if effect.__class__.__name__ != "DetectorList":
            continue
        for row in effect.table:
            rows.append({
                "detector_id": int(_row_scalar(row, "id")),
                "image_plane_id": int(effect.meta.get("image_plane_id")),
                "channel": effect_name(effect).replace("detector_", "").upper(),
                "detector": effect.meta.get("detector", ""),
                "x_size": _row_scalar(row, "x_size"),
                "y_size": _row_scalar(row, "y_size"),
                "pixel_size_mm": _row_scalar(row, "pixel_size"),
                "gain_e_per_adu": _row_scalar(row, "gain"),
            })
    return Table(rows=sorted(rows, key=lambda row: row["detector_id"]))


def _selector_value_for_detector(selector: Any, detector_row: Any) -> int | None:
    selector_key = selector.meta.get("selector_key")
    if selector_key == "detector_id":
        return int(detector_row["detector_id"])
    if selector_key == "aperture_id":
        return int(detector_row["image_plane_id"])
    return None


def _resolved_meta_summary(effect: Any, cmds: Any) -> str:
    from scopesim.utils import from_currsys

    keys = (
        "filename",
        "dit",
        "ndit",
        "value",
        "noise_std",
        "bias",
        "binx",
        "biny",
        "current_slit",
        "fwhm",
        "alpha",
        "strehl",
        "wavelength",
    )
    parts = []
    for key in keys:
        if key not in effect.meta:
            continue
        value = effect.meta[key]
        try:
            value = from_currsys(value, cmds=cmds)
        except Exception:
            pass
        parts.append(f"{key}={value}")
    if not parts:
        return ""
    return ", ".join(parts)


def detector_selector_matrix(ztrain: Any) -> Table:
    """Return detector selector settings resolved for each detector row."""
    detectors = detector_geometry_table(ztrain)
    selectors = [
        effect for effect in active_effects(ztrain)
        if hasattr(effect, "wheel_effects")
    ]
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        for selector in selectors:
            selector_value = _selector_value_for_detector(selector, detector)
            if selector_value is None:
                continue
            try:
                selected_effect = resolve_effect(selector, selector_value)
            except KeyError:
                continue
            rows.append({
                "detector_id": int(detector["detector_id"]),
                "channel": str(detector["channel"]),
                "detector": str(detector["detector"]),
                "selector": effect_name(selector),
                "selector_key": selector.meta.get("selector_key"),
                "selector_value": selector_value,
                "effect_class": selected_effect.__class__.__name__,
                "settings": _resolved_meta_summary(selected_effect, ztrain.cmds),
            })
    return Table(rows=rows)


def detector_background_budget_table(
    ztrain: Any,
    post_diffuse_data: Mapping[str, Any] | None = None,
    post_diffuse_consistency: Table | None = None,
) -> Table:
    """Return per-detector background and noise budget terms.

    The additive signal columns follow the detector pipeline: image-plane
    diffuse rates are multiplied by ``DIT * NDIT``; dark current is already a
    detector term in e-/s/pix; read noise is reported as RMS after NDIT.
    Detector QE is already included in the diffuse rate and is not treated as
    an emissive detector source.
    """
    detectors = detector_geometry_table(ztrain)
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        detector_id = int(detector["detector_id"])
        image_plane_id = int(detector["image_plane_id"])
        exposure = _selected_detector_effect(
            ztrain, "exposure_integration_selector", detector,
        )
        dark_current = _selected_detector_effect(
            ztrain, "dark_current_selector", detector,
        )
        read_noise = _selected_detector_effect(
            ztrain, "readout_noise_selector", detector,
        )
        bias = _selected_detector_effect(ztrain, "bias_selector", detector)

        dit = _resolved_detector_meta(exposure, "dit", ztrain.cmds, detector_id)
        ndit = _resolved_detector_meta(exposure, "ndit", ztrain.cmds, detector_id)
        exposure_time = dit * ndit
        diffuse_rate = _post_diffuse_rate_for_image_plane(
            ztrain,
            image_plane_id,
            post_diffuse_data=post_diffuse_data,
            post_diffuse_consistency=post_diffuse_consistency,
        )
        dark_rate = _resolved_detector_meta(
            dark_current, "value", ztrain.cmds, detector_id, default=0.0,
        )
        read_noise_single = _resolved_detector_meta(
            read_noise, "noise_std", ztrain.cmds, detector_id, default=0.0,
        )
        read_ndit = _resolved_detector_meta(
            read_noise, "ndit", ztrain.cmds, detector_id, default=ndit,
        )
        bias_level = _resolved_detector_meta(
            bias, "bias", ztrain.cmds, detector_id, default=0.0,
        )

        diffuse_counts = diffuse_rate * exposure_time
        dark_counts = dark_rate * exposure_time
        additive_signal = diffuse_counts + dark_counts
        full_well = _resolved_detector_cmd_value(
            ztrain.cmds, "!DET.full_well", detector_id, default=np.nan,
        )
        if np.isfinite(full_well) and full_well > 0:
            full_well_fraction = additive_signal / full_well
        else:
            full_well_fraction = np.nan
        diffuse_noise = np.sqrt(max(diffuse_counts, 0.0))
        dark_noise = np.sqrt(max(dark_counts, 0.0))
        read_noise_total = read_noise_single * np.sqrt(read_ndit)
        total_noise = np.sqrt(
            diffuse_noise**2 + dark_noise**2 + read_noise_total**2,
        )

        rows.append({
            "detector_id": detector_id,
            "image_plane_id": image_plane_id,
            "channel": str(detector["channel"]),
            "detector": str(detector["detector"]),
            "dit_s": dit,
            "ndit": ndit,
            "exposure_time_s": exposure_time,
            "post_diffuse_rate_ph_s_pix": diffuse_rate,
            "post_diffuse_e_pix": diffuse_counts,
            "dark_current_e_s_pix": dark_rate,
            "dark_current_e_pix": dark_counts,
            "additive_signal_e_pix": additive_signal,
            "bias_e_pix": bias_level,
            "full_well_e": full_well,
            "signal_fraction_of_full_well": full_well_fraction,
            "saturation_status": _saturation_status(full_well_fraction),
            "read_noise_e_rms": read_noise_total,
            "diffuse_shot_noise_e_rms": diffuse_noise,
            "dark_shot_noise_e_rms": dark_noise,
            "total_noise_e_rms": total_noise,
        })

    return Table(rows=rows)


def _selected_detector_effect(
    ztrain: Any,
    selector_name: str,
    detector_row: Any,
) -> Any | None:
    try:
        selector = get_effect(ztrain, selector_name)
    except ValueError:
        return None
    selector_value = _selector_value_for_detector(selector, detector_row)
    if selector_value is None:
        return None
    try:
        return resolve_effect(selector, selector_value)
    except KeyError:
        return None


def _resolved_detector_meta(
    effect: Any | None,
    key: str,
    cmds: Any,
    detector_id: int,
    *,
    default: float = np.nan,
) -> float:
    from scopesim.utils import from_currsys

    if effect is None or key not in getattr(effect, "meta", {}):
        return float(default)
    value = from_currsys(effect.meta[key], cmds)
    if isinstance(value, Mapping):
        value = from_currsys(value[detector_id], cmds)
    return float(value)


def _resolved_detector_cmd_value(
    cmds: Any,
    key: str,
    detector_id: int,
    *,
    default: float = np.nan,
) -> float:
    from scopesim.utils import from_currsys

    try:
        value = from_currsys(key, cmds)
    except Exception:
        return float(default)

    if isinstance(value, Mapping):
        if detector_id in value:
            value = value[detector_id]
        elif str(detector_id) in value:
            value = value[str(detector_id)]
        else:
            return float(default)
        value = from_currsys(value, cmds)
    elif (
        isinstance(value, (list, tuple, np.ndarray))
        and not hasattr(value, "unit")
    ):
        if len(value) <= detector_id:
            return float(default)
        value = from_currsys(value[detector_id], cmds)

    return float(value)


def _saturation_status(full_well_fraction: float) -> str:
    if not np.isfinite(full_well_fraction):
        return "unknown"
    if full_well_fraction >= 1.0:
        return "saturated"
    if full_well_fraction >= 0.8:
        return "near_saturation"
    return "ok"


def _post_diffuse_rate_for_image_plane(
    ztrain: Any,
    image_plane_id: int,
    *,
    post_diffuse_data: Mapping[str, Any] | None = None,
    post_diffuse_consistency: Table | None = None,
) -> float:
    if post_diffuse_consistency is not None:
        matches = post_diffuse_consistency[
            np.asarray(post_diffuse_consistency["image_plane_id"], dtype=int)
            == image_plane_id
        ]
        if len(matches):
            if (
                "effect_included" in matches.colnames
                and not bool(matches[0]["effect_included"])
            ):
                return 0.0
            return float(matches[0]["effect_rate_ph_s_pix"])

    if post_diffuse_data is not None:
        for channel in post_diffuse_data["channels"].values():
            if int(channel["image_plane_id"]) == image_plane_id:
                return float(channel["total_rate_ph_s_pix"])

    try:
        selector = get_effect(
            ztrain, "post_echelle_diffuse_background_selector", active_only=False,
        )
        if not getattr(selector, "include", True):
            return 0.0
        effect = resolve_effect(selector, image_plane_id)
    except (ValueError, KeyError):
        return 0.0
    if not getattr(effect, "include", True):
        return 0.0
    return float(effect.background_value(ztrain.image_planes[image_plane_id]))


def dichroic_path_throughput(
    dichroic_tree: Any,
    aperture_id: int,
    wave: u.Quantity,
) -> tuple[np.ndarray, OrderedDict[str, np.ndarray]]:
    """Return total and per-component dichroic throughput for an aperture."""
    table = dichroic_tree.table
    id_col = table.colnames[0]
    rows = table[table[id_col] == aperture_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one dichroic row for {id_col}={aperture_id}; found "
            f"{len(rows)}"
        )

    action_lookup = {"T": "transmission", "R": "reflection", "X": None}
    components: OrderedDict[str, np.ndarray] = OrderedDict()
    total = np.ones(wave.size, dtype=float)
    row = rows[0]
    for dichroic_name in table.colnames[1:]:
        action = action_lookup.get(str(row[dichroic_name]))
        if action is None:
            continue
        curve = getattr(dichroic_tree.dichroics[dichroic_name].surface, action)
        values = evaluate_curve(curve, wave)
        components[f"{dichroic_name}:{action[0].upper()}"] = values
        total *= values
    return total, components


def build_transmission_sanity_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
) -> dict[str, Any]:
    """Build channel/order throughput data for transmission sanity plots."""
    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    qe_selector = _get_qe_selector(
        ztrain, qe_selector_name, active_only=active_only,
    )
    trace_list = get_effect(ztrain, "trace_list_analytical")
    trace_eff = get_effect(ztrain, "trace_eff_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for aperture_id in aperture_ids:
        detector_qe = resolve_effect(qe_selector, aperture_id)

        dichroic_total, dichroic_components = dichroic_path_throughput(
            dichroic_tree, aperture_id, wave,
        )
        components = channel_optical_components(
            ztrain, aperture_id, qe_selector=qe_selector,
        )
        surface_rows = optical_surface_rows(components, wave, groups=groups)
        optics_groups, group_counts = optical_surface_group_throughputs(
            surface_rows, wave,
        )
        optics_total = (
            np.prod(list(optics_groups.values()), axis=0)
            if optics_groups
            else np.ones(wave.size)
        )
        image_plane = None
        traces = traces_for_aperture.get(aperture_id, [])
        if traces:
            image_plane_id = int(traces[0].meta["image_plane_id"])
            if (
                hasattr(ztrain, "image_planes")
                and image_plane_id < len(ztrain.image_planes)
            ):
                image_plane = ztrain.image_planes[image_plane_id]

        qe_values = (
            effective_diffuse_qe(detector_qe, wave, footprint=image_plane)
            if getattr(detector_qe, "uses_detector_footprint", False)
            else evaluate_throughput(detector_qe, wave)
        )
        qe_midpoint_values = evaluate_throughput(detector_qe, wave)
        pre_disperser_total = dichroic_total * optics_total

        orders: OrderedDict[str, dict[str, Any]] = OrderedDict()
        qe_methods: set[str] = set()
        for trace in traces:
            order_eff = _as_float_array(
                trace_eff.efficiency_generator(trace.trace_id, wave),
            )
            mask = (wave >= trace.wave_min * u.um) & (wave <= trace.wave_max * u.um)
            order_eff = np.where(mask, order_eff, np.nan)
            order_qe, qe_method = evaluate_trace_detector_qe(
                detector_qe, trace, wave, image_plane=image_plane,
            )
            qe_methods.add(qe_method)
            orders[trace.trace_id] = {
                "disperser": order_eff,
                "detector_qe": order_qe,
                "detector_qe_method": qe_method,
                "total": pre_disperser_total * order_eff * order_qe,
                "wave_min": trace.wave_min * u.um,
                "wave_max": trace.wave_max * u.um,
            }

        channels[aperture_id] = {
            "label": channel_label(aperture_id, traces),
            "dichroic_total": dichroic_total,
            "dichroic_components": dichroic_components,
            "optics_groups": optics_groups,
            "optics_group_counts": group_counts,
            "optics_total": optics_total,
            "detector_qe": qe_values,
            "detector_qe_midpoint": qe_midpoint_values,
            "detector_qe_label": (
                "detector QE axis average"
                if getattr(detector_qe, "uses_detector_footprint", False)
                else "detector QE"
            ),
            "detector_qe_midpoint_label": (
                "detector QE at taper midpoint"
                if getattr(detector_qe, "uses_detector_footprint", False)
                else "detector QE midpoint"
            ),
            "detector_qe_accounting": detector_qe_accounting_summary(detector_qe),
            "order_detector_qe_methods": sorted(qe_methods),
            "pre_disperser_total": pre_disperser_total,
            "orders": orders,
        }

    return {"wave_nm": wave_nm, "channels": channels}


def validate_transmission_sanity_data(data: Mapping[str, Any]) -> None:
    """Validate transmission sanity-check data."""
    for aperture_id, channel in data["channels"].items():
        if not channel["optics_groups"]:
            raise ValueError(f"No optics groups for aperture_id={aperture_id}")
        if not channel["orders"]:
            raise ValueError(f"No echelle orders for aperture_id={aperture_id}")
        arrays = [
            channel["dichroic_total"],
            channel["optics_total"],
            channel["detector_qe"],
            channel["detector_qe_midpoint"],
            channel["pre_disperser_total"],
        ]
        arrays.extend(channel["optics_groups"].values())
        arrays.extend(order["disperser"] for order in channel["orders"].values())
        arrays.extend(order["detector_qe"] for order in channel["orders"].values())
        for arr in arrays:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise ValueError(
                    f"Empty/non-finite curve for aperture_id={aperture_id}"
                )
            if np.nanmin(finite) < -1e-6 or np.nanmax(finite) > 1.5:
                warnings.warn(
                    f"Throughput outside expected range for "
                    f"aperture_id={aperture_id}: "
                    f"{np.nanmin(finite):.3g}..{np.nanmax(finite):.3g}",
                    stacklevel=2,
                )
