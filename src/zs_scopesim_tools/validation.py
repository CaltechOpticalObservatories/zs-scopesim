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

from scopesim.utils import from_currsys

from .plots import (
    plot_detector_background_budget,
    plot_detector_image_grid,
    plot_emissivity_sanity,
    plot_post_disperser_diffuse_background,
    plot_readout_delta_overview,
    plot_readout_cross_dispersion_cut,
    plot_readout_overview,
    plot_slit_adc_psf_scenes,
    plot_slit_loss_by_arm,
    plot_slit_pair_geometry,
    plot_slit_width_loss,
    plot_source,
    plot_resolving_power_echellogram,
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
    """Return included effects from an optical train.

    ScopeSim-refactor candidate: optical-train effect lookup should live on
    the train or optics manager instead of in notebook validation code.
    """
    return [
        eff for eff in ztrain.optics_manager.all_effects
        if getattr(eff, "include", True)
    ]


def get_effect(ztrain: Any, display_name: str, *, active_only: bool = True) -> Any:
    """Fetch one optical-train effect by display name.

    ScopeSim-refactor candidate: exact effect lookup by notebook-facing name is
    a general optical-train inspection primitive.
    """
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
    """Return a plain effect, or one selected entry from a SelectorWheel.

    ScopeSim-refactor candidate: selector-wheel resolution should be exposed by
    ScopeSim's selector effects.
    """
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

def _looks_like_qe_effect(effect: Any) -> bool:
    def _candidate_qe_effects(effect: Any) -> list[Any]:
        if hasattr(effect, "wheel_effects"):
            return list(effect.wheel_effects.values())
        return [effect]

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


def _surface_effect_group_name(
    parent: Any,
    effect: Any,
    component_metadata: Mapping[str, Any] | None = None,
) -> str:
    configured_group = None
    for meta in (getattr(effect, "meta", {}), getattr(parent, "meta", {})):
        group_name = _clean_metadata_value(meta.get("throughput_group"))
        if group_name is not None:
            configured_group = group_name
            break

    override_group = _clean_metadata_value(
        (component_metadata or {}).get("throughput_group"),
    )
    if configured_group is not None and override_group is not None:
        if configured_group != override_group:
            raise ValueError(
                f"{effect_name(parent)!r} configured throughput_group "
                f"{configured_group!r} conflicts with explicit override "
                f"{override_group!r}."
            )
    if configured_group is not None:
        return configured_group
    if override_group is not None:
        return override_group

    raise ValueError(
        f"{effect_name(parent)!r} selected {effect_name(effect)!r} without "
        "throughput_group metadata."
    )


def _surface_effect_phase_name(
    parent: Any,
    effect: Any,
    component_metadata: Mapping[str, Any] | None = None,
) -> str:
    configured_phase = None
    for meta in (getattr(effect, "meta", {}), getattr(parent, "meta", {})):
        phase_name = _clean_emission_phase(meta.get("emission_phase"))
        if phase_name is not None:
            configured_phase = phase_name
            break
    override_phase = _clean_emission_phase(
        (component_metadata or {}).get("emission_phase"),
    )
    if configured_phase is not None and override_phase is not None:
        if configured_phase != override_phase:
            raise ValueError(
                f"{effect_name(parent)!r} configured emission_phase "
                f"{configured_phase!r} conflicts with explicit override "
                f"{override_phase!r}."
            )
    if configured_phase is not None:
        return configured_phase
    if override_phase is not None:
        return override_phase
    raise ValueError(
        f"{effect_name(parent)!r} selected {effect_name(effect)!r} without "
        "emission_phase metadata."
    )


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
    """Return active per-aperture optical components before trace mapping.

    ScopeSim-refactor candidate: this is optical-train path introspection and
    should eventually be provided by the train/configuration layer.
    """
    excluded_ids = {id(qe_selector)} if qe_selector is not None else set()
    components: list[dict[str, Any]] = []
    for parent in active_effects(ztrain):
        if id(parent) in excluded_ids or _looks_like_qe_effect(parent):
            continue
        if not hasattr(parent, "wheel_effects"):
            if _is_surface_list_effect(parent):
                components.append({
                    "selector": parent,
                    "selector_name": effect_name(parent),
                    "effect": parent,
                })
            continue
        if getattr(parent, "meta", {}).get("selector_key") != "aperture_id":
            continue
        if aperture_id not in parent.wheel_effects:
            continue
        effect = resolve_effect(parent, aperture_id)
        if _looks_like_qe_effect(effect):
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
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten discovered optical components into ordered surface rows."""
    component_metadata = component_metadata or {}
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
                metadata = component_metadata.get(selector_name, {})
                if _real_colname("throughput_group", row.colnames) is None:
                    group_name = _surface_effect_group_name(
                        parent, effect, metadata,
                    )
                else:
                    group_name = surface_group_for_row(row)
                if _real_colname("emission_phase", row.colnames) is None:
                    phase_name = _surface_effect_phase_name(
                        parent, effect, metadata,
                    )
                else:
                    phase_name = emission_phase_for_row(row)
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
        metadata = component_metadata.get(selector_name, {})
        group_name = _surface_effect_group_name(parent, effect, metadata)
        phase_name = _surface_effect_phase_name(parent, effect, metadata)
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


def surface_group_for_row(row: Any) -> str:
    """Return explicit row throughput group."""
    group_col = _real_colname("throughput_group", row.colnames)
    name_col = _real_colname("name", row.colnames)
    surface_name = str(_row_scalar(row, name_col)) if name_col else repr(row)
    if group_col is None:
        raise ValueError(
            f"Surface row {surface_name!r} has no throughput_group column."
        )
    group_name = _clean_metadata_value(_row_scalar(row, group_col))
    if group_name is None:
        raise ValueError(
            f"Surface row {surface_name!r} has an empty throughput_group."
        )
    return group_name


def emission_phase_for_row(row: Any) -> str:
    """Return explicit row emission phase."""
    phase_col = _real_colname("emission_phase", row.colnames)
    name_col = _real_colname("name", row.colnames)
    surface_name = str(_row_scalar(row, name_col)) if name_col else repr(row)
    if phase_col is None:
        raise ValueError(
            f"Surface row {surface_name!r} has no emission_phase column."
        )
    phase_name = _clean_emission_phase(_row_scalar(row, phase_col))
    if phase_name is None:
        raise ValueError(
            f"Surface row {surface_name!r} has an empty emission_phase."
        )
    return phase_name


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


def _emission_density_plot_values(values: u.Quantity | None) -> np.ndarray | None:
    """Return thermal emission density in ScopeSim's PHOTLAM-like internal unit."""
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
    coords = _trace_detector_positions(
        trace, wave, image_plane, slit_positions_arcsec=np.array([0.0]),
    )
    if coords is None:
        return None
    return {
        key: np.asarray(value)[0]
        for key, value in coords.items()
    }


def _trace_slit_positions_arcsec(trace: Any) -> np.ndarray:
    """Return sampled slit coordinates carried by a trace table."""
    table = getattr(trace, "table", None)
    s_colname = getattr(trace, "meta", {}).get("s_colname", "s")
    if table is None or s_colname not in table.colnames:
        return np.array([0.0])
    values = table[s_colname]
    try:
        if getattr(values, "unit", None) is not None:
            values = u.Quantity(values).to_value(u.arcsec)
        else:
            values = np.asarray(values, dtype=float)
    except Exception:
        values = np.asarray(values, dtype=float)
    values = np.unique(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0])
    return values


def _trace_detector_positions(
    trace: Any,
    wave: u.Quantity,
    image_plane: Any | None = None,
    *,
    slit_positions_arcsec: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    """Return detector coordinates for trace slit positions and wavelength grid."""
    if not all(hasattr(trace, attr) for attr in ("xilam2x", "xilam2y")):
        return None

    wave_um = wave.to_value(u.um)
    if slit_positions_arcsec is None:
        slit_positions_arcsec = _trace_slit_positions_arcsec(trace)
    xi = np.asarray(slit_positions_arcsec, dtype=float)
    xi_grid = np.broadcast_to(xi[:, None], (xi.size, wave_um.size))
    wave_grid = np.broadcast_to(wave_um[None, :], (xi.size, wave_um.size))
    x_mm = np.asarray(trace.xilam2x(xi_grid, wave_grid), dtype=float)
    y_mm = np.asarray(trace.xilam2y(xi_grid, wave_grid), dtype=float)
    coords = {
        "detector_x_mm": x_mm,
        "detector_y_mm": y_mm,
        "slit_position_arcsec": xi,
    }

    if image_plane is None:
        return coords

    try:
        from astropy.wcs import WCS

        x_pix, y_pix = WCS(image_plane.header, key="D").all_world2pix(
            x_mm.ravel(), y_mm.ravel(), 0,
        )
        coords["detector_x"] = np.asarray(x_pix, dtype=float).reshape(x_mm.shape)
        coords["detector_y"] = np.asarray(y_pix, dtype=float).reshape(y_mm.shape)
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


def trace_resolution_diagnostic_table(
    ztrain: Any,
    *,
    trace_list_name: str = "trace_list_analytical",
    allow_diagnostic_psf: bool = False,
) -> Table:
    """Return the complete native-resolution grid from the configured train."""
    seeing = _required_cmd_quantity(ztrain.cmds, "!OBS.seeing", u.arcsec)
    zenith_angle = _zenith_angle_from_airmass(
        _required_cmd_float(ztrain.cmds, "!OBS.airmass"),
    )
    fwhm_func, _ = _configured_psf_fwhm_func(
        ztrain,
        allow_diagnostic_fallback=allow_diagnostic_psf,
    )
    trace_list = get_effect(ztrain, trace_list_name)
    spectrographs = get_effect(ztrain, "trace_eff_analytical")._spectrographs
    slit_selector = get_effect(ztrain, "slitwheel_selector")
    mapped = ztrain.trace_detector_coordinates(xi=0 * u.arcsec)
    mapped_trace_ids = np.asarray(mapped["trace_id"], dtype=str)
    fovs_by_trace: dict[str, list[Any]] = defaultdict(list)
    for fov in ztrain.fov_manager.fovs:
        fovs_by_trace[str(fov.trace_id)].append(fov)
    rows: list[dict[str, Any]] = []

    for aperture_id, traces in traces_by_aperture(trace_list).items():
        label = channel_label(aperture_id, traces)
        slit = slit_selector.get_effect(aperture_id).current_slit
        slit_y = u.Quantity(slit.data["y"]).to_value(u.arcsec)
        active_slit_width_arcsec = slit_y.max() - slit_y.min()

        for trace in traces:
            trace_id = str(trace.trace_id)
            trace_fovs = fovs_by_trace[trace_id]
            if len(trace_fovs) != 1:
                raise NotImplementedError(f"Trace {trace_id!r} requires one continuous configured FOV; found {len(trace_fovs)}.")

            mapped_indices = np.flatnonzero(mapped_trace_ids == trace_id)
            if mapped_indices.size < 2 or np.any(np.diff(mapped_indices) != 1):
                raise NotImplementedError(f"Trace {trace_id!r} does not have one continuous mapped run.")

            node_wave = mapped["wavelength"][mapped_indices].to(u.um)
            node_x_pix = mapped["detector_x"][mapped_indices].to_value(u.pixel)
            node_y_pix = mapped["detector_y"][mapped_indices].to_value(u.pixel)
            node_dx_pix = np.diff(node_x_pix)
            node_dy_pix = np.diff(node_y_pix)

            if (np.any(np.diff(node_wave) <= 0 * u.um)
                or np.any(node_dx_pix <= 0)
                or np.any(np.abs(node_dx_pix) <= np.abs(node_dy_pix))
                or not (np.all(np.isfinite(node_x_pix)) and
                        np.all(np.isfinite(node_y_pix)))):
                raise NotImplementedError(f"Trace {trace_id!r} does not have one continuous, "
                                          f"wavelength-ordered detector-x run.")

            image_plane_id = int(trace.meta["image_plane_id"])
            image_plane_header = ztrain.image_planes[image_plane_id].header
            pixel_scale_arcsec = np.sqrt(_image_plane_pixel_area(ztrain, image_plane_id).to_value(u.arcsec**2))
            spectral_fwhm_pix = trace.meta["nominal_fwhm_pix"] * active_slit_width_arcsec / trace.meta["nominal_slit_width"]

            readout_indices = np.unique(mapped["readout_index"][mapped_indices])
            if readout_indices.size != 1:
                raise NotImplementedError(f"Trace {trace_id!r} does not map to exactly one configured readout.")
            detector_manager = ztrain.detector_managers[int(readout_indices[0])]
            if len(detector_manager) != 1:
                raise NotImplementedError(f"Trace {trace_id!r} requires one configured detector per image plane.")
            detector_header = detector_manager[0].header
            detector_naxis1 = int(detector_header["NAXIS1"])
            detector_naxis2 = int(detector_header["NAXIS2"])

            prefix, _, order_text = trace_id.rpartition("_")
            order = int(order_text)
            spectrograph = spectrographs[prefix]
            order_index = np.flatnonzero(spectrograph.orders == order)
            if order_index.size != 1:
                raise ValueError(f"Trace {trace_id!r} does not identify one configured spectrograph order.")
            fsr_wave_low, fsr_wave_high = spectrograph.edge_wave(fsr=True)[order_index[0]].to(u.um)
            domain_wave_low = max(fsr_wave_low, node_wave[0])
            domain_wave_high = min(fsr_wave_high, node_wave[-1])
            if domain_wave_high <= domain_wave_low:
                raise ValueError(f"Trace {trace_id!r} does not intersect its configured primary FSR.")

            domain_coordinates = ztrain.trace_detector_coordinates(
                xi=0 * u.arcsec,
                wavelengths={trace_id: u.Quantity([domain_wave_low, domain_wave_high])},
            )
            domain_x_low, domain_x_high = domain_coordinates["detector_x"].to_value(u.pixel)
            domain_x_low = max(domain_x_low, -0.5)
            domain_x_high = min(domain_x_high, detector_naxis1 - 0.5)
            n_float_bins = int(np.floor((domain_x_high - domain_x_low) / spectral_fwhm_pix))
            float_boundaries = domain_x_low + np.arange(n_float_bins + 1) * spectral_fwhm_pix
            raster_boundaries = np.floor(float_boundaries + 1).astype(int)
            detector_x0_pix = raster_boundaries[:-1]
            detector_x1_pix = raster_boundaries[1:]
            raster_x_low = detector_x0_pix - 0.5
            raster_x_high = detector_x1_pix - 0.5
            complete = (
                (detector_x0_pix >= 0)
                & (detector_x1_pix <= detector_naxis1)
                & (detector_x1_pix > detector_x0_pix)
                & (raster_x_low >= domain_x_low)
                & (raster_x_high <= domain_x_high)
            )
            detector_x0_pix = detector_x0_pix[complete]
            detector_x1_pix = detector_x1_pix[complete]
            if detector_x0_pix.size < 2:
                raise ValueError(f"Trace {trace_id!r} has fewer than two complete native elements.")
            spectral_width_pix = detector_x1_pix - detector_x0_pix
            raster_x_low = detector_x0_pix - 0.5
            raster_x_high = detector_x1_pix - 0.5
            raster_x_center = 0.5 * (detector_x0_pix + detector_x1_pix) - 0.5

            node_wave_nm = node_wave.to_value(u.nm)
            wave_low_nm = np.interp(raster_x_low, node_x_pix, node_wave_nm)
            wave_high_nm = np.interp(raster_x_high, node_x_pix, node_wave_nm)
            requested_wave_nm = np.interp(raster_x_center, node_x_pix, node_wave_nm)
            center_coordinates = ztrain.trace_detector_coordinates(
                xi=0 * u.arcsec,
                wavelengths={trace_id: requested_wave_nm * u.nm},
            )
            center_trace_ids = np.asarray(center_coordinates["trace_id"], dtype=str)
            wave_nm = center_coordinates["wavelength"].to_value(u.nm)
            detector_x_pix = center_coordinates["detector_x"].to_value(u.pixel)
            detector_y_pix = center_coordinates["detector_y"].to_value(u.pixel)
            if (len(center_coordinates) != len(requested_wave_nm)
                or np.any(center_trace_ids != trace_id)
                or not (np.all(np.isfinite(detector_x_pix)) and np.all(np.isfinite(detector_y_pix)))
                or np.any(detector_x_pix < raster_x_low)
                or np.any(detector_x_pix > raster_x_high)):
                raise ValueError(f"Trace {trace_id!r} native centers did not remap through the configured train.")

            dispersion_nm_pix = (wave_high_nm - wave_low_nm) / spectral_width_pix
            if not np.all(np.isfinite(dispersion_nm_pix)) or np.any(dispersion_nm_pix <= 0):
                raise ValueError(f"Trace {trace_id!r} has non-positive or non-finite native dispersion.")

            resolving_power = wave_nm / (spectral_fwhm_pix * dispersion_nm_pix)
            psf_fwhm = fwhm_func(wave_nm * u.nm, zenith_angle, seeing).to_value(u.arcsec)
            spatial_fwhm_pix = psf_fwhm / pixel_scale_arcsec
            resel_footprint_pix = spectral_fwhm_pix * spatial_fwhm_pix

            for idx in range(wave_nm.size):
                rows.append({
                    "channel": label,
                    "aperture_id": int(aperture_id),
                    "image_plane_id": image_plane_id,
                    "trace_id": trace_id,
                    "sample_index": idx,
                    "wave_nm": wave_nm[idx],
                    "wave_low_nm": wave_low_nm[idx],
                    "wave_high_nm": wave_high_nm[idx],
                    "detector_x_pix": detector_x_pix[idx],
                    "detector_y_pix": detector_y_pix[idx],
                    "detector_x0_pix": detector_x0_pix[idx],
                    "detector_x1_pix": detector_x1_pix[idx],
                    "spectral_width_pix": spectral_width_pix[idx],
                    "detector_naxis1": detector_naxis1,
                    "detector_naxis2": detector_naxis2,
                    "pixel_scale_arcsec_pix": pixel_scale_arcsec,
                    "dispersion_nm_pix": dispersion_nm_pix[idx],
                    "resolving_power_R": resolving_power[idx],
                    "spectral_fwhm_pix": spectral_fwhm_pix,
                    "active_slit_width_arcsec": active_slit_width_arcsec,
                    "seeing_fwhm_arcsec": psf_fwhm[idx],
                    "spatial_fwhm_pix": spatial_fwhm_pix[idx],
                    "resel_footprint_pix": resel_footprint_pix[idx],
                })
    return Table(rows=sorted(rows, key=lambda row: row["image_plane_id"]))


def trace_resolution_summary_table(table: Table) -> Table:
    """Return compact per-channel trace-resolution diagnostics."""
    rows: list[dict[str, Any]] = []
    for key in table.group_by(["image_plane_id", "channel"]).groups.keys:
        channel = str(key["channel"])
        image_plane_id = int(key["image_plane_id"])
        group = table[
            (table["channel"] == channel)
            & (table["image_plane_id"] == image_plane_id)
        ]
        wave_nm = group["wave_nm"]
        dispersion_nm_pix = group["dispersion_nm_pix"]
        resolving_power = group["resolving_power_R"]
        spatial_fwhm_pix = group["spatial_fwhm_pix"]
        spectral_fwhm_pix = group["spectral_fwhm_pix"][0]
        resel_footprint_pix = spectral_fwhm_pix * spatial_fwhm_pix

        rows.append({
            "channel": channel,
            "image_plane_id": image_plane_id,
            "n_traces": len({str(value) for value in group["trace_id"]}),
            "wave_min_nm": np.min(wave_nm),
            "wave_max_nm": np.max(wave_nm),
            "dispersion_min_nm_pix": np.min(dispersion_nm_pix),
            "dispersion_median_nm_pix": np.median(dispersion_nm_pix),
            "dispersion_max_nm_pix": np.max(dispersion_nm_pix),
            "trace_R_min": np.min(resolving_power),
            "trace_R_median": np.median(resolving_power),
            "trace_R_max": np.max(resolving_power),
            "spectral_fwhm_pix": spectral_fwhm_pix,
            "resel_footprint_min_pix": np.min(resel_footprint_pix),
            "resel_footprint_median_pix": np.median(resel_footprint_pix),
            "resel_footprint_max_pix": np.max(resel_footprint_pix),
            "active_slit_width_arcsec": group["active_slit_width_arcsec"][0],
            "seeing_fwhm_median_arcsec": np.median(
                group["seeing_fwhm_arcsec"],
            ),
            "pixel_scale_arcsec_pix": group["pixel_scale_arcsec_pix"][0],
            "spatial_fwhm_min_pix": np.min(spatial_fwhm_pix),
            "spatial_fwhm_median_pix": np.median(spatial_fwhm_pix),
            "spatial_fwhm_max_pix": np.max(spatial_fwhm_pix),
        })
    return Table(rows=sorted(rows, key=lambda row: row["image_plane_id"]))


def _compact_range(
    minimum: Any,
    maximum: Any,
    *,
    precision: int,
    scale: float = 1.0,
) -> str:
    minimum = float(minimum) / scale
    maximum = float(maximum) / scale
    if np.isclose(minimum, maximum, rtol=1e-4, atol=10 ** -precision):
        return f"{minimum:.{precision}f}"
    return f"{minimum:.{precision}f}-{maximum:.{precision}f}"


def trace_resolution_summary_display_table(table: Table):
    """Return a compact, rounded trace-resolution summary for notebooks."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for row in table:
        rows.append({
            "ch": str(row["channel"]),
            "id": int(row["image_plane_id"]),
            "orders": int(row["n_traces"]),
            "wave nm": (
                f"{float(row['wave_min_nm']):.0f}-"
                f"{float(row['wave_max_nm']):.0f}"
            ),
            "slit": f"{float(row['active_slit_width_arcsec']):.2f}\"",
            "spec FWHM pix": f"{float(row['spectral_fwhm_pix']):.2f}",
            "disp nm/pix": _compact_range(
                row["dispersion_min_nm_pix"],
                row["dispersion_max_nm_pix"],
                precision=4,
            ),
            "R k": _compact_range(
                row["trace_R_min"],
                row["trace_R_max"],
                precision=1,
                scale=1000.0,
            ),
            "resel pix": _compact_range(
                row["resel_footprint_min_pix"],
                row["resel_footprint_max_pix"],
                precision=1,
            ),
            "seeing": (
                f"{float(row['seeing_fwhm_median_arcsec']):.2f}\"/"
                f"{float(row['spatial_fwhm_median_pix']):.1f} pix"
            ),
        })
    return pd.DataFrame(rows)


def _finite_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


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
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
    blocking_component_names: tuple[str, ...] = ("ir_blocking_filter_selector",),
    diffuse_extraction_pixels: float = 1.0,
) -> dict[str, Any]:
    """Build split pre/post-disperser emissivity sanity-check data.

    ``component_metadata`` supplies explicit metadata for selected single
    surface effects that ScopeSim/IRDB cannot currently tag. Configured
    metadata wins; conflicting overrides raise instead of silently replacing
    the instrument definition.
    """

    from scopesim.effects.illumination import integrate_spectral_background

    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)
    positional_qe_by_aperture = positional_qe_by_aperture or {}
    blocked_components = set(blocking_component_names)
    telescope_area = _telescope_area(ztrain)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    qe_selector = _get_qe_selector(ztrain, qe_selector_name, active_only=active_only)
    trace_list = get_effect(ztrain, "trace_list_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = dichroic_tree.table[dichroic_tree.table.colnames[0]].tolist()

    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    details: list[dict[str, Any]] = []
    for aperture_id in aperture_ids:
        detector_qe = resolve_effect(qe_selector, aperture_id)
        traces = traces_for_aperture.get(aperture_id, [])
        image_plane_id = int(traces[0].meta["image_plane_id"]) if traces else aperture_id

        image_pixel_area = _image_plane_pixel_area(ztrain, image_plane_id)

        qe_values = effective_diffuse_qe(detector_qe, wave, positional_qe=positional_qe_by_aperture.get(aperture_id))

        components = channel_optical_components(ztrain, aperture_id, qe_selector=qe_selector)

        surface_rows = optical_surface_rows(
            components, wave, component_metadata=component_metadata,
        )
        phase_terms, counts, surface_details = optical_surface_emissivity_terms(surface_rows, wave, qe_values=qe_values)

        unblocked_rows = _rows_excluding_components(surface_rows, blocked_components)

        unblocked_phase_terms, _unblocked_counts, _unblocked_details = (
            optical_surface_emissivity_terms(unblocked_rows, wave, qe_values=qe_values)
        )
        pre_total = _sum_terms(phase_terms["pre_disperser"], wave.size)
        post_total = _sum_terms(phase_terms["post_disperser"], wave.size)
        post_unblocked_total = _sum_terms(unblocked_phase_terms["post_disperser"], wave.size)
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

        diffuse_extract_equiv = np.full(wave.size, diffuse_rate_per_pix * diffuse_extraction_pixels, dtype=float)
        diffuse_unblocked_extract_equiv = np.full(wave.size, diffuse_unblocked_rate_per_pix * diffuse_extraction_pixels,
                                                  dtype=float)

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
    value = from_currsys("!TEL.area", ztrain.cmds)
    quantity = u.Quantity(value)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity.value * u.m**2
    return quantity.to(u.m**2)


def build_post_disperser_diffuse_background_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
    blocking_component_names: tuple[str, ...] = ("ir_blocking_filter_selector",),
) -> dict[str, Any]:
    """Build image-plane post-disperser diffuse background data.

    ``component_metadata`` supplies explicit metadata for selected single
    surface effects that ScopeSim/IRDB cannot currently tag. Configured
    metadata wins; conflicting overrides raise instead of silently replacing
    the instrument definition.
    """
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
    post_diffuse_selector = _optional_effect(
        ztrain,
        "post_echelle_diffuse_background_selector",
        active_only=active_only,
    )
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
        effect_data = _post_disperser_diffuse_data_from_effect(
            post_diffuse_selector,
            image_plane_id,
            image_plane,
            wave,
            component_metadata=component_metadata,
        )
        if effect_data is None:
            qe_values = effective_diffuse_qe(
                detector_qe,
                wave,
                positional_qe=positional_qe_by_aperture.get(aperture_id),
                footprint=image_plane,
            )
            components = channel_optical_components(
                ztrain, aperture_id, qe_selector=qe_selector,
            )
            surface_rows = optical_surface_rows(
                components, wave, component_metadata=component_metadata,
            )
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
            detector_qe_accounting = detector_qe_accounting_summary(detector_qe)
        else:
            spectra = effect_data["spectra"]
            spectra_unblocked = effect_data["spectra_without_blocking"]
            surface_details = effect_data["details"]
            qe_values = effect_data["detector_qe"]
            detector_qe_accounting = effect_data["detector_qe_accounting"]
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
            "detector_qe_accounting": detector_qe_accounting,
            "pixel_area": image_pixel_area,
            "telescope_area": telescope_area,
            "detector_qe_note": "throughput only; detector emissivity is not modeled",
        }

    return {"wave_nm": wave_nm, "channels": channels, "details": Table(rows=details)}


def _optional_effect(
    ztrain: Any,
    display_name: str,
    *,
    active_only: bool = True,
) -> Any | None:
    effects = (
        active_effects(ztrain)
        if active_only else ztrain.optics_manager.all_effects
    )
    matches = [eff for eff in effects if effect_name(eff) == display_name]
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one active effect named {display_name!r}; found "
            f"{len(matches)}."
        )
    return matches[0] if matches else None


def _post_disperser_diffuse_data_from_effect(
    selector: Any | None,
    image_plane_id: int,
    image_plane: Any,
    wave: u.Quantity,
    *,
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if selector is None or not hasattr(selector, "wheel_effects"):
        return None
    if image_plane_id not in selector.wheel_effects:
        return None

    effect = resolve_effect(selector, image_plane_id)
    if not all(
        hasattr(effect, attr)
        for attr in ("_surface_list", "_detector_qe", "_downstream_throughput_values")
    ):
        return None

    qe_values = effective_diffuse_qe(
        effect._detector_qe,
        wave,
        positional_qe=getattr(effect, "_positional_qe", None),
        footprint=image_plane,
    )
    surface_rows = optical_surface_rows(
        [{
            "selector": selector,
            "selector_name": effect_name(selector),
            "effect": effect._surface_list,
        }],
        wave,
        component_metadata=component_metadata,
    )
    spectra_unblocked, surface_details = optical_surface_post_disperser_diffuse_terms(
        surface_rows,
        wave,
        qe_values=qe_values,
        emission_phase=effect.meta.get("emission_phase", "post_disperser"),
    )
    downstream = effect._downstream_throughput_values(wave)
    spectra = OrderedDict(
        (name, spectrum * downstream)
        for name, spectrum in spectra_unblocked.items()
    )
    for row in surface_details:
        row["downstream_throughput_applied"] = bool(np.any(downstream != 1.0))
    return {
        "spectra": spectra,
        "spectra_without_blocking": spectra_unblocked,
        "details": surface_details,
        "detector_qe": qe_values,
        "detector_qe_accounting": detector_qe_accounting_summary(
            effect._detector_qe,
        ),
    }


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
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
) -> Table:
    """Compare notebook helper rates to the configured image-plane effect.

    ``helper_data`` is normally the output of
    :func:`build_post_disperser_diffuse_background_data`. The ``matched`` rate
    is recomputed on the effect's own wavelength grid when possible, so the
    table separates wavelength-sampling differences from real wiring problems.
    ``component_metadata`` must match the metadata passed to
    :func:`build_post_disperser_diffuse_background_data`.
    """
    selector = get_effect(ztrain, effect_display_name, active_only=False)
    matched_data_by_image_plane = {}
    if match_effect_grid:
        matched_data_by_image_plane = _matched_post_diffuse_data_by_image_plane(
            ztrain,
            helper_data,
            selector,
            qe_selector_name=qe_selector_name,
            component_metadata=component_metadata,
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
        if match_effect_grid and matched_channel is None:
            raise ValueError(
                f"No matched helper data for image_plane_id={image_plane_id}, "
                f"aperture_id={aperture_id}."
            )
        matched_rate = float(matched_channel["total_rate_ph_s_pix"]) if (
            matched_channel is not None
        ) else np.nan
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
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    active_only: bool = True,
) -> dict[int, Mapping[str, Any]]:
    matched = {}
    cache = {}
    for channel in helper_data["channels"].values():
        image_plane_id = int(channel["image_plane_id"])
        effect = resolve_effect(selector, image_plane_id)
        if not hasattr(effect, "_waveset"):
            raise ValueError(
                f"{effect_name(effect)!r} for image_plane_id={image_plane_id} "
                "has no _waveset method for matched-grid validation."
            )
        wave_nm = effect._waveset().to(u.nm)
        cache_key = tuple(np.round(wave_nm.to_value(u.nm), 12))
        if cache_key not in cache:
            cache[cache_key] = build_post_disperser_diffuse_background_data(
                ztrain,
                wave_nm=wave_nm,
                component_metadata=component_metadata,
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
    in_slit = (np.abs(x) <= 0.5 * length) & (np.abs(y) <= 0.5 * width)

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


def _slit_selector_widths_for_key(
    ztrain: Any,
    slit_key: str,
    *,
    selector_name: str = "slitwheel_selector",
) -> u.Quantity:
    if not hasattr(ztrain, "optics_manager"):
        return np.array([]) * u.arcsec
    try:
        selector = get_effect(ztrain, selector_name)
    except ValueError:
        return np.array([]) * u.arcsec

    values: list[float] = []
    for effect in selector.wheel_effects.values():
        meta = getattr(effect, "meta", {}) or {}
        if meta.get("current_slit") != slit_key:
            continue
        for name in meta.get("slit_names", ()):
            try:
                values.append(float(name))
            except (TypeError, ValueError):
                continue
    return np.array(sorted(set(values)), dtype=float) * u.arcsec


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


def _required_cmd_value(cmds: Any, key: str) -> Any:
    from scopesim.utils import from_currsys

    try:
        value = from_currsys(key, cmds)
    except Exception as exc:
        raise ValueError(f"Required ScopeSim command {key!r} is not set.") from exc
    if isinstance(value, str) and value == key:
        raise ValueError(f"Required ScopeSim command {key!r} is unresolved.")
    return value


def _required_cmd_float(cmds: Any, key: str) -> float:
    value = _required_cmd_value(cmds, key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Required ScopeSim command {key!r} must resolve to a float; "
            f"got {value!r}."
        ) from exc


def _required_cmd_quantity(cmds: Any, key: str, unit: u.UnitBase) -> u.Quantity:
    value = _required_cmd_value(cmds, key)
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


def _format_arcsec_value(value: u.Quantity) -> str:
    return f"{u.Quantity(value).to_value(u.arcsec):.3g}"


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
    return _zenith_angle_from_airmass(_required_cmd_float(cmds, "!OBS.airmass"))


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


def _diagnostic_fallback_psf_fwhm(
    wave: u.Quantity,
    zenith_angle: u.Quantity,
    seeing: u.Quantity,
) -> u.Quantity:
    return _natural_seeing_fwhm(wave, seeing, zenith_angle)


def _active_moffat_psf(train_or_cmds: Any) -> Any | None:
    if not hasattr(train_or_cmds, "optics_manager"):
        return None
    for effect in active_effects(train_or_cmds):
        if effect.__class__.__name__ in {"AOEnhanceablePSF", "MoffatPSF"}:
            if callable(getattr(effect, "fwhm", None)):
                return effect
    return None


def _scopesim_psf_fwhm_func(effect: Any) -> Callable[
    [u.Quantity, u.Quantity, u.Quantity], u.Quantity
]:
    def fwhm(
        wave: u.Quantity,
        _zenith_angle: u.Quantity,
        _seeing: u.Quantity,
    ) -> u.Quantity:
        return _quantity_with_default_unit(
            effect.fwhm(u.Quantity(wave).to(u.um)),
            u.arcsec,
        )

    return fwhm


def _configured_psf_fwhm_func(
    train_or_cmds: Any,
    *,
    allow_diagnostic_fallback: bool = False,
) -> tuple[Callable[[u.Quantity, u.Quantity, u.Quantity], u.Quantity], Any | None]:
    effect = _active_moffat_psf(train_or_cmds)
    if effect is None:
        if not allow_diagnostic_fallback:
            raise ValueError(
                "No active ScopeSim Moffat-like PSF effect was found. Pass "
                "allow_diagnostic_psf=True only for an explicit diagnostic "
                "seeing-law calculation."
            )
        return _diagnostic_fallback_psf_fwhm, None
    return _scopesim_psf_fwhm_func(effect), effect


def _atmospheric_refraction_shift(
    wave: u.Quantity,
    zenith_angle: u.Quantity,
    cmds: Any,
    *,
    wave_ref: u.Quantity = 500 * u.nm,
) -> u.Quantity:
    from scopesim.effects.atmo_dispersion import refractive_index

    temperature = _required_cmd_quantity(cmds, "!ATMO.temperature", u.deg_C)
    pressure = _required_cmd_quantity(cmds, "!ATMO.pressure", u.bar)
    humidity = _required_cmd_float(cmds, "!ATMO.humidity")
    x_co2 = _required_cmd_float(cmds, "!ATMO.x_co2")
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
    fwhm_func: Callable[[u.Quantity, u.Quantity, u.Quantity], u.Quantity],
) -> np.ndarray:
    x = u.Quantity(positions["x"]).to_value(u.arcsec)
    y = u.Quantity(positions["y"]).to_value(u.arcsec)
    weights = (
        np.asarray(positions["weight"], dtype=float)
        if "weight" in positions.colnames
        else np.ones(len(positions), dtype=float)
    )
    fwhm_values = fwhm_func(wave, zenith_angle, seeing).to_value(u.arcsec)
    shift_values = u.Quantity(shifts).to_value(u.arcsec)

    image = np.zeros_like(x_grid, dtype=float)
    for shift, fwhm in zip(shift_values, fwhm_values, strict=True):
        for xpos, ypos, weight in zip(x, y, weights, strict=True):
            image += weight * _moffat_image(
                x_grid,
                y_grid,
                x0=xpos,
                y0=ypos + shift,
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
    x_extent = max(length + 8 * max_fwhm, 1.2 * length)
    y_extent = max(4.0, width + 8 * max_fwhm + 2 * max_shift)
    x = np.arange(-0.5 * x_extent, 0.5 * x_extent + step, step)
    y = np.arange(-0.5 * y_extent, 0.5 * y_extent + step, step)
    x_grid, y_grid = np.meshgrid(x, y)
    slit_mask = (
        (np.abs(x_grid) <= 0.5 * length)
        & (np.abs(y_grid) <= 0.5 * width)
    )
    pixel_area = step**2

    throughput = np.empty(wave.size, dtype=float)
    for idx, (shift, fwhm) in enumerate(zip(shifts_arcsec, fwhm_values, strict=True)):
        image = _moffat_image(
            x_grid, y_grid, x0=0.0, y0=float(shift),
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
    *,
    seeing: u.Quantity,
    allow_diagnostic_fallback: bool = False,
) -> OrderedDict[str, dict[str, Any]]:
    base_fwhm_func, psf_effect = _configured_psf_fwhm_func(
        train_or_cmds,
        allow_diagnostic_fallback=allow_diagnostic_fallback,
    )
    base_beta = float(getattr(psf_effect, "alpha", beta) or beta)
    if psf_effect is None:
        base_note = (
            "Fallback diagnostic seeing law; no active ScopeSim Moffat-like "
            "PSF effect was available."
        )
    else:
        base_note = (
            f"FWHM from active ScopeSim {effect_name(psf_effect)} "
            f"({psf_effect.__class__.__name__})."
        )

    modes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    modes["no_ao"] = {
        "label": f'{_format_arcsec_value(seeing)}" NS',
        "style": "-",
        "beta": base_beta,
        "fwhm_func": base_fwhm_func,
        "current": True,
        "note": base_note,
    }

    effect = _active_ao_enhanceable_psf(train_or_cmds)
    if effect is None or not hasattr(effect, "ao_scale"):
        return modes

    ao_enabled = _metadata_bool(effect.meta.get("enable_ao"), default=False)
    modes["no_ao"]["current"] = not ao_enabled
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
        return (base_fwhm_func(wave, zenith_angle, seeing) * scale).to(u.arcsec)

    modes["ao"] = {
        "label": "AO",
        "style": "--",
        "beta": float(getattr(effect, "alpha", beta) or beta),
        "fwhm_func": ao_fwhm,
        "current": ao_enabled,
        "note": (
            "AO design FWHM from active AOEnhanceablePSF. "
            "Dimensionless absolute AO tables are interpreted as arcsec, "
            "matching ScopeSim's PSF quantification convention. "
            "This diagnostic curve does not change the optical train."
        ),
    }
    return modes


def _unique_float_values(values: u.Quantity, *, atol: float = 1.0e-9) -> list[float]:
    finite = sorted(
        float(value)
        for value in u.Quantity(values).to_value(u.arcsec)
        if np.isfinite(value)
    )
    unique: list[float] = []
    for value in finite:
        if not unique or not np.isclose(value, unique[-1], rtol=0.0, atol=atol):
            unique.append(value)
    return unique


def _slit_loss_slit_specs(
    selector_widths: u.Quantity,
    current_slit: u.Quantity,
) -> OrderedDict[str, dict[str, Any]]:
    current = float(u.Quantity(current_slit).to_value(u.arcsec))
    supported = _unique_float_values(selector_widths)
    values = list(supported)
    if not any(
        np.isclose(value, current, rtol=0.0, atol=1.0e-6)
        for value in values
    ):
        values.append(current)
        values = sorted(values)

    if len(values) > 2:
        selected = [values[0], values[-1]]
        if not any(
            np.isclose(value, current, rtol=0.0, atol=1.0e-6)
            for value in selected
        ):
            selected.insert(1, current)
    else:
        selected = values

    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for value in sorted(selected):
        current_match = np.isclose(value, current, rtol=0.0, atol=1.0e-6)
        if len(supported) >= 2 and np.isclose(
            value, supported[0], rtol=0.0, atol=1.0e-6,
        ):
            role = "narrowest"
        elif len(supported) >= 2 and np.isclose(
            value, supported[-1], rtol=0.0, atol=1.0e-6,
        ):
            role = "widest"
        else:
            role = "current"
        key = (
            "current"
            if current_match
            else f"{value:.3f}".rstrip("0").rstrip(".")
        )
        if key in specs:
            continue
        specs[key] = {
            "label": f'{value:.2f}"',
            "slit_width": value * u.arcsec,
            "role": role,
            "current": bool(current_match),
        }
    return specs


def _airmass_label(airmass: float) -> str:
    return f"X={airmass:.2f}".rstrip("0").rstrip(".")


def _slit_loss_airmass_specs(
    current_airmass: float,
) -> OrderedDict[str, dict[str, Any]]:
    current_airmass = max(1.0, float(current_airmass))
    base = [
        ("zenith", 1.0, "X=1"),
        ("current", current_airmass, _airmass_label(current_airmass)),
        ("z60", 2.0, "z=60 deg (X=2)"),
    ]
    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for name, airmass, label in base:
        zenith_angle = _zenith_angle_from_airmass(airmass)
        match = None
        for key, spec in specs.items():
            if np.isclose(spec["airmass"], airmass, rtol=0.0, atol=1.0e-6):
                match = key
                break
        if match is not None:
            specs[match]["current"] |= name == "current"
            if name == "current":
                specs[match]["name"] = "current"
                specs[match]["label"] = label
            continue
        specs[name] = {
            "name": name,
            "label": label,
            "airmass": airmass,
            "zenith_angle": zenith_angle,
            "current": name == "current",
        }
    return specs


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
    allow_diagnostic_psf: bool = False,
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
    seeing = _required_cmd_quantity(cmds, "!OBS.seeing", u.arcsec)
    zenith_angle = _zenith_angle_from_cmds(cmds)
    slit_width = u.Quantity(slit_width).to(u.arcsec)
    slit_length = u.Quantity(slit_length).to(u.arcsec)
    fwhm_func, psf_effect = _configured_psf_fwhm_func(
        train_or_cmds,
        allow_diagnostic_fallback=allow_diagnostic_psf,
    )
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
        fwhm_func(wave, zenith_angle, seeing).to_value(u.arcsec),
    ))
    width = slit_width.to_value(u.arcsec)
    length = slit_length.to_value(u.arcsec)
    x_extent = max(
        1.12 * length,
        float(np.ptp(all_x)) + 6 * max_fwhm,
    )
    y_extent = max(
        3.5,
        width + 2 * max_shift + 6 * max_fwhm,
        float(np.ptp(all_y)) + 2 * max_shift + 4 * max_fwhm,
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
                    fwhm_func=fwhm_func,
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
        "airmass": _required_cmd_float(cmds, "!OBS.airmass"),
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
            (
                f"PSF FWHM comes from active ScopeSim {effect_name(psf_effect)}."
                if psf_effect is not None
                else (
                    "PSF FWHM uses a fallback diagnostic seeing law because no "
                    "active ScopeSim Moffat-like PSF effect was available."
                )
            ),
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
    allow_diagnostic_psf: bool = False,
) -> dict[str, Any]:
    """Return centered point-source slit-loss curves by arm."""
    cmds = _cmds_from_train_or_cmds(train_or_cmds)
    arms = arms or OrderedDict({
        "VIS": (310 * u.nm, 980 * u.nm, "!INST.vis_curr_slit"),
        "NIR": (980 * u.nm, 2450 * u.nm, "!INST.nir_curr_slit"),
    })
    seeing = _required_cmd_quantity(cmds, "!OBS.seeing", u.arcsec)
    airmass = max(1.0, _required_cmd_float(cmds, "!OBS.airmass"))
    current_zenith_angle = _zenith_angle_from_airmass(airmass)
    psf_modes = _slit_loss_psf_modes(
        train_or_cmds,
        beta,
        seeing=seeing,
        allow_diagnostic_fallback=allow_diagnostic_psf,
    )
    adc_error, adc_note = _adc_zenith_angle_error(
        train_or_cmds, adc_zenith_angle_error,
    )
    airmass_specs = _slit_loss_airmass_specs(airmass)
    adc_specs = OrderedDict({
        "ad_only": {
            "label": "ADC off",
            "mode": "ad",
            "current": False,
            "note": "Full atmospheric dispersion before slit clipping.",
        },
        "adc_residual": {
            "label": "ADC on",
            "mode": "adc",
            "current": True,
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
        slit_width = _required_cmd_quantity(cmds, slit_key, u.arcsec)
        selector_slits = _slit_selector_widths_for_key(train_or_cmds, slit_key)
        slit_specs = _slit_loss_slit_specs(selector_slits, slit_width)
        curves: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for psf_name, psf_spec in psf_modes.items():
            for slit_name, slit_spec in slit_specs.items():
                for airmass_name, airmass_spec in airmass_specs.items():
                    zenith_angle = airmass_spec["zenith_angle"]
                    ad_shift = _atmospheric_refraction_shift(
                        wave, zenith_angle, cmds, wave_ref=wave_ref,
                    )
                    for adc_name, adc_spec in adc_specs.items():
                        if adc_spec["mode"] == "adc":
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
                            slit_width=slit_spec["slit_width"],
                            slit_length=slit_length,
                            beta=psf_spec["beta"],
                            grid_step=grid_step,
                            fwhm_func=psf_spec["fwhm_func"],
                        )
                        key = (
                            f"{psf_name}_slit_{slit_name}_airmass_"
                            f"{airmass_name}_{adc_name}"
                        )
                        curves[key] = {
                            "label": (
                                f"{psf_spec['label']}, "
                                f"{slit_spec['label']} slit, "
                                f"{airmass_spec['label']}, {adc_spec['label']}"
                            ),
                            "throughput": throughput,
                            "loss": 1.0 - throughput,
                            "shift_arcsec": shifts,
                            "linestyle": psf_spec["style"],
                            "psf_mode": psf_name,
                            "psf_label": psf_spec["label"],
                            "current_psf": bool(psf_spec.get("current", False)),
                            "psf_note": psf_spec["note"],
                            "slit_name": slit_name,
                            "slit_label": slit_spec["label"],
                            "slit_role": slit_spec["role"],
                            "slit_width_arcsec": slit_spec["slit_width"],
                            "current_slit": bool(slit_spec["current"]),
                            "airmass_name": airmass_spec["name"],
                            "airmass_label": airmass_spec["label"],
                            "airmass": float(airmass_spec["airmass"]),
                            "zenith_angle_deg": zenith_angle.to(u.deg),
                            "current_airmass": bool(airmass_spec["current"]),
                            "adc_state": adc_name,
                            "adc_label": adc_spec["label"],
                            "current_adc": bool(adc_spec["current"]),
                            "note": adc_spec["note"],
                        }

        alias_airmass_keys = {
            "current": next(
                key for key, spec in airmass_specs.items() if spec["current"]
            ),
            "zenith": next(
                key for key, spec in airmass_specs.items()
                if np.isclose(spec["airmass"], 1.0, rtol=0.0, atol=1.0e-6)
            ),
            "z60": next(
                key for key, spec in airmass_specs.items()
                if np.isclose(spec["airmass"], 2.0, rtol=0.0, atol=1.0e-6)
            ),
        }
        for psf_name in psf_modes:
            for old_name, slit_name, airmass_name, adc_name in (
                ("current_adc_residual", "current", "current", "adc_residual"),
                ("zenith", "current", "zenith", "ad_only"),
                ("elevation_60_ad_only", "current", "z60", "ad_only"),
                ("elevation_60_adc_residual", "current", "z60", "adc_residual"),
            ):
                airmass_key = alias_airmass_keys[airmass_name]
                target = (
                    f"{psf_name}_slit_{slit_name}_airmass_"
                    f"{airmass_key}_{adc_name}"
                )
                if target in curves:
                    curves[f"{psf_name}_{old_name}"] = {
                        **curves[target],
                        "alias_for": target,
                    }
        arm_data[arm_name] = {
            "wave_nm": wave,
            "slit_width_arcsec": slit_width,
            "slit_length_arcsec": u.Quantity(slit_length).to(u.arcsec),
            "selector_slit_widths_arcsec": selector_slits,
            "slit_variants": slit_specs,
            "curves": curves,
        }

    return {
        "arms": arm_data,
        "seeing_arcsec": seeing,
        "airmass": airmass,
        "current_zenith_angle_deg": current_zenith_angle.to(u.deg),
        "active_psf_mode": next(
            name for name, spec in psf_modes.items() if spec["current"]
        ),
        "psf_modes": psf_modes,
        "airmass_modes": airmass_specs,
        "adc_modes": adc_specs,
        "wave_ref_nm": u.Quantity(wave_ref).to(u.nm),
        "adc_zenith_angle_error_deg": adc_error,
        "notes": [
            "Centered point-source loss from a pre-slit Moffat PSF.",
            "This diagnostic exposes slit loss that ScopeSim does not measure directly.",
        ],
    }


def build_slit_width_loss_data(
    train_or_cmds: Any,
    *,
    slit_widths: u.Quantity | None = None,
    arms: Mapping[str, tuple[u.Quantity, str]] | None = None,
    slit_length: u.Quantity = 10.0 * u.arcsec,
    grid_step: u.Quantity = 0.04 * u.arcsec,
    beta: float = 4.765,
    allow_diagnostic_psf: bool = False,
) -> dict[str, Any]:
    """Return centered point-source slit loss over a sweep of slit widths.

    This isolates PSF/slit coupling: the source is centered and no atmospheric
    dispersion shift is applied. When an optical train is supplied, FWHM values
    come from the active ScopeSim Moffat-like PSF effect. The default VIS/NIR
    wavelength arrays are matched by index; ``plot_slit_width_loss`` depends on
    that contract and raises if the paired arms diverge.
    """
    cmds = _cmds_from_train_or_cmds(train_or_cmds)
    slit_widths = (
        np.linspace(0.25, 2.0, 20) * u.arcsec
        if slit_widths is None
        else _quantity_with_default_unit(slit_widths, u.arcsec)
    ).to(u.arcsec)
    arms = arms or OrderedDict({
        "VIS": (
            np.array([350, 500, 750, 950]) * u.nm,
            "!INST.vis_curr_slit",
        ),
        "NIR": (
            np.array([1000, 1250, 1650, 2200]) * u.nm,
            "!INST.nir_curr_slit",
        ),
    })
    seeing = _required_cmd_quantity(cmds, "!OBS.seeing", u.arcsec)
    zenith_angle = _zenith_angle_from_cmds(cmds)
    psf_modes = _slit_loss_psf_modes(
        train_or_cmds,
        beta,
        seeing=seeing,
        allow_diagnostic_fallback=allow_diagnostic_psf,
    )

    arm_data: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for arm_name, (wave_values, slit_key) in arms.items():
        wave = _quantity_with_default_unit(wave_values, u.nm).to(u.nm)
        current_slit = _required_cmd_quantity(cmds, slit_key, u.arcsec)
        curves: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for psf_name, psf_spec in psf_modes.items():
            for wave_value in wave:
                losses = []
                one_wave = np.array([wave_value.to_value(u.nm)]) * u.nm
                zero_shift = np.zeros(1) * u.arcsec
                for slit_width in slit_widths:
                    throughput = _slit_throughput_curve(
                        one_wave,
                        zero_shift,
                        seeing=seeing,
                        zenith_angle=zenith_angle,
                        slit_width=slit_width,
                        slit_length=slit_length,
                        beta=psf_spec["beta"],
                        grid_step=grid_step,
                        fwhm_func=psf_spec["fwhm_func"],
                    )
                    losses.append(1.0 - float(throughput[0]))
                wave_nm = wave_value.to_value(u.nm)
                curves[f"{psf_name}_{wave_nm:.0f}nm"] = {
                    "label": f"{psf_spec['label']}, {wave_nm:.0f} nm",
                    "loss": np.asarray(losses, dtype=float),
                    "throughput": 1.0 - np.asarray(losses, dtype=float),
                    "linestyle": psf_spec["style"],
                    "psf_mode": psf_name,
                    "psf_note": psf_spec["note"],
                    "wavelength_nm": float(wave_nm),
                }
        arm_data[arm_name] = {
            "slit_widths_arcsec": slit_widths,
            "current_slit_width_arcsec": current_slit,
            "selector_slit_widths_arcsec": _slit_selector_widths_for_key(
                train_or_cmds, slit_key,
            ),
            "wavelengths_nm": wave,
            "slit_length_arcsec": u.Quantity(slit_length).to(u.arcsec),
            "curves": curves,
        }

    return {
        "arms": arm_data,
        "seeing_arcsec": seeing,
        "psf_modes": psf_modes,
        "notes": [
            "Centered point-source PSF loss with no atmospheric-dispersion shift.",
            "Use this to inspect narrow-slit throughput sensitivity by arm.",
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
            dark_current, "value", ztrain.cmds, detector_id,
        )
        read_noise_single = _resolved_detector_meta(
            read_noise, "noise_std", ztrain.cmds, detector_id,
        )
        read_ndit = _resolved_detector_meta(
            read_noise, "ndit", ztrain.cmds, detector_id,
        )
        bias_level = _resolved_detector_meta(
            bias, "bias", ztrain.cmds, detector_id,
        )

        diffuse_counts = diffuse_rate * exposure_time
        dark_counts = dark_rate * exposure_time
        additive_signal = diffuse_counts + dark_counts
        full_well = _required_detector_cmd_float(
            ztrain.cmds, "!DET.full_well", detector_id,
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


def science_truth_crosscheck_table(
    ztrain: Any,
    *,
    post_diffuse_data: Mapping[str, Any] | None = None,
    detector_budget: Table | None = None,
    extraction_pixels: float = 1.0,
    spectral_resolution: float | None = None,
) -> Table:
    """Return compact physical sanity anchors for notebook inspection.

    This table intentionally reuses already-built validation products. It does
    not add a second simulation path; it presents standard-reference quantities
    that make detector/background scales easier to audit by eye.
    """
    if detector_budget is None:
        detector_budget = detector_background_budget_table(
            ztrain, post_diffuse_data=post_diffuse_data,
        )
    if spectral_resolution is None:
        spectral_resolution = _required_cmd_float(
            ztrain.cmds, "!SIM.spectral.spectral_resolution",
        )

    post_channels = _post_diffuse_channels_by_image_plane(post_diffuse_data)
    rows: list[dict[str, Any]] = []
    for budget in detector_budget:
        image_plane_id = int(budget["image_plane_id"])
        channel_data = post_channels.get(image_plane_id, {})
        pixel_area = u.Quantity(
            channel_data.get("pixel_area", np.nan * u.arcsec**2),
        ).to(u.arcsec**2)
        pixel_area_value = pixel_area.to_value(u.arcsec**2)
        diffuse_rate = float(budget["post_diffuse_rate_ph_s_pix"])
        exposure_time = float(budget["exposure_time_s"])
        wavelength_mid = _channel_mid_wavelength_nm(channel_data)
        resolution_element = (
            wavelength_mid / spectral_resolution
            if np.isfinite(wavelength_mid) and np.isfinite(spectral_resolution)
            and spectral_resolution > 0
            else np.nan
        )
        diffuse_rate_arcsec2 = (
            diffuse_rate / pixel_area_value
            if np.isfinite(pixel_area_value) and pixel_area_value > 0
            else np.nan
        )
        extraction_rate = diffuse_rate * float(extraction_pixels)
        extraction_counts = extraction_rate * exposure_time
        status = str(budget["saturation_status"])
        rows.append({
            "detector_id": int(budget["detector_id"]),
            "image_plane_id": image_plane_id,
            "channel": str(budget["channel"]),
            "detector": str(budget["detector"]),
            "current_slit_arcsec": _current_slit_arcsec(
                ztrain, str(budget["channel"]),
            ),
            "wavelength_mid_nm": wavelength_mid,
            "spectral_resolution_R": float(spectral_resolution),
            "resolution_element_nm": resolution_element,
            "pixel_area_arcsec2": pixel_area_value,
            "post_diffuse_ph_s_pix": diffuse_rate,
            "post_diffuse_ph_s_arcsec2": diffuse_rate_arcsec2,
            "extraction_pixels": float(extraction_pixels),
            "post_diffuse_ph_s_extraction": extraction_rate,
            "post_diffuse_e_extraction": extraction_counts,
            "additive_signal_e_pix": float(budget["additive_signal_e_pix"]),
            "full_well_e": float(budget["full_well_e"]),
            "signal_fraction_of_full_well": float(
                budget["signal_fraction_of_full_well"],
            ),
            "saturation_status": status,
            "note": _science_truth_note(status),
        })
    return Table(rows=rows)


def _zenith_angle_from_airmass(airmass: float) -> u.Quantity:
    airmass = max(1.0, float(airmass))
    return (
        np.degrees(np.arccos(np.clip(1.0 / airmass, 0.0, 1.0))) * u.deg
    )


def _trace_dispersion_nm_per_pixel(
    trace: Any,
    image_plane: Any | None,
    wave_mid_nm: float,
) -> float:
    trace_min_nm = float(trace.wave_min) * 1000.0
    trace_max_nm = float(trace.wave_max) * 1000.0
    span_nm = trace_max_nm - trace_min_nm
    if not np.isfinite(span_nm) or span_nm <= 0:
        return np.nan
    step_nm = min(
        max(abs(wave_mid_nm) * 1.0e-5, span_nm * 1.0e-4, 1.0e-5),
        0.45 * span_nm,
    )
    wave_pair = np.array([
        max(trace_min_nm, wave_mid_nm - step_nm),
        min(trace_max_nm, wave_mid_nm + step_nm),
    ]) * u.nm
    if wave_pair[1] <= wave_pair[0]:
        return np.nan
    coords = _trace_center_detector_positions(trace, wave_pair, image_plane)
    if coords is None:
        return np.nan
    if "detector_x" in coords and "detector_y" in coords:
        dx = float(coords["detector_x"][1] - coords["detector_x"][0])
        dy = float(coords["detector_y"][1] - coords["detector_y"][0])
        pixel_distance = np.hypot(dx, dy)
    else:
        pixel_size = getattr(trace, "meta", {}).get("pixel_size")
        try:
            pixel_size = float(pixel_size)
        except (TypeError, ValueError):
            pixel_size = np.nan
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            return np.nan
        dx = float(coords["detector_x_mm"][1] - coords["detector_x_mm"][0])
        dy = float(coords["detector_y_mm"][1] - coords["detector_y_mm"][0])
        pixel_distance = np.hypot(dx, dy) / pixel_size
    if not np.isfinite(pixel_distance) or pixel_distance <= 0:
        return np.nan
    return float((wave_pair[1] - wave_pair[0]).to_value(u.nm) / pixel_distance)


def resolution_element_footprint_table(
    ztrain: Any,
    *,
    spectral_resolution: float | None = None,
    allow_diagnostic_psf: bool = False,
) -> Table:
    """Return a per-channel FWHM resolution-element footprint estimate.

    The spectral footprint is derived from ``lambda / R`` and the trace
    wavelength-to-detector-pixel mapping. The spatial footprint is derived from
    the active PSF FWHM and the image-plane pixel area. This is a compact
    detector-scale sanity estimate, not an extracted-spectrum model.
    """
    if spectral_resolution is None:
        spectral_resolution = _required_cmd_float(
            ztrain.cmds, "!SIM.spectral.spectral_resolution",
        )
    if not np.isfinite(spectral_resolution) or spectral_resolution <= 0:
        raise ValueError(
            "Spectral resolution must be finite and positive; got "
            f"{spectral_resolution!r}."
        )
    seeing = _required_cmd_quantity(ztrain.cmds, "!OBS.seeing", u.arcsec)
    zenith_angle = _zenith_angle_from_airmass(
        _required_cmd_float(ztrain.cmds, "!OBS.airmass"),
    )
    fwhm_func, psf_effect = _configured_psf_fwhm_func(
        ztrain,
        allow_diagnostic_fallback=allow_diagnostic_psf,
    )
    traces_for_aperture = traces_by_aperture(
        get_effect(ztrain, "trace_list_analytical"),
    )

    rows: list[dict[str, Any]] = []
    for aperture_id, traces in traces_for_aperture.items():
        label = channel_label(aperture_id, traces)
        image_plane_id = int(traces[0].meta["image_plane_id"]) if traces else aperture_id
        image_plane = (
            ztrain.image_planes[image_plane_id]
            if hasattr(ztrain, "image_planes")
            and image_plane_id < len(ztrain.image_planes)
            else None
        )
        pixel_area = _image_plane_pixel_area(ztrain, image_plane_id)
        pixel_scale = np.sqrt(pixel_area.to_value(u.arcsec**2))
        wave_mid_values = []
        resolution_elements = []
        dispersion_values = []
        spectral_pixels = []
        psf_values = []
        spatial_pixels = []
        for trace in traces:
            wave_mid_nm = 500.0 * float(trace.wave_min + trace.wave_max)
            resolution_element_nm = wave_mid_nm / spectral_resolution
            dispersion_nm_pix = _trace_dispersion_nm_per_pixel(
                trace, image_plane, wave_mid_nm,
            )
            spectral_fwhm_pix = (
                resolution_element_nm / dispersion_nm_pix
                if np.isfinite(dispersion_nm_pix) and dispersion_nm_pix > 0
                else np.nan
            )
            psf_fwhm = fwhm_func(
                np.array([wave_mid_nm]) * u.nm,
                zenith_angle,
                seeing,
            )[0].to_value(u.arcsec)
            spatial_fwhm_pix = (
                psf_fwhm / pixel_scale
                if np.isfinite(pixel_scale) and pixel_scale > 0
                else np.nan
            )
            wave_mid_values.append(wave_mid_nm)
            resolution_elements.append(resolution_element_nm)
            dispersion_values.append(dispersion_nm_pix)
            spectral_pixels.append(spectral_fwhm_pix)
            psf_values.append(psf_fwhm)
            spatial_pixels.append(spatial_fwhm_pix)

        spectral_fwhm_pix = float(np.nanmedian(spectral_pixels))
        spatial_fwhm_pix = float(np.nanmedian(spatial_pixels))
        resel_pixels = (
            max(spectral_fwhm_pix, 1.0) * max(spatial_fwhm_pix, 1.0)
            if np.isfinite(spectral_fwhm_pix) and np.isfinite(spatial_fwhm_pix)
            else np.nan
        )
        rows.append({
            "aperture_id": int(aperture_id),
            "channel": label,
            "image_plane_id": image_plane_id,
            "current_slit_arcsec": _current_slit_arcsec(ztrain, label),
            "spectral_resolution_R": float(spectral_resolution),
            "wavelength_median_nm": float(np.nanmedian(wave_mid_values)),
            "resolution_element_median_nm": float(
                np.nanmedian(resolution_elements),
            ),
            "dispersion_median_nm_pix": float(np.nanmedian(dispersion_values)),
            "spectral_fwhm_pix": spectral_fwhm_pix,
            "psf_fwhm_median_arcsec": float(np.nanmedian(psf_values)),
            "pixel_scale_arcsec_pix": float(pixel_scale),
            "spatial_fwhm_pix": spatial_fwhm_pix,
            "resel_pixels_fwhm": float(resel_pixels),
            "snr_resel_scale": float(np.sqrt(resel_pixels)),
            "n_orders": len(traces),
            "psf_model": effect_name(psf_effect) if psf_effect is not None else "diagnostic",
            "note": (
                "Approximate FWHM footprint: spectral term from lambda/R and "
                "trace dispersion; spatial term from active PSF FWHM and "
                "image-plane pixel area. This is not an optimal extraction."
            ),
        })
    return Table(rows=rows)


def resolution_element_snr_summary_table(
    snr_images: list[np.ndarray],
    footprint_table: Table,
    *,
    titles: list[str] | None = None,
) -> Table:
    """Return per-channel median positive S/N scaled to a FWHM resel estimate."""
    titles = titles or [str(row["channel"]) for row in footprint_table]
    footprint_by_channel = {
        str(row["channel"]): row for row in footprint_table
    }
    rows: list[dict[str, Any]] = []
    for title, snr in zip(titles, snr_images, strict=True):
        channel = str(title)
        footprint = footprint_by_channel[channel]
        image = np.asarray(snr, dtype=float)
        positive = image[np.isfinite(image) & (image > 0)]
        median_pixel_snr = (
            float(np.nanmedian(positive)) if positive.size else np.nan
        )
        scale = float(footprint["snr_resel_scale"])
        rows.append({
            "channel": channel,
            "median_positive_pixel_snr": median_pixel_snr,
            "resel_pixels_fwhm": float(footprint["resel_pixels_fwhm"]),
            "snr_resel_scale": scale,
            "median_resel_snr": median_pixel_snr * scale,
            "positive_snr_pixels": int(positive.size),
            "spectral_fwhm_pix": float(footprint["spectral_fwhm_pix"]),
            "spatial_fwhm_pix": float(footprint["spatial_fwhm_pix"]),
        })
    return Table(rows=rows)


def readout_delta_summary_table(
    signal_hdul: Any,
    reference_hdul: Any,
    titles: list[str] | None = None,
) -> Table:
    """Summarize source-minus-reference detector readout differences."""
    signal_readouts = list(signal_hdul)
    reference_readouts = list(reference_hdul)
    if len(signal_readouts) != len(reference_readouts):
        raise ValueError(
            "signal_hdul and reference_hdul contain different readout counts: "
            f"{len(signal_readouts)} != {len(reference_readouts)}"
        )
    titles = titles or [f"detector {idx}" for idx in range(len(signal_readouts))]
    rows = []
    for idx, (title, signal, reference) in enumerate(
        zip(titles, signal_readouts, reference_readouts, strict=False),
    ):
        try:
            delta = np.asarray(signal.data, dtype=float) - np.asarray(
                reference.data, dtype=float,
            )
        except AttributeError:
            delta = np.asarray(signal[1].data, dtype=float) - np.asarray(
                reference[1].data, dtype=float,
            )
        finite = delta[np.isfinite(delta)]
        rows.append({
            "readout_id": idx,
            "channel": str(title),
            "shape": "x".join(str(value) for value in delta.shape),
            "sum_delta_e": float(np.nansum(delta)),
            "positive_delta_e": float(np.nansum(np.clip(delta, 0, None))),
            "negative_delta_e": float(np.nansum(np.clip(delta, None, 0))),
            "max_delta_e": float(np.nanmax(finite)) if finite.size else np.nan,
            "min_delta_e": float(np.nanmin(finite)) if finite.size else np.nan,
            "max_abs_delta_e": (
                float(np.nanmax(np.abs(finite))) if finite.size else np.nan
            ),
            "nonzero_pixels": int(np.count_nonzero(delta)),
        })
    return Table(rows=rows)


def _post_diffuse_channels_by_image_plane(
    post_diffuse_data: Mapping[str, Any] | None,
) -> dict[int, Mapping[str, Any]]:
    if post_diffuse_data is None:
        return {}
    return {
        int(channel["image_plane_id"]): channel
        for channel in post_diffuse_data.get("channels", {}).values()
    }


def _channel_mid_wavelength_nm(channel_data: Mapping[str, Any]) -> float:
    wave_min = float(channel_data.get("trace_wave_min_nm", np.nan))
    wave_max = float(channel_data.get("trace_wave_max_nm", np.nan))
    if np.isfinite(wave_min) and np.isfinite(wave_max):
        return 0.5 * (wave_min + wave_max)
    return np.nan


def _current_slit_arcsec(ztrain: Any, channel: str) -> float:
    key = "!INST.vis_curr_slit" if channel.upper() in {"B", "G", "R"} else (
        "!INST.nir_curr_slit"
    )
    return _required_cmd_float(ztrain.cmds, key)


def _science_truth_note(status: str) -> str:
    prefix = (
        "Diffuse rate is integrated image-plane background; "
        "resolution_element_nm is an anchor, not a spectral bin for this term."
    )
    if status == "saturated":
        return (
            f"{prefix} Additive detector signal exceeds configured full well."
        )
    if status == "near_saturation":
        return f"{prefix} Additive detector signal is near full well."
    return prefix


def _selected_detector_effect(
    ztrain: Any,
    selector_name: str,
    detector_row: Any,
) -> Any:
    selector = get_effect(ztrain, selector_name)
    selector_value = _selector_value_for_detector(selector, detector_row)
    if selector_value is None:
        raise ValueError(
            f"{selector_name!r} selector_key={selector.meta.get('selector_key')!r} "
            f"cannot be resolved for detector_id={int(detector_row['detector_id'])}."
        )
    try:
        return resolve_effect(selector, selector_value)
    except KeyError as exc:
        raise KeyError(
            f"{selector_name!r} has no entry for selector value "
            f"{selector_value!r} while resolving detector_id="
            f"{int(detector_row['detector_id'])}."
        ) from exc


def _resolved_detector_meta(
    effect: Any,
    key: str,
    cmds: Any,
    detector_id: int,
    *,
    default: float | None = None,
) -> float:
    from scopesim.utils import from_currsys

    if key not in getattr(effect, "meta", {}):
        if default is not None:
            return float(default)
        raise ValueError(
            f"{effect_name(effect)!r} has no required metadata key {key!r} "
            f"for detector_id={detector_id}."
        )
    value = from_currsys(effect.meta[key], cmds)
    if isinstance(value, Mapping):
        if detector_id in value:
            value = value[detector_id]
        elif str(detector_id) in value:
            value = value[str(detector_id)]
        else:
            raise ValueError(
                f"{effect_name(effect)!r} metadata key {key!r} has no "
                f"detector_id={detector_id} entry."
            )
        value = from_currsys(value, cmds)
    return float(value)


def _required_detector_cmd_float(
    cmds: Any,
    key: str,
    detector_id: int,
) -> float:
    value = _required_cmd_value(cmds, key)

    if isinstance(value, Mapping):
        if detector_id in value:
            value = value[detector_id]
        elif str(detector_id) in value:
            value = value[str(detector_id)]
        else:
            raise ValueError(
                f"Required ScopeSim command {key!r} has no "
                f"detector_id={detector_id} entry."
            )
        value = _resolved_for_display(value, cmds)
    elif (
        isinstance(value, (list, tuple, np.ndarray))
        and not hasattr(value, "unit")
    ):
        if len(value) <= detector_id:
            raise ValueError(
                f"Required ScopeSim command {key!r} has length {len(value)} "
                f"and no detector_id={detector_id} entry."
            )
        value = _resolved_for_display(value[detector_id], cmds)

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

    selector = get_effect(
        ztrain, "post_echelle_diffuse_background_selector", active_only=False,
    )
    if not getattr(selector, "include", True):
        return 0.0
    effect = resolve_effect(selector, image_plane_id)
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
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    qe_selector_name: str | None = "detector_qe_selector",
    active_only: bool = True,
) -> dict[str, Any]:
    """Build channel/order throughput data for transmission sanity plots.

    ``component_metadata`` supplies explicit metadata for selected single
    surface effects that ScopeSim/IRDB cannot currently tag. Configured
    metadata wins; conflicting overrides raise instead of silently replacing
    the instrument definition.
    """
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
        surface_rows = optical_surface_rows(
            components, wave, component_metadata=component_metadata,
        )
        optics_groups, group_counts = optical_surface_group_throughputs(
            surface_rows, wave,
        )
        telescope_throughput = optics_groups.get(
            "telescope",
            np.ones(wave.size, dtype=float),
        )
        instrument_optics_groups = OrderedDict(
            (name, values)
            for name, values in optics_groups.items()
            if name != "telescope"
        )
        optics_total = (
            np.prod(list(optics_groups.values()), axis=0)
            if optics_groups
            else np.ones(wave.size)
        )
        instrument_optics_total = (
            np.prod(list(instrument_optics_groups.values()), axis=0)
            if instrument_optics_groups
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
        pre_disperser_instrument_total = (
            dichroic_total * instrument_optics_total
        )
        pre_disperser_total = pre_disperser_instrument_total * telescope_throughput

        orders: OrderedDict[str, dict[str, Any]] = OrderedDict()
        qe_methods: set[str] = set()
        for trace in traces:
            order_eff = _as_float_array(
                trace_eff.efficiency_generator(trace.trace_id, wave),
            )
            mask = (wave >= trace.wave_min * u.um) & (wave <= trace.wave_max * u.um)
            if not np.any(mask):
                continue
            order_eff = np.where(mask, order_eff, np.nan)
            order_qe, qe_method = evaluate_trace_detector_qe(
                detector_qe, trace, wave, image_plane=image_plane,
            )
            order_qe = np.where(mask, order_qe, np.nan)
            qe_methods.add(qe_method)
            orders[trace.trace_id] = {
                "disperser": order_eff,
                "detector_qe": order_qe,
                "detector_qe_method": qe_method,
                "instrument": (
                    pre_disperser_instrument_total * order_eff * order_qe
                ),
                "total_with_telescope_no_slit": (
                    pre_disperser_total * order_eff * order_qe
                ),
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
            "instrument_optics_groups": instrument_optics_groups,
            "telescope_throughput": telescope_throughput,
            "optics_total": optics_total,
            "instrument_optics_total": instrument_optics_total,
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
            "pre_disperser_instrument_total": pre_disperser_instrument_total,
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
            channel["instrument_optics_total"],
            channel["telescope_throughput"],
            channel["detector_qe"],
            channel["detector_qe_midpoint"],
            channel["pre_disperser_instrument_total"],
            channel["pre_disperser_total"],
        ]
        arrays.extend(channel["optics_groups"].values())
        arrays.extend(order["disperser"] for order in channel["orders"].values())
        arrays.extend(order["detector_qe"] for order in channel["orders"].values())
        arrays.extend(order["instrument"] for order in channel["orders"].values())
        arrays.extend(
            order["total_with_telescope_no_slit"]
            for order in channel["orders"].values()
        )
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


def snr_selected_spectral_bins(
    paired_samples: Table,
    *,
    bin_factors: tuple[int, ...],
    integration: str,
    slit_arcsec: float,
    ao_enabled: bool,
) -> tuple[Table, Table]:
    """Calculate all-element and S/N-selected limiting-magnitude curves.

    Unprefixed input columns describe the counterfactual ``no OH`` sky;
    ``simulation_*`` columns describe the configured sky including
    airglow/interline emission.

    ``contiguous_order_run`` starts when sample indices, detector rectangles,
    or wavelength edges stop being contiguous. ``bin_index_within_run`` is
    local and zero-based. ``non_source_variance_e2`` is the matched-sky plus
    detector variance that does not scale with trial source flux.
    ``retained_source_signal_fraction`` is selected signal divided by all
    available signal. ``complete_native_bin`` requires exactly ``bin_factor``
    valid contiguous native elements.
    """
    native = paired_samples[
        (paired_samples["bin_factor"] == 1)
        & (paired_samples["native_count"] == 1)
    ]
    unsorted_channels = np.asarray(native["channel"], dtype=str)
    channel_order = sorted(
        set(unsorted_channels),
        key=lambda channel: np.median(native["wave_nm"][unsorted_channels == channel]),
    )
    native = native[np.lexsort((
        np.asarray(native["sample_index"], dtype=int),
        np.asarray(native["trace_id"], dtype=str),
        np.asarray(native["channel"], dtype=str),
    ))]

    sky_models = ("simulation", "no OH")
    source_signal_native = np.stack((
        np.asarray(native["simulation_signal_e"], dtype=float),
        np.asarray(native["signal_e"], dtype=float),
    ))
    source_shot_variance_native = np.stack((
        np.asarray(native["simulation_source_var_e"], dtype=float),
        np.asarray(native["source_var_e"], dtype=float),
    ))
    non_source_variance_native = np.stack((
        np.asarray(native["simulation_fixed_variance_e"], dtype=float),
        np.asarray(native["fixed_variance_e"], dtype=float),
    ))
    sky_variance_native = np.stack((
        np.asarray(native["simulation_background_var_e"], dtype=float),
        np.asarray(native["background_var_e"], dtype=float),
    ))
    detector_variance_native = np.stack((
        np.asarray(native["simulation_detector_var_e"], dtype=float),
        np.asarray(native["detector_var_e"], dtype=float),
    ))
    native_wave_nm = np.asarray(native["wave_nm"], dtype=float)
    native_width_nm = np.asarray(native["wave_high_nm"] - native["wave_low_nm"], dtype=float)
    native_resolution_width_nm = np.asarray(native["resolution_width_nm"], dtype=float)
    added_configured_sky_e = np.asarray(
        native["simulation_background_e"] - native["background_e"], dtype=float,
    )

    input_abmag = float(native["input_abmag"][0])
    target_snr = float(native["target_snr"][0])
    q2 = target_snr**2
    curve_rows, native_rows = [], []

    for bin_factor in bin_factors:
        bin_factor = int(bin_factor)
        if bin_factor <= 0:
            raise ValueError(f"bin_factor must be positive, got {bin_factor}")

        channels = np.asarray(native["channel"], dtype=str)
        traces = np.asarray(native["trace_id"], dtype=str)
        bin_native_indices, bin_metadata = [], []
        for channel in channel_order:
            channel_rows = channels == channel
            trace_order = sorted(
                set(traces[channel_rows]),
                key=lambda value: (
                    str(value).rpartition("_")[0],
                    int(str(value).rpartition("_")[2]),
                ),
            )
            for trace_id in trace_order:
                indices = np.flatnonzero(channel_rows & (traces == trace_id))
                indices = indices[np.argsort(native["sample_index"][indices])]
                new_run = np.r_[
                    True,
                    (np.diff(native["sample_index"][indices]) != 1)
                    | (native["x1"][indices[:-1]] != native["x0"][indices[1:]])
                    | ~np.isclose(
                        native["wave_high_nm"][indices[:-1]],
                        native["wave_low_nm"][indices[1:]],
                        rtol=0, atol=1e-12,
                    ),
                ]
                contiguous_runs = np.cumsum(new_run)
                for contiguous_order_run in np.unique(contiguous_runs):
                    run_indices = indices[contiguous_runs == contiguous_order_run]
                    for bin_index_within_run, start in enumerate(
                        range(0, len(run_indices), bin_factor)
                    ):
                        group = run_indices[start:start + bin_factor]
                        bin_native_indices.append(group)
                        bin_metadata.append({
                            "channel": channel,
                            "trace_id": trace_id,
                            "contiguous_order_run": int(contiguous_order_run),
                            "bin_index_within_run": bin_index_within_run,
                            "native_start_sample_index": int(native["sample_index"][group[0]]),
                            "native_stop_sample_index": int(native["sample_index"][group[-1]]) + 1,
                            "nominal_wavelength_low_nm": float(native["wave_low_nm"][group[0]]),
                            "nominal_wavelength_high_nm": float(native["wave_high_nm"][group[-1]]),
                        })

        n_bins = len(bin_native_indices)
        padded_native_index = np.full((n_bins, bin_factor), -1, dtype=int)
        for bin_index, indices in enumerate(bin_native_indices):
            padded_native_index[bin_index, :len(indices)] = indices
        native_present = padded_native_index >= 0
        assert np.all(np.bincount(
            padded_native_index[native_present], minlength=len(native),
        ) == 1)
        safe_index = np.where(native_present, padded_native_index, 0)

        def gather(values):
            values = np.asarray(values)
            if values.ndim == 1:
                values = values[None, :]
            return np.take_along_axis(
                values[:, None, :], safe_index[None, :, :], axis=2,
            )

        source_signal = gather(source_signal_native)
        source_shot_variance = gather(source_shot_variance_native)
        non_source_variance = gather(non_source_variance_native)
        sky_variance = gather(sky_variance_native)
        detector_variance = gather(detector_variance_native)
        wavelength_nm = gather(native_wave_nm)[0]
        wavelength_width_nm = gather(native_width_nm)[0]
        resolution_width_nm = np.where(
            native_present, gather(native_resolution_width_nm)[0], 0.0,
        ).sum(axis=1)

        valid = (
            native_present[None, :, :]
            & np.isfinite(source_signal)
            & np.isfinite(source_shot_variance)
            & np.isfinite(non_source_variance)
            & (source_shot_variance + non_source_variance > 0)
        )
        available_count = valid.sum(axis=2)
        assert np.all(available_count > 0)
        native_snr = np.full_like(source_signal, -np.inf)
        native_snr[valid] = (
            source_signal[valid]
            / np.sqrt(source_shot_variance[valid] + non_source_variance[valid])
        )

        available_signal = np.where(valid, source_signal, 0.0)
        available_source_variance = np.where(valid, source_shot_variance, 0.0)
        available_non_source_variance = np.where(valid, non_source_variance, 0.0)
        available_sky_variance = np.where(valid, sky_variance, 0.0)
        available_detector_variance = np.where(valid, detector_variance, 0.0)
        sort_order = np.argsort(-native_snr, axis=2, kind="stable")
        sorted_signal = np.take_along_axis(available_signal, sort_order, axis=2)
        sorted_variance = np.take_along_axis(
            available_source_variance + available_non_source_variance,
            sort_order, axis=2,
        )
        prefix_available = (
            np.arange(bin_factor)[None, None, :] < available_count[:, :, None]
        )
        cumulative_snr = np.full_like(sorted_signal, np.nan)
        cumulative_snr[prefix_available] = (
            np.cumsum(sorted_signal, axis=2)[prefix_available]
            / np.sqrt(np.cumsum(sorted_variance, axis=2)[prefix_available])
        )
        best_prefix_count = np.nanargmax(cumulative_snr, axis=2) + 1
        sorted_retained = (
            np.arange(bin_factor)[None, None, :] < best_prefix_count[:, :, None]
        )
        retained_mask = np.zeros_like(sorted_retained)
        np.put_along_axis(retained_mask, sort_order, sorted_retained, axis=2)
        retained_mask &= valid
        retained_count = retained_mask.sum(axis=2)
        assert np.all((retained_count >= 1) & (retained_count <= available_count))

        components = {
            "source_signal_e": available_signal,
            "source_shot_variance_e2": available_source_variance,
            "sky_variance_e2": available_sky_variance,
            "detector_variance_e2": available_detector_variance,
            "non_source_variance_e2": available_non_source_variance,
        }
        all_sums = {name: values.sum(axis=2) for name, values in components.items()}
        selected_sums = {
            name: (values * retained_mask).sum(axis=2)
            for name, values in components.items()
        }
        all_total_variance = (
            all_sums["source_shot_variance_e2"]
            + all_sums["non_source_variance_e2"]
        )
        selected_total_variance = (
            selected_sums["source_shot_variance_e2"]
            + selected_sums["non_source_variance_e2"]
        )
        all_snr = all_sums["source_signal_e"] / np.sqrt(all_total_variance)
        selected_snr = (
            selected_sums["source_signal_e"] / np.sqrt(selected_total_variance)
        )
        assert np.all(
            selected_snr >= all_snr - 1e-12 * np.maximum(1.0, np.abs(all_snr))
        )

        available_width_nm = np.where(valid, wavelength_width_nm[None, :, :], 0.0)
        selected_wavelength_fraction = (
            (available_width_nm * retained_mask).sum(axis=2)
            / available_width_nm.sum(axis=2)
        )
        all_effective_wave_nm = (
            wavelength_nm[None, :, :] * available_signal
        ).sum(axis=2) / all_sums["source_signal_e"]
        selected_effective_wave_nm = (
            wavelength_nm[None, :, :] * available_signal * retained_mask
        ).sum(axis=2) / selected_sums["source_signal_e"]
        selected_signal_fraction = (
            selected_sums["source_signal_e"] / all_sums["source_signal_e"]
        )

        for sky_index, sky_model in enumerate(sky_models):
            for selection_name, sums, total_variance, snr, counts, width_fraction, effective_wave, signal_fraction in (
                (
                    "all elements", all_sums, all_total_variance, all_snr,
                    available_count, np.ones((2, n_bins)), all_effective_wave_nm,
                    np.ones((2, n_bins)),
                ),
                (
                    "S/N-selected", selected_sums, selected_total_variance,
                    selected_snr, retained_count, selected_wavelength_fraction,
                    selected_effective_wave_nm, selected_signal_fraction,
                ),
            ):
                signal = sums["source_signal_e"][sky_index]
                source_variance = sums["source_shot_variance_e2"][sky_index]
                non_source = sums["non_source_variance_e2"][sky_index]
                discriminant = (
                    (q2 * source_variance) ** 2
                    + 4 * signal**2 * q2 * non_source
                )
                flux_scale = (
                    q2 * source_variance + np.sqrt(discriminant)
                ) / (2 * signal**2)
                limiting_magnitude = input_abmag - 2.5 * np.log10(flux_scale)

                for bin_index, metadata in enumerate(bin_metadata):
                    nominal_wave_nm = 0.5 * (
                        metadata["nominal_wavelength_low_nm"]
                        + metadata["nominal_wavelength_high_nm"]
                    )
                    complete = (
                        len(bin_native_indices[bin_index]) == bin_factor
                        and available_count[sky_index, bin_index] == bin_factor
                    )
                    curve_rows.append({
                        "integration": integration,
                        "slit_arcsec": float(slit_arcsec),
                        "ao_enabled": bool(ao_enabled),
                        "bin_factor": bin_factor,
                        "sky_model": sky_model,
                        "native_element_selection": selection_name,
                        **metadata,
                        "nominal_wavelength_nm": nominal_wave_nm,
                        "effective_wavelength_nm": effective_wave[sky_index, bin_index],
                        "calculated_resolving_power": nominal_wave_nm / resolution_width_nm[bin_index],
                        "input_abmag": input_abmag,
                        "target_snr": target_snr,
                        "limiting_magnitude_ab": limiting_magnitude[bin_index],
                        **{
                            name: values[sky_index, bin_index]
                            for name, values in sums.items()
                        },
                        "total_variance_e2": total_variance[sky_index, bin_index],
                        "snr_at_input": snr[sky_index, bin_index],
                        "available_native_elements": int(available_count[sky_index, bin_index]),
                        "retained_native_elements": int(counts[sky_index, bin_index]),
                        "retained_wavelength_fraction": width_fraction[sky_index, bin_index],
                        "retained_source_signal_fraction": signal_fraction[sky_index, bin_index],
                        "complete_native_bin": bool(complete),
                    })

        for sky_index, sky_model in enumerate(sky_models):
            for bin_index, metadata in enumerate(bin_metadata):
                for slot, native_index in enumerate(padded_native_index[bin_index]):
                    if native_index < 0:
                        continue
                    native_rows.append({
                        "integration": integration,
                        "slit_arcsec": float(slit_arcsec),
                        "ao_enabled": bool(ao_enabled),
                        "bin_factor": bin_factor,
                        "sky_model": sky_model,
                        "channel": metadata["channel"],
                        "trace_id": metadata["trace_id"],
                        "contiguous_order_run": metadata["contiguous_order_run"],
                        "bin_index_within_run": metadata["bin_index_within_run"],
                        "native_sample_index": int(native["sample_index"][native_index]),
                        "native_wavelength_low_nm": float(native["wave_low_nm"][native_index]),
                        "native_wavelength_high_nm": float(native["wave_high_nm"][native_index]),
                        "native_wavelength_nm": float(native["wave_nm"][native_index]),
                        "source_signal_e": source_signal[sky_index, bin_index, slot],
                        "source_shot_variance_e2": source_shot_variance[sky_index, bin_index, slot],
                        "non_source_variance_e2": non_source_variance[sky_index, bin_index, slot],
                        "total_variance_e2": (
                            source_shot_variance[sky_index, bin_index, slot]
                            + non_source_variance[sky_index, bin_index, slot]
                        ),
                        "native_snr": native_snr[sky_index, bin_index, slot],
                        "available_for_selection": bool(valid[sky_index, bin_index, slot]),
                        "retained_by_snr_selection": bool(retained_mask[sky_index, bin_index, slot]),
                        "configured_minus_no_oh_background_e": added_configured_sky_e[native_index],
                    })

    return Table(rows=curve_rows), Table(rows=native_rows)


def fixed_bin_no_oh_limiting_magnitude_curves(
    samples: Table,
    *,
    bin_factors: tuple[int, ...],
    integration: str,
    slit_arcsec: float,
    ao_enabled: bool,
) -> Table:
    """Reshape authoritative fixed-bin no-OH products into catalog rows.

    Existing ``m5_ab`` values are copied, not recalculated. Native rows supply
    only the source-weighted effective wavelength.
    """
    selected = samples[np.isin(samples["bin_factor"], bin_factors)]
    native = samples[
        (samples["bin_factor"] == 1) & (samples["native_count"] == 1)
    ]
    channels = np.asarray(selected["channel"], dtype=str)
    traces = np.asarray(selected["trace_id"], dtype=str)
    channel_order = sorted(
        set(channels),
        key=lambda channel: np.median(selected["wave_nm"][channels == channel]),
    )
    rows = []

    for channel in channel_order:
        channel_rows = channels == channel
        for trace_id in sorted(set(traces[channel_rows])):
            for bin_factor in bin_factors:
                indices = np.flatnonzero(
                    channel_rows
                    & (traces == trace_id)
                    & (selected["bin_factor"] == bin_factor)
                )
                indices = indices[np.argsort(selected["bin_index"][indices])]
                if len(indices) == 0:
                    continue
                new_run = np.r_[
                    True,
                    (np.diff(selected["bin_index"][indices]) != 1)
                    | (
                        selected["native_stop_index"][indices[:-1]]
                        != selected["native_start_index"][indices[1:]]
                    )
                    | ~np.isclose(
                        selected["wave_high_nm"][indices[:-1]],
                        selected["wave_low_nm"][indices[1:]],
                        rtol=0, atol=1e-12,
                    ),
                ]
                contiguous_runs = np.cumsum(new_run)
                for contiguous_order_run in np.unique(contiguous_runs):
                    run_indices = indices[contiguous_runs == contiguous_order_run]
                    for bin_index_within_run, row_index in enumerate(run_indices):
                        sample = selected[row_index]
                        native_rows = native[
                            (native["channel"] == channel)
                            & (native["trace_id"] == trace_id)
                            & (native["sample_index"] >= sample["native_start_index"])
                            & (native["sample_index"] < sample["native_stop_index"])
                        ]
                        effective_wave_nm = np.sum(
                            native_rows["wave_nm"] * native_rows["signal_e"]
                        ) / np.sum(native_rows["signal_e"])
                        rows.append({
                            "integration": integration,
                            "slit_arcsec": float(slit_arcsec),
                            "ao_enabled": bool(ao_enabled),
                            "bin_factor": int(bin_factor),
                            "sky_model": "no OH",
                            "native_element_selection": "all elements",
                            "channel": channel,
                            "trace_id": trace_id,
                            "contiguous_order_run": int(contiguous_order_run),
                            "bin_index_within_run": bin_index_within_run,
                            "native_start_sample_index": int(sample["native_start_index"]),
                            "native_stop_sample_index": int(sample["native_stop_index"]),
                            "nominal_wavelength_low_nm": float(sample["wave_low_nm"]),
                            "nominal_wavelength_high_nm": float(sample["wave_high_nm"]),
                            "nominal_wavelength_nm": float(sample["wave_nm"]),
                            "effective_wavelength_nm": effective_wave_nm,
                            "calculated_resolving_power": float(sample["extraction_R"]),
                            "input_abmag": float(sample["input_abmag"]),
                            "target_snr": float(sample["target_snr"]),
                            "limiting_magnitude_ab": float(sample["m5_ab"]),
                            "source_signal_e": float(sample["signal_e"]),
                            "source_shot_variance_e2": float(sample["source_var_e"]),
                            "sky_variance_e2": float(sample["background_var_e"]),
                            "detector_variance_e2": float(sample["detector_var_e"]),
                            "non_source_variance_e2": float(sample["fixed_variance_e"]),
                            "total_variance_e2": float(
                                sample["source_var_e"] + sample["fixed_variance_e"]
                            ),
                            "snr_at_input": float(sample["snr_at_input"]),
                            "available_native_elements": int(sample["native_count"]),
                            "retained_native_elements": int(sample["native_count"]),
                            "retained_wavelength_fraction": 1.0,
                            "retained_source_signal_fraction": 1.0,
                            "complete_native_bin": bool(sample["native_count"] == bin_factor),
                        })
    return Table(rows=rows)
