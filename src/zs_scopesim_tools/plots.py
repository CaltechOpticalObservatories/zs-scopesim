"""Plotting helpers for ZShooter ScopeSim validation notebooks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM


def _as_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def _source_label(source: Any) -> str:
    meta = getattr(source, "meta", {}) or {}
    for key in ("name", "object", "description", "function_call"):
        value = meta.get(key)
        if value:
            return str(value)
    return f"{source.__class__.__name__}@{id(source):x}"


def _row_label(source: Any, table: Table, row_index: int) -> str:
    for key in ("label", "name", "object", "source"):
        if key in table.colnames:
            value = table[key][row_index]
            if value is not None and str(value):
                return str(value)
    return f"{_source_label(source)} row {row_index}"


def _plot_spectrum_values(
    ax: Any,
    wave: u.Quantity,
    values: Any,
    *,
    label: str,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    ax.plot(
        wave.to_value(u.um),
        _as_float_array(values),
        label=label,
        lw=linewidth,
        alpha=alpha,
        ls=linestyle,
    )


def _evaluate_spectrum_values(spectrum: Any, wave: u.Quantity, weight: float = 1.0) -> Any:
    return spectrum(wave) * weight


def _source_field_image_data(field: Any) -> np.ndarray | None:
    if hasattr(field, "data"):
        return np.asarray(field.data)
    source_field = getattr(field, "field", None)
    data = getattr(source_field, "data", None)
    if data is None:
        return None
    data = np.asarray(data)
    return data if data.ndim >= 2 else None


def _is_table_source_field(field: Any) -> bool:
    table = getattr(field, "field", None)
    return isinstance(table, Table) and {"x", "y"}.issubset(table.colnames)


def _quantity_column(table: Table, name: str, unit: u.UnitBase) -> u.Quantity:
    values = table[name]
    quantity = getattr(values, "quantity", values)
    quantity = u.Quantity(quantity)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity.value * unit
    return quantity.to(unit)


def _plot_table_source_field(
    source: Any,
    ax_image: Any,
    ax_spectrum: Any,
    field: Any,
    wave: u.Quantity,
    *,
    individual: bool,
    spectrum_linewidth: float,
    spectrum_alpha: float,
    spectrum_linestyle: str,
) -> None:
    table = field.field
    x = _quantity_column(table, "x", u.arcsec).to_value(u.arcsec)
    y = _quantity_column(table, "y", u.arcsec).to_value(u.arcsec)
    refs = (
        np.asarray(table["ref"], dtype=int)
        if "ref" in table.colnames
        else np.zeros(len(table), dtype=int)
    )
    weights = (
        np.asarray(table["weight"], dtype=float)
        if "weight" in table.colnames
        else np.ones(len(table), dtype=float)
    )
    sizes = 36 + 84 * weights / max(np.nanmax(weights), 1.0)
    ax_image.axhline(0, color="0.65", lw=0.8, zorder=0)
    ax_image.axvline(0, color="0.65", lw=0.8, zorder=0)
    scatter = ax_image.scatter(
        x,
        y,
        c=refs,
        s=sizes,
        cmap="tab10",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    if len(table) <= 10:
        for idx, (xpos, ypos) in enumerate(zip(x, y, strict=True)):
            ax_image.annotate(
                _row_label(source, table, idx),
                (xpos, ypos),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=4,
            )
    ax_image.set_aspect("equal", adjustable="datalim")
    ax_image.set_xlabel("x [arcsec]")
    ax_image.set_ylabel("y [arcsec]")
    ax_image.set_title("Source Positions")
    if len(np.unique(refs)) > 1:
        ax_image.legend(
            *scatter.legend_elements(prop="colors", fmt="{x:.0f}"),
            title="ref",
            frameon=False,
            loc="best",
        )

    spectra = getattr(field, "spectra", {})
    if individual:
        for row_index, ref in enumerate(refs):
            spectrum = spectra.get(int(ref), spectra.get(ref))
            if spectrum is None:
                continue
            values = _evaluate_spectrum_values(
                spectrum, wave, weight=float(weights[row_index]),
            )
            _plot_spectrum_values(
                ax_spectrum,
                wave,
                values,
                label=_row_label(source, table, row_index),
                linewidth=spectrum_linewidth,
                alpha=spectrum_alpha,
                linestyle=spectrum_linestyle,
            )
        return

    total_values = None
    for row_index, ref in enumerate(refs):
        spectrum = spectra.get(int(ref), spectra.get(ref))
        if spectrum is None:
            continue
        values = _evaluate_spectrum_values(
            spectrum, wave, weight=float(weights[row_index]),
        )
        total_values = values if total_values is None else total_values + values
    if total_values is not None:
        _plot_spectrum_values(
            ax_spectrum,
            wave,
            total_values,
            label=f"{_source_label(source)} total ({len(table)} points)",
            linewidth=spectrum_linewidth,
            alpha=spectrum_alpha,
            linestyle=spectrum_linestyle,
        )


def _plot_image_source_field(
    ax_image: Any,
    ax_spectrum: Any,
    field: Any,
    wave: u.Quantity,
    *,
    spectrum_linewidth: float,
    spectrum_alpha: float,
    spectrum_linestyle: str,
) -> None:
    data = _source_field_image_data(field)
    if data is None:
        ax_image.text(
            0.5, 0.5, "No spatial image", ha="center", va="center",
            transform=ax_image.transAxes,
        )
        ax_image.set_axis_off()
    else:
        if data.ndim > 2:
            data = np.nanmean(data, axis=0)
        ax_image.imshow(data, origin="lower", cmap="viridis")
        ax_image.set_title("Spatial Profile")

    try:
        spectrum = field.spectrum
    except Exception:
        spectrum = None
    if spectrum is not None:
        _plot_spectrum_values(
            ax_spectrum,
            wave,
            _evaluate_spectrum_values(spectrum, wave),
            label="spectrum",
            linewidth=spectrum_linewidth,
            alpha=spectrum_alpha,
            linestyle=spectrum_linestyle,
        )
    else:
        for ref, candidate in getattr(field, "spectra", {}).items():
            _plot_spectrum_values(
                ax_spectrum,
                wave,
                _evaluate_spectrum_values(candidate, wave),
                label=f"ref {ref}",
                linewidth=spectrum_linewidth,
                alpha=spectrum_alpha,
                linestyle=spectrum_linestyle,
            )


def plot_source(
    source: Any,
    wave: u.Quantity | None = None,
    *,
    individual: bool = False,
    spectrum_yscale: str = "linear",
    spectrum_linewidth: float = 1.5,
    spectrum_alpha: float = 1.0,
    spectrum_linestyle: str = "-",
):
    """Plot each source field's spatial profile/positions and spectrum."""
    import matplotlib.pyplot as plt

    wave = wave if wave is not None else np.linspace(0.3, 2.5, 1001) * u.um
    num_fields = len(source.fields)
    if num_fields == 0:
        raise ValueError(
            "Source contains no fields. For point sources built with "
            "scopesim.source.source.Source, pass x=[...], y=[...], and ref=[...] "
            "arrays so ScopeSim creates a table-backed source field."
        )
    fig, axs = plt.subplots(
        figsize=(6, 2 * num_fields),
        nrows=num_fields,
        ncols=2,
        width_ratios=[1, 2],
        constrained_layout=True,
    )
    if num_fields == 1:
        axs = np.array([axs])
    for idx, field in enumerate(source.fields):
        ax_image, ax_spectrum = axs[idx]
        if _is_table_source_field(field):
            _plot_table_source_field(
                source, ax_image, ax_spectrum, field, wave,
                individual=individual,
                spectrum_linewidth=spectrum_linewidth,
                spectrum_alpha=spectrum_alpha,
                spectrum_linestyle=spectrum_linestyle,
            )
        else:
            _plot_image_source_field(
                ax_image,
                ax_spectrum,
                field,
                wave,
                spectrum_linewidth=spectrum_linewidth,
                spectrum_alpha=spectrum_alpha,
                spectrum_linestyle=spectrum_linestyle,
            )
        ax_spectrum.set_title("Spectrum")
        ax_spectrum.set_xlabel("Wavelength [um]")
        ax_spectrum.set_yscale(spectrum_yscale)
        ax_spectrum.grid(alpha=0.2)
        if ax_spectrum.get_legend_handles_labels()[0]:
            ax_spectrum.legend(frameon=False)
    return fig, axs


def _plot_quantity_values(values: u.Quantity) -> np.ndarray:
    try:
        return values.to_value(PHOTLAM)
    except Exception:
        return _as_float_array(values)


def plot_post_disperser_diffuse_background(data: Mapping[str, Any]):
    """Plot post-disperser diffuse spectra and integrated image-plane rates."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.2), sharex=True, constrained_layout=True,
    )
    colors = {
        "camera": "tab:cyan",
        "collimator": "tab:green",
        "preoptics": "tab:blue",
        "other": "0.5",
    }
    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        trace_min = channel.get("trace_wave_min_nm", np.nan)
        trace_max = channel.get("trace_wave_max_nm", np.nan)
        if np.isfinite(trace_min) and np.isfinite(trace_max):
            ax.axvspan(
                trace_min, trace_max, color="0.2", alpha=0.08,
                label="trace wavelength span",
            )

        for name, spectrum in channel["spectra"].items():
            ax.plot(
                wave,
                _plot_quantity_values(spectrum),
                lw=1.4,
                color=colors.get(name, "0.5"),
                label=f"{name} diffuse",
            )
        if channel.get("total_spectrum_without_blocking") is not None:
            ax.plot(
                wave,
                _plot_quantity_values(
                    channel["total_spectrum_without_blocking"],
                ),
                lw=2.0,
                ls="--",
                color="0.25",
                alpha=0.85,
                label="total without ir block",
            )
        if channel["total_spectrum"] is not None:
            ax.plot(
                wave,
                _plot_quantity_values(channel["total_spectrum"]),
                lw=2.2,
                color="black",
                alpha=0.95,
                label="total with ir block",
            )
        blocked_rate = float(channel.get("blocking_delta_rate_ph_s_pix", 0.0))
        unblocked_rate = float(
            channel.get("total_rate_without_blocking_ph_s_pix", np.nan),
        )
        if np.isfinite(unblocked_rate) and unblocked_rate > 0:
            blocked_fraction = blocked_rate / unblocked_rate
            blocked_text = (
                f"IR block removes {blocked_rate:.2g} ph/s/pix\n"
                f"({blocked_fraction:.1%} of unblocked diffuse)"
            )
        else:
            blocked_text = f"IR block removes {blocked_rate:.2g} ph/s/pix"
        ax.text(
            0.02,
            0.96,
            blocked_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "0.75",
                "alpha": 0.78,
            },
        )
        ax.set_title(
            f"{channel['label']} Image Plane {channel['image_plane_id']}: "
            f"{channel['total_rate_ph_s_pix']:.3g} ph/s/pix",
            pad=8,
        )
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spectral background [PHOTLAM equiv.]")
    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    dedup = OrderedDict(zip(labels, handles))
    fig.suptitle("Post-Disperser Diffuse Background", fontsize=12)
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside lower center",
        ncol=5, frameon=False,
    )
    return fig, axes


def plot_emissivity_sanity(data: Mapping[str, Any]):
    """Plot split pre/post-disperser emissivity sanity-check data."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    group_colors = {
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "other": "0.5",
    }

    ymax = 0.0
    for channel in data["channels"].values():
        curves = (
            list(channel["pre_disperser_terms"].values())
            + list(channel["post_disperser_terms"].values())
            + [channel["pre_disperser_output_equiv"],
               channel["post_disperser_after_qe"],
               channel.get("post_disperser_without_blocking_after_qe",
                           np.zeros_like(channel["post_disperser_after_qe"]))]
        )
        for curve in curves:
            finite = curve[np.isfinite(curve)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)))
    ymax = max(1e-3, ymax * 1.08)

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["pre_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.0, ls="--", alpha=0.75,
                color=group_colors.get(name, "0.5"),
                label=f"pre {name}",
            )
        for name, values in channel["post_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.1, ls="-", alpha=0.7,
                color=group_colors.get(name, "0.5"),
                label=f"post {name}",
            )

        ax.plot(
            wave, channel["pre_disperser_output_equiv"], lw=2.0,
            color="tab:purple", alpha=0.9,
            label="pre total before trace QE", zorder=8,
        )
        ax.plot(
            wave,
            channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            ),
            lw=2.0,
            ls="--",
            color="0.25",
            alpha=0.85,
            label="post diffuse without ir block",
            zorder=9,
        )
        ax.plot(
            wave, channel["post_disperser_after_qe"], lw=2.4,
            color="black", alpha=0.95,
            label="post diffuse with ir block+QE", zorder=10,
        )
        blocked_delta = channel.get("post_disperser_blocked_delta")
        if blocked_delta is not None and np.nanmax(np.abs(blocked_delta)) > 0:
            ax.plot(
                wave, blocked_delta, lw=1.8, ls="-.",
                color="tab:brown", alpha=0.9,
                label="removed by ir block", zorder=9,
            )
        extract_curve = channel.get("post_disperser_extract_equiv_rate_ph_s")
        if extract_curve is not None:
            rate_ax = ax.twinx()
            rate_ax.plot(
                wave,
                extract_curve,
                lw=1.2,
                ls=":",
                color="tab:orange",
                alpha=0.8,
                label="integrated diffuse extraction equiv.",
            )
            ax.plot(
                [], [], lw=1.2, ls=":", color="tab:orange",
                alpha=0.8, label="integrated diffuse extraction equiv.",
            )
            rate_ax.set_yscale("symlog", linthresh=1e-6)
            rate_ax.tick_params(axis="y", labelsize=7, colors="tab:orange")
            if aperture_id in (2, 5):
                rate_ax.set_ylabel("Integrated diffuse [ph/s]", color="tab:orange")
            peak_rate = float(np.nanmax(extract_curve))
        else:
            peak_rate = np.nan
        peak_lines = []
        for label, values in (
            ("pre", channel["pre_disperser_output_equiv"]),
            ("post", channel["post_disperser_after_qe"]),
            ("no block", channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            )),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                peak_lines.append(f"{label}: {np.nanmax(finite):.2g}")
        if np.isfinite(peak_rate):
            peak_lines.append(f"int: {peak_rate:.2g} ph/s")
        ax.text(
            0.02,
            0.04,
            "peak " + "\n".join(peak_lines),
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                  "ec": "0.75", "alpha": 0.78},
        )
        ax.plot(
            wave, channel["detector_qe"], lw=0.9, ls=":",
            color="tab:red", alpha=0.8, label="QE throughput",
        )
        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Thermal emission density [PHOTLAM equiv.] (symlog)")

    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    dedup = OrderedDict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside upper center",
        ncol=6, frameon=False,
    )
    return fig, axes


def plot_detector_background_budget(table: Table):
    """Plot additive detector-background and noise terms by channel."""
    import matplotlib.pyplot as plt

    channels = [str(value) for value in table["channel"]]
    x = np.arange(len(channels))
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.6), constrained_layout=True,
    )
    signal_ax, noise_ax = axes

    diffuse = np.asarray(table["post_diffuse_e_pix"], dtype=float)
    dark = np.asarray(table["dark_current_e_pix"], dtype=float)
    bias = np.asarray(table["bias_e_pix"], dtype=float)
    if "additive_signal_e_pix" in table.colnames:
        additive_signal = np.asarray(table["additive_signal_e_pix"], dtype=float)
    else:
        additive_signal = diffuse + dark
    signal_ax.bar(x, diffuse, width=0.7, label="post-disperser diffuse")
    signal_ax.bar(x, dark, width=0.7, bottom=diffuse, label="dark current")
    signal_ax.plot(x, bias, "o", color="black", label="bias offset")
    if "full_well_e" in table.colnames:
        full_well = np.asarray(table["full_well_e"], dtype=float)
        first_full_well = True
        for xi, well_depth in zip(x, full_well, strict=True):
            if not np.isfinite(well_depth) or well_depth <= 0:
                continue
            signal_ax.hlines(
                well_depth, xi - 0.36, xi + 0.36,
                colors="tab:red", linewidth=2.5,
                label="full well" if first_full_well else None,
            )
            first_full_well = False
    signal_ax.set_yscale("symlog", linthresh=1.0)
    signal_ax.set_xticks(x, channels)
    signal_ax.set_ylabel("Detector signal [e-/pix]")
    signal_ax.set_title("Additive Signal")
    signal_ax.grid(axis="y", alpha=0.2)
    if "saturation_status" in table.colnames:
        statuses = [str(value) for value in table["saturation_status"]]
    elif "full_well_e" in table.colnames:
        statuses = []
        full_well = np.asarray(table["full_well_e"], dtype=float)
        for signal, well_depth in zip(additive_signal, full_well, strict=True):
            if not np.isfinite(well_depth) or well_depth <= 0:
                statuses.append("unknown")
            elif signal >= well_depth:
                statuses.append("saturated")
            elif signal >= 0.8 * well_depth:
                statuses.append("near_saturation")
            else:
                statuses.append("ok")
    else:
        statuses = ["unknown"] * len(channels)
    for tick, status in zip(signal_ax.get_xticklabels(), statuses, strict=True):
        if status == "saturated":
            tick.set_color("tab:red")
            tick.set_fontweight("bold")
        elif status == "near_saturation":
            tick.set_color("tab:orange")

    if "signal_fraction_of_full_well" in table.colnames:
        well_fractions = np.asarray(table["signal_fraction_of_full_well"], dtype=float)
    elif "full_well_e" in table.colnames:
        full_well = np.asarray(table["full_well_e"], dtype=float)
        well_fractions = np.full(len(additive_signal), np.nan)
        valid = np.isfinite(full_well) & (full_well > 0)
        well_fractions[valid] = additive_signal[valid] / full_well[valid]
    else:
        well_fractions = np.full(len(additive_signal), np.nan)

    saturated = [
        channel for channel, status in zip(channels, statuses, strict=True)
        if status == "saturated"
    ]
    near_saturated = [
        channel for channel, status in zip(channels, statuses, strict=True)
        if status == "near_saturation"
    ]
    if saturated:
        saturated_fractions = np.asarray([
            fraction for fraction, status
            in zip(well_fractions, statuses, strict=True)
            if status == "saturated"
        ])
        finite_saturated = saturated_fractions[np.isfinite(saturated_fractions)]
        max_fraction = (
            np.nanmax(finite_saturated) if finite_saturated.size else np.nan
        )
        signal_ax.text(
            0.02, 0.98,
            f"SATURATED: {', '.join(saturated)} exceed full well "
            f"(max {max_fraction:.2g}x)",
            transform=signal_ax.transAxes, va="top", ha="left",
            color="tab:red", fontsize="small", fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "tab:red", "alpha": 0.85},
        )
    elif near_saturated:
        signal_ax.text(
            0.02, 0.98,
            f"Near full well: {', '.join(near_saturated)}",
            transform=signal_ax.transAxes, va="top", ha="left",
            color="tab:orange", fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "tab:orange", "alpha": 0.85},
        )
    signal_ax.legend(frameon=False, fontsize="small")

    width = 0.2
    noise_terms = [
        ("diffuse shot", "diffuse_shot_noise_e_rms", "tab:blue"),
        ("dark shot", "dark_shot_noise_e_rms", "tab:green"),
        ("read", "read_noise_e_rms", "tab:orange"),
        ("total", "total_noise_e_rms", "black"),
    ]
    offsets = (np.arange(len(noise_terms)) - 1.5) * width
    for offset, (label, column, color) in zip(offsets, noise_terms, strict=True):
        noise_ax.bar(
            x + offset, np.asarray(table[column], dtype=float),
            width=width, color=color, label=label,
        )
    noise_ax.set_yscale("symlog", linthresh=1.0)
    noise_ax.set_xticks(x, channels)
    noise_ax.set_ylabel("Noise [e- RMS/pix]")
    noise_ax.set_title("Noise Terms")
    noise_ax.grid(axis="y", alpha=0.2)
    noise_ax.legend(frameon=False, fontsize="small")
    return fig, axes


def plot_transmission_sanity(data: Mapping[str, Any]):
    """Plot channel/order throughput sanity-check data."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
        constrained_layout=True,
    )

    group_colors = {
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "other": "0.5",
    }

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["optics_groups"].items():
            ax.plot(
                wave, values, lw=1.0,
                color=group_colors.get(name, "0.5"),
                label=name,
            )

        ax.plot(
            wave, channel["dichroic_total"], lw=1.2,
            color="tab:purple", label="dichroics",
        )
        ax.plot(
            wave, channel["detector_qe"], lw=1.8, ls=":",
            color="tab:red", alpha=0.9,
            label=channel.get("detector_qe_label", "detector QE"),
        )
        midpoint_qe = channel.get("detector_qe_midpoint")
        if midpoint_qe is not None:
            ax.plot(
                wave, midpoint_qe, lw=1.4, ls="--",
                color="tab:red", alpha=0.75,
                label=channel.get(
                    "detector_qe_midpoint_label",
                    "detector QE midpoint",
                ),
            )

        for idx, (_trace_id, order) in enumerate(channel["orders"].items()):
            order_label = "disperser/order" if idx == 0 else None
            order_qe_label = "QE at order trace" if idx == 0 else None
            total_label = "total/order" if idx == 0 else None
            ax.plot(
                wave, order["disperser"], lw=0.7, color="tab:orange",
                alpha=0.35, label=order_label,
            )
            ax.plot(
                wave, order["detector_qe"], lw=1.0, color="tab:pink",
                alpha=0.35, label=order_qe_label,
            )
            ax.plot(
                wave, order["total"], lw=1.8, color="black",
                alpha=0.65, label=total_label,
            )

        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=7, frameon=False,
    )
    return fig, axes


def plot_slit_pair_geometry(
    table: Table,
    *,
    slit_width: u.Quantity = 0.7 * u.arcsec,
    slit_length: u.Quantity = 10 * u.arcsec,
):
    """Plot a two-point source geometry relative to a rectangular slit."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x = u.Quantity(table["x"]).to_value(u.arcsec)
    y = u.Quantity(table["y"]).to_value(u.arcsec)
    width = u.Quantity(slit_width).to_value(u.arcsec)
    length = u.Quantity(slit_length).to_value(u.arcsec)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.add_patch(Rectangle(
        (-0.5 * length, -0.5 * width),
        length,
        width,
        fill=False,
        lw=1.6,
        color="black",
        label="slit",
    ))
    ax.scatter(x, y, s=50, color="tab:red", zorder=3, label="sources")
    for idx, (xpos, ypos) in enumerate(zip(x, y, strict=True)):
        ax.annotate(str(idx), (xpos, ypos), xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    pad = max(width, np.ptp(y) if y.size > 1 else width, 0.25) * 0.7
    ax.set_xlim(min(-0.5 * length, x.min()) - pad,
                max(0.5 * length, x.max()) + pad)
    ax.set_ylim(min(-width, y.min()) - pad, max(width, y.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Along slit [arcsec]")
    ax.set_ylabel("Across slit [arcsec]")
    ax.legend(frameon=False)
    return fig, ax


def plot_slit_adc_psf_scenes(data: Mapping[str, Any]):
    """Plot PSF-convolved slit scenes with AD-only and ADC-residual shifts."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scenarios = data["scenarios"]
    variants = data["variants"]
    x = u.Quantity(data["x_arcsec"]).to_value(u.arcsec)
    y = u.Quantity(data["y_arcsec"]).to_value(u.arcsec)
    extent = [x.min(), x.max(), y.min(), y.max()]
    width = u.Quantity(data["slit_width_arcsec"]).to_value(u.arcsec)
    length = u.Quantity(data["slit_length_arcsec"]).to_value(u.arcsec)

    nrows = len(scenarios)
    ncols = len(variants)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 4.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (scenario_name, scenario) in enumerate(scenarios.items()):
        table = scenario["positions"]
        xpos = u.Quantity(table["x"]).to_value(u.arcsec)
        ypos = u.Quantity(table["y"]).to_value(u.arcsec)
        for col, (variant_name, variant) in enumerate(variants.items()):
            ax = axes[row, col]
            image = scenario["images"][variant_name]
            finite = image[np.isfinite(image)]
            if finite.size:
                vmax = np.nanpercentile(finite, 99.6)
                vmin = np.nanpercentile(finite, 5.0)
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.add_patch(Rectangle(
                (-0.5 * length, -0.5 * width),
                length,
                width,
                fill=False,
                lw=2.4,
                edgecolor="white",
                label="slit" if row == 0 and col == 0 else None,
            ))
            ax.scatter(
                xpos,
                ypos,
                s=54,
                marker="o",
                facecolors="none",
                edgecolors="#7FDBFF",
                linewidths=1.8,
                label="source centers" if row == 0 and col == 0 else None,
            )
            ax.axvline(0, color="white", lw=0.8, alpha=0.45)
            ax.axhline(0, color="white", lw=0.8, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{scenario_name}\n{variant['label']}", fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel("Along slit [arcsec]")
            if col == 0:
                ax.set_ylabel("Across slit [arcsec]")
            if col == ncols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
                cbar.set_label("Relative PSF intensity")

    fig.suptitle(
        "Slit/AD/PSF Scene Check "
        f"(airmass {data['airmass']:.2f}, "
        f"seeing {data['seeing_arcsec'].to_value(u.arcsec):.2f} arcsec)",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="outside lower center", ncol=len(handles),
            frameon=False,
        )
    return fig, axes


def plot_slit_loss_by_arm(data: Mapping[str, Any]):
    """Plot centered point-source slit loss for each spectrograph arm."""
    import matplotlib.pyplot as plt

    arms = data["arms"]
    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(6.4 * len(arms), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (arm_name, arm) in zip(axes.flat, arms.items(), strict=True):
        wave = u.Quantity(arm["wave_nm"]).to_value(u.nm)
        for curve_name, curve in arm["curves"].items():
            ax.plot(
                wave,
                curve["loss"],
                lw=2.8,
                color=curve.get("color"),
                ls=curve.get("linestyle", "-"),
                label=curve["label"],
            )
        ax.text(
            0.02,
            0.96,
            f"slit {arm['slit_width_arcsec'].to_value(u.arcsec):.2f} arcsec\n"
            f"seeing {data['seeing_arcsec'].to_value(u.arcsec):.2f} arcsec",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "0.75",
                "alpha": 0.78,
            },
        )
        ax.set_title(
            f"{arm_name} Slit Loss",
            pad=8,
        )
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Slit loss fraction")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center",
        ncol=min(3, max(1, len(handles))),
        frameon=False,
    )
    return fig, axes


def plot_slit_width_loss(data: Mapping[str, Any]):
    """Plot centered point-source slit loss as a function of slit width."""
    import matplotlib.pyplot as plt

    arms = data["arms"]
    if len(arms) != 2:
        raise ValueError(
            "plot_slit_width_loss expects the matched two-arm data produced by "
            "build_slit_width_loss_data."
        )
    arm_names = list(arms)
    arm_waves = {
        name: u.Quantity(arms[name]["wavelengths_nm"]).to_value(u.nm)
        for name in arm_names
    }
    n_waves = len(arm_waves[arm_names[0]])
    if any(len(values) != n_waves for values in arm_waves.values()):
        raise ValueError("Matched slit-width loss arms have different wavelengths.")

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, n_waves))
    curve_labels: dict[tuple[str, str, float], str] = {}
    curve_colors: dict[tuple[str, float], Any] = {}
    psf_modes = data["psf_modes"]
    for color, wave_pair in zip(
        colors,
        zip(*(arm_waves[name] for name in arm_names), strict=True),
        strict=True,
    ):
        wave_text = "/".join(f"{wave_nm:.0f}" for wave_nm in wave_pair)
        for arm_name, wave_nm in zip(arm_names, wave_pair, strict=True):
            wave_key = round(float(wave_nm), 9)
            curve_colors[(arm_name, wave_key)] = color
            for psf_mode, psf_spec in psf_modes.items():
                curve_labels[(arm_name, psf_mode, wave_key)] = (
                    f"{psf_spec['label']}, {wave_text} nm"
                )

    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(6.4 * len(arms), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (arm_name, arm) in zip(axes.flat, arms.items(), strict=True):
        slit_widths = u.Quantity(arm["slit_widths_arcsec"]).to_value(u.arcsec)
        for curve in arm["curves"].values():
            wave_key = round(float(curve["wavelength_nm"]), 9)
            color = curve_colors[(arm_name, wave_key)]
            ax.plot(
                slit_widths,
                curve["loss"],
                lw=2.4,
                color=color,
                ls=curve.get("linestyle", "-"),
                label=curve_labels[(arm_name, curve["psf_mode"], wave_key)],
            )
            selector_slits = u.Quantity(
                arm.get("selector_slit_widths_arcsec", []),
            ).to_value(u.arcsec)
            selector_slits = selector_slits[
                (selector_slits >= slit_widths.min())
                & (selector_slits <= slit_widths.max())
            ]
            if selector_slits.size:
                ax.scatter(
                    selector_slits,
                    np.interp(selector_slits, slit_widths, curve["loss"]),
                    s=34,
                    marker="o",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=4,
                    label="available slit widths",
                )

        current_slit = u.Quantity(
            arm["current_slit_width_arcsec"],
        ).to_value(u.arcsec)
        if np.isfinite(current_slit):
            ax.axvline(
                current_slit, color="0.25", lw=1.6, ls=":",
                label="current slit",
            )
        ax.set_title(f"{arm_name} PSF Loss vs Slit Width", pad=8)
        ax.set_xlabel("Slit width [arcsec]")
        ax.set_ylabel("Slit loss fraction")
        ax.set_ylim(0, 1)
        ax.set_xlim(slit_widths.min(), slit_widths.max())
        ax.grid(alpha=0.25)

    legend_entries: OrderedDict[str, Any] = OrderedDict()
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_entries.setdefault(label, handle)
    fig.legend(
        list(legend_entries.values()), list(legend_entries),
        loc="outside lower center",
        ncol=min(4, max(1, len(legend_entries))),
        frameon=False,
        fontsize="small",
    )
    return fig, axes


def _readout_image_data(channel_hdul: Any) -> np.ndarray:
    image_hdu = (
        channel_hdul
        if hasattr(channel_hdul, "data")
        else channel_hdul[1]
    )
    return np.asarray(image_hdu.data, dtype=float)


def _clip_fraction_label(clip: float) -> str:
    return f"p{100.0 * clip:.4g}"


def _readout_display_limits(
    data: np.ndarray,
    clip: float | None,
    *,
    symmetric: bool,
) -> tuple[float, float, str]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0, "no finite data") if symmetric else (0.0, 1.0, "no finite data")

    if clip is None:
        if symmetric:
            limit = float(np.nanmax(np.abs(finite)))
            if not np.isfinite(limit) or limit <= 0:
                limit = 1.0
            return -limit, limit, "unclipped"
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax, "unclipped"

    clip = float(clip)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be None or a positive value.")

    if clip <= 1:
        if symmetric:
            limit = float(np.nanquantile(np.abs(finite), clip))
            label = f"{_clip_fraction_label(clip)} |value|"
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanquantile(finite, clip))
            label = _clip_fraction_label(clip)
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            return vmin, vmax, label
    else:
        limit = clip
        label = f"{clip:.4g}"
        if not symmetric:
            vmin = min(float(np.nanmin(finite)), 0.0)
            vmax = clip
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            return vmin, vmax, label

    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit, label


def _readout_grid_axes(n_images: int):
    import matplotlib.pyplot as plt

    ncols = min(3, max(1, n_images))
    nrows = int(np.ceil(n_images / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def _plot_readout_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    *,
    clip: float | None,
    symmetric: bool,
    cmap: str,
    title_suffix: str = "",
    annotate_max_abs: bool = False,
):
    fig, axes = _readout_grid_axes(len(images))
    for ax, title, data in zip(axes.flat, titles, images, strict=False):
        vmin, vmax, clip_label = _readout_display_limits(
            data, clip, symmetric=symmetric,
        )
        im = ax.imshow(
            data,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(f"{title}{title_suffix}", pad=8)
        if annotate_max_abs:
            finite = data[np.isfinite(data)]
            max_abs = np.nanmax(np.abs(finite)) if finite.size else 0.0
            ax.text(
                0.02,
                0.96,
                f"max |delta| {max_abs:.3g}\nclip {clip_label}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": "0.75",
                    "alpha": 0.78,
                },
            )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(f"clip {clip_label}")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    return fig, axes


def plot_readout_overview(
    hdul: Any,
    titles: list[str] | None = None,
    *,
    clip: float | None = 0.995,
):
    """Plot detector readout images from a ScopeSim readout result."""
    readouts = list(hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(readouts))]
    images = [_readout_image_data(channel_hdul) for channel_hdul in readouts]
    return _plot_readout_image_grid(
        images,
        titles,
        clip=clip,
        symmetric=False,
        cmap="cividis",
    )


def plot_readout_delta_overview(
    signal_hdul: Any,
    reference_hdul: Any,
    titles: list[str] | None = None,
    *,
    clip: float | None = 0.995,
):
    """Plot source-minus-reference detector readout images."""
    signal_readouts = list(signal_hdul)
    reference_readouts = list(reference_hdul)
    if len(signal_readouts) != len(reference_readouts):
        raise ValueError(
            "signal_hdul and reference_hdul contain different readout counts: "
            f"{len(signal_readouts)} != {len(reference_readouts)}"
        )
    titles = titles or [f"detector {idx}" for idx in range(len(signal_readouts))]
    images = [
        _readout_image_data(signal) - _readout_image_data(reference)
        for signal, reference in zip(signal_readouts, reference_readouts, strict=True)
    ]
    return _plot_readout_image_grid(
        images,
        titles,
        clip=clip,
        symmetric=True,
        cmap="coolwarm",
        title_suffix=" Source - Empty",
        annotate_max_abs=True,
    )


def plot_readout_cross_dispersion_cut(
    hdul: Any,
    titles: list[str] | None = None,
    *,
    central_columns: int = 50,
):
    """Plot row profiles from the median of central detector columns."""
    import matplotlib.pyplot as plt

    readouts = list(hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(readouts))]
    ncols = min(3, max(1, len(readouts)))
    nrows = int(np.ceil(len(readouts) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, title, channel_hdul in zip(axes.flat, titles, readouts, strict=False):
        data = _readout_image_data(channel_hdul)
        nx = data.shape[1]
        ncut = max(1, min(int(central_columns), 50, nx))
        x0 = nx // 2 - ncut // 2
        x1 = x0 + ncut
        profile = np.nanmedian(data[:, x0:x1], axis=1)
        ax.plot(np.arange(profile.size), profile, lw=1.8, color="tab:blue")
        ax.set_title(f"{title} central {ncut} cols", pad=8)
        ax.set_xlabel("Detector row [pix]")
        ax.set_ylabel("Median counts")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(readouts):]:
        ax.axis("off")
    return fig, axes
