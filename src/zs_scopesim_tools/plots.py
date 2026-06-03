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


def plot_source(source: Any, wave: u.Quantity | None = None):
    """Plot each source field's spatial profile and spectrum."""
    import matplotlib.pyplot as plt

    wave = wave if wave is not None else np.linspace(0.3, 2.5, 1001) * u.um
    num_fields = len(source.fields)
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
        ax_image.imshow(field.data, origin="lower", cmap="viridis")
        ax_spectrum.plot(wave, field.spectrum(wave))
        if idx == 0:
            ax_image.set_title("Spatial Profile")
            ax_spectrum.set_title("Spectrum")
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
        2, 3, figsize=(16, 7.5), sharex=True, constrained_layout=True,
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
        if channel["total_spectrum"] is not None:
            ax.plot(
                wave,
                _plot_quantity_values(channel["total_spectrum"]),
                lw=1.5,
                color="black",
                alpha=0.75,
                label="total diffuse",
            )
        ax.set_title(
            f"{channel['label']} (image plane {channel['image_plane_id']}): "
            f"{channel['total_rate_ph_s_pix']:.3g} ph/s/pix",
        )
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spectral background [PHOTLAM equiv.]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    dedup = OrderedDict(zip(labels, handles))
    fig.suptitle(
        "Post-disperser diffuse emission is added after dichroic/echelle "
        "trace mapping; pre-disperser thermal light remains dichroic-filtered.",
        fontsize=11,
    )
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside upper center",
        ncol=4, frameon=False,
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
               channel["post_disperser_after_qe"]]
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
            wave, channel["post_disperser_after_qe"], lw=2.4,
            color="black", alpha=0.95,
            label="post diffuse after downstream+QE", zorder=10,
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

    handles, labels = axes.flat[0].get_legend_handles_labels()
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
    signal_ax.bar(x, diffuse, width=0.7, label="post-disperser diffuse")
    signal_ax.bar(x, dark, width=0.7, bottom=diffuse, label="dark current")
    signal_ax.plot(x, bias, "o", color="black", label="bias offset")
    signal_ax.set_yscale("symlog", linthresh=1.0)
    signal_ax.set_xticks(x, channels)
    signal_ax.set_ylabel("Detector signal [e-/pix]")
    signal_ax.set_title("Additive Signal")
    signal_ax.grid(axis="y", alpha=0.2)
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
        (-0.5 * width, -0.5 * length),
        width,
        length,
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
    pad = max(width, np.ptp(x) if x.size > 1 else width, 0.25) * 0.7
    ax.set_xlim(min(-width, x.min()) - pad, max(width, x.max()) + pad)
    ax.set_ylim(min(-0.5 * length, y.min()) - pad,
                max(0.5 * length, y.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Across slit [arcsec]")
    ax.set_ylabel("Along slit [arcsec]")
    ax.legend(frameon=False)
    return fig, ax
