"""Diagnostic figures for the evidence-driven Task 1 journey."""
from __future__ import annotations

import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import design_data as data


COLOURS = {
    "observed": "#007c91", "truth": "#202124", "Shared ridge": "#e07a5f",
    "Sensor-specific ridge": "#3d5a80", "Period-routed ridge": "#43aa8b",
    "Raw Attention": "#e09f3e", "Raw delay mixer": "#7b2cbf",
    "Attention + decomposition": "#577590", "Autoformer-inspired": "#d1495b",
}


def _colour(name):
    return COLOURS.get(name, "#777777")


def _product_name(study, part, window):
    return study.world.product_names[int(part["product"][window])]


def _time(context=True):
    samples = np.arange(-data.CONTEXT, 0) if context else np.arange(data.HORIZON)
    return data.seconds(samples)


def _plot_forecast(axis, context, target, forecasts, *, title):
    axis.plot(_time(True), context, color=COLOURS["observed"], lw=1.35, label="observed context")
    axis.plot(_time(False), target, color=COLOURS["truth"], lw=2.1, label="ground truth")
    for name, forecast in forecasts.items():
        axis.plot(_time(False), forecast, ls="--", lw=1.35, color=_colour(name), label=name)
    axis.axvline(0, color="#555", ls=":", lw=1, label="forecast boundary")
    axis.set(title=title, ylabel="vibration acceleration (m/s²)")


def _two_regime_cases(study, region="validation"):
    part = study.region(region)
    products = part["product"]
    cases = []
    for product in (0, 2):
        positions = np.flatnonzero(products == product)
        if not len(positions):
            raise ValueError(f"{region} needs eligible windows from {study.world.product_names[product]}")
        cases.append(int(positions[len(positions) // 2]))
    return 0, tuple(cases)  # Station 1 is established and illustrates two product paces.


def regime_forecasts(study, names, region="validation"):
    """One station under Product A and C: its batch pace, rather than identity, changes."""
    part, predictions = study.region(region), study.predictions(region)
    sensor, cases = _two_regime_cases(study, region)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), layout="constrained", sharey=True)
    rows = []
    for axis, window in zip(axes, cases):
        product = _product_name(study, part, window)
        period = part["primary_period"][sensor, window]
        forecasts = {name: predictions[name][sensor, window] for name in names}
        _plot_forecast(axis, part["context"][sensor, window], part["target"][sensor, window], forecasts,
                       title=(f"Station {sensor + 1}: {product} "
                              f"(actual period {period:.1f} samples = "
                              f"{period / data.SAMPLING_RATE_HZ:.2f} s)"))
        for name, forecast in forecasts.items():
            rows.append({"station": sensor + 1, "product": product,
                         "actual period (samples)": period, "model": name,
                         "window RMSE (m/s²)": data.rmse(forecast, part["target"][sensor, window])})
    axes[-1].set_xlabel("time relative to forecast origin (s)")
    axes[0].legend(ncol=3, fontsize=8)
    return pd.DataFrame(rows)


def attention_behaviour(study, region="validation", name="Raw Attention"):
    """Two Attention maps for the same station at two product paces, plus their difference.

    This supports only the Part 1 claim that fixed W_Q, W_K, W_V still build a different
    per-context mixing operation. It does not claim Attention has recovered the physical
    period -- the raw model has not been asked to, and a lag-collapsed profile computed this
    way is dominated by drift and, at extreme displacements, by averaging over very few
    diagonal entries, so it is not a reliable period estimate.
    """
    import design_controls as controls
    part = study.region(region)
    sensor, cases = _two_regime_cases(study, region)
    model = study._models[name]
    batch = np.stack([part["x"][sensor, window] for window in cases])[..., None].astype(np.float32)
    with torch.no_grad():
        model.eval()(torch.as_tensor(batch, device=controls.device))
    attention = model.attention_layers[0].last_attention.cpu().numpy().mean(1)
    products = [_product_name(study, part, window) for window in cases]
    periods = [float(part["primary_period"][sensor, window]) for window in cases]
    difference = attention[0] - attention[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    shared_kwargs = dict(vmin=float(attention.min()), vmax=float(attention.max()))
    for axis, matrix, product in zip(axes[:2], attention, products):
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="magma", **shared_kwargs)
        axis.set(xlabel="key position in 96-sample context",
                 ylabel="query position in 96-sample context",
                 title=f"Station {sensor + 1}, {product}: Attention map")
        fig.colorbar(image, ax=axis, fraction=.046, pad=.02, label="mean head weight")
    bound = float(np.abs(difference).max())
    image = axes[2].imshow(difference, aspect="auto", origin="lower", cmap="coolwarm",
                           vmin=-bound, vmax=bound)
    axes[2].set(xlabel="key position in 96-sample context",
               ylabel="query position in 96-sample context",
               title=f"{products[0]} minus {products[1]}:\nsame $W_Q,W_K,W_V$, different mixing weights")
    fig.colorbar(image, ax=axes[2], fraction=.046, pad=.02, label="weight difference")

    rows = [{"station": sensor + 1, "product": product, "actual period (samples)": period}
           for product, period in zip(products, periods)]
    rows.append({"station": sensor + 1, "product": f"{products[0]} vs {products[1]} (difference)",
                "actual period (samples)": np.nan,
                "mean |Attention weight difference|": float(np.abs(difference).mean()),
                "max |Attention weight difference|": bound})
    return pd.DataFrame(rows)


def filter_decomposition(study, region="validation"):
    """Show how several candidate widths split the same context into trend and residual."""
    part, kernel = study.region(region), study.decomposition_kernel
    widths = list(study.kernel_selection["width"])
    raw_estimate = data.estimate_period(part["context"], detrend_kernel=0)
    gap = np.abs(raw_estimate - part["primary_period"])
    eligible = study.mask("established", region)
    sensor, window = np.unravel_index(np.argmax(np.where(eligible, gap, -np.inf)), gap.shape)
    context = part["context"][sensor, window]
    product = _product_name(study, part, window)
    true_period = part["primary_period"][sensor, window]

    trends = {width: data.moving_average(context, width) for width in widths}
    residuals = {width: context - trends[width] for width in widths}
    _, raw_scores = data.dense_period_scores(context, detrend_kernel=0)
    grid, clean_scores = data.dense_period_scores(context, detrend_kernel=kernel)

    shades = np.linspace(0.35, 0.95, len(widths))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
    axes[0].plot(_time(True), context, color=COLOURS["observed"], lw=1.1, label="raw context")
    for width, shade in zip(widths, shades):
        selected = width == kernel
        axes[0].plot(_time(True), trends[width], color="#577590", alpha=shade,
                     lw=2.2 if selected else 1.1, ls="-" if selected else "--",
                     label=f"width {width}" + (" (selected)" if selected else ""))
    axes[0].axvline(0, color="#555", ls=":", lw=.8)
    axes[0].set(xlabel="time relative to forecast origin (s)", ylabel="vibration acceleration (m/s²)",
                title=f"Station {sensor + 1}, {product}: candidate trend widths")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[1].axhline(0, color="#999", lw=.8)
    for width, shade in zip(widths, shades):
        selected = width == kernel
        axes[1].plot(_time(True), residuals[width], color="#43aa8b", alpha=shade,
                     lw=2.2 if selected else 1.1, ls="-" if selected else "--",
                     label=f"width {width}" + (" (selected)" if selected else ""))
    axes[1].axvline(0, color="#555", ls=":", lw=.8)
    axes[1].set(xlabel="time relative to forecast origin (s)", ylabel="residual acceleration (m/s²)",
                title="Residual shape changes with the averaging width")
    axes[1].legend(fontsize=6.5, ncol=2)
    axes[2].plot(grid, raw_scores, color="#999", label="raw context")
    axes[2].plot(grid, clean_scores, color="#43aa8b", label=f"after width-{kernel} decomposition")
    axes[2].axvline(true_period, color="#222", ls=":",
                    label=f"actual period: {true_period:.1f}")
    axes[2].set(xlabel="candidate period (samples)", ylabel="normalized projection score",
                title="Decomposition sharpens the product pace")
    axes[2].legend(fontsize=7)

    rows = [{"width": width, "selected": width == kernel,
             "residual std (m/s²)": float(np.std(residuals[width])),
             "trend range (m/s²)": float(np.ptp(trends[width]))}
            for width in widths]
    return pd.DataFrame(rows)


def recurrence_recovery(study, region="train"):
    """Oracle diagnostic: compare period estimates with synthetic-world truth only."""
    part = study.region(region)
    keep = study.mask("established", region)
    rows = []
    settings = [("raw", 0)] + [(f"width {width}", width)
                                for width in study.kernel_selection["width"]]
    for label, width in settings:
        estimate = data.estimate_period(part["context"], detrend_kernel=width)
        error = np.abs(estimate - part["primary_period"])
        rows.append({"representation": label,
                     "oracle recovery within one sample": float((error[keep] <= 1).mean()),
                     "median absolute period error (samples)": float(np.median(error[keep])),
                     "selected": width == study.decomposition_kernel})
    table = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    axis.bar(table["representation"], table["oracle recovery within one sample"],
             color=["#d1495b" if selected else "#9db4c0" for selected in table.selected])
    axis.set(ylabel="oracle recovery within one sample", ylim=(0, 1),
             title="Oracle diagnostic: compare with the synthetic true period")
    axis.tick_params(axis="x", rotation=20)
    return table


def decomposition_ablation(study, region="validation"):
    names = ["Raw Attention", "Attention + decomposition"]
    part = study.region(region)
    predictions = {name: study.neural_prediction(name, region) for name in names}
    table = study.grouped_table(names, region)

    raw_error = np.mean((predictions["Raw Attention"] - part["target"]) ** 2, axis=-1)
    decomposed_error = np.mean(
        (predictions["Attention + decomposition"] - part["target"]) ** 2, axis=-1)
    improvement = raw_error - decomposed_error
    cases = [("largest decomposition gain", improvement),
             ("smallest decomposition gain", -improvement)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), layout="constrained")
    for axis, (label, criterion) in zip(axes, cases):
        sensor, window = np.unravel_index(np.argmax(criterion), criterion.shape)
        product = _product_name(study, part, window)
        _plot_forecast(axis, part["context"][sensor, window], part["target"][sensor, window],
                       {name: prediction[sensor, window] for name, prediction in predictions.items()},
                       title=f"{label}: Station {sensor + 1}, {product}")
    axes[-1].set_xlabel("time relative to forecast origin (s)")
    axes[0].legend(ncol=4, fontsize=8)
    return table


def delay_recurrence(study, region="validation"):
    """Signal-level diagnostic; it is distinct from the model's learned q/k scores."""
    part, kernel = study.region(region), study.decomposition_kernel
    candidates = study.mask("established", region) & study.mask("Product A", region)
    # Product A gives room to show P, 2P, and usually 3P in the permitted 8--48 delay band.
    centre = np.mean(study.world.product_period_bands[0])
    criterion = np.where(candidates,
                         -np.abs(part["primary_period"] - centre)
                         + 0.05 * part["slow_change"] / part["slow_change"].max(), -np.inf)
    sensor, window = np.unravel_index(np.argmax(criterion), candidates.shape)
    context = part["context"][sensor, window]
    trend = data.moving_average(context, kernel)
    residual = context - trend

    def autocorrelation(signal):
        centred = signal - signal.mean()
        denominator = np.dot(centred, centred)
        return np.array([np.dot(centred, np.roll(centred, delay)) / denominator
                         for delay in range(49)])

    raw, clean = autocorrelation(context), autocorrelation(residual)
    period = float(part["primary_period"][sensor, window])
    expected = period * np.arange(1, int(48 // period) + 1)
    marked = np.rint(expected).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
    axes[0].plot(_time(True), context, color=COLOURS["observed"], lw=1.15, label="raw context")
    axes[0].plot(_time(True), trend, color="#577590", lw=1.8, label="slow trend")
    axes[0].set(xlabel="time relative to forecast origin (s)", ylabel="vibration acceleration (m/s²)",
                title="Raw context mixes drift and vibration")
    axes[0].legend(fontsize=7)
    axes[1].plot(_time(True), residual, color="#43aa8b", lw=1.2, label="decomposed residual")
    axes[1].axhline(0, color="#999", lw=.8)
    axes[1].set(xlabel="time relative to forecast origin (s)", ylabel="residual acceleration (m/s²)",
                title=f"Product A residual repeats every about {period:.1f} samples")
    axes[1].legend(fontsize=7)
    delay = np.arange(8, 49)
    axes[2].plot(delay, raw[delay], color="#999", label="raw signal correlation")
    axes[2].plot(delay, clean[delay], color="#43aa8b", lw=1.7,
                 label="residual signal correlation")
    for multiple, (ideal, delay_mark) in enumerate(zip(expected, marked), 1):
        axes[2].axvline(delay_mark, color="#222", ls=":", lw=.8)
        axes[2].annotate(f"{multiple}P≈{ideal:.1f}", (delay_mark, .97), xytext=(2, 0),
                         textcoords="offset points", fontsize=8)
    axes[2].set(xlabel="candidate delay (samples)", ylabel="normalized circular correlation",
                title="Signal-level recurrence diagnostic (not learned q/k scores)")
    axes[2].legend(fontsize=7)
    return pd.DataFrame({"station": [sensor + 1] * len(expected),
                         "product": ["Product A"] * len(expected),
                         "expected delay (samples)": expected,
                         "nearest integer delay": marked,
                         "residual signal correlation": clean[marked]})


def timing_figure(lengths=(48, 96, 192, 384, 768), d=64, heads=4, batch=2,
                  repeats=8, device="cpu"):
    """Time only the already-projected mixing operations."""
    import design_models as models
    selected_device = torch.device(device)

    def synchronize():
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

    score_fn = models._require(models._DELAY_SCORES, "delay_scores")
    aggregate = models._require(models._AGGREGATE, "aggregate_delays")
    features, rows = d // heads, []
    for length in lengths:
        q = torch.randn(batch, heads, features, length, device=selected_device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        attention_q, attention_k, attention_v = (x.permute(0, 1, 3, 2) for x in (q, k, v))

        def attend():
            weights = (attention_q @ attention_k.transpose(-1, -2) / math.sqrt(features)).softmax(-1)
            return weights @ attention_v

        def delay_mix():
            scores = score_fn(q, k)
            band = scores[:, 8:min(48, length - 1) + 1]
            count = min(math.ceil(2 * math.log(length)), band.shape[-1])
            selected, index = band.topk(count, -1)
            return aggregate(v, index + 8, selected.softmax(-1))

        with torch.no_grad():
            attend(); delay_mix()
            timings = {}
            for label, operation in (("Attention ms", attend), ("Delay mixer ms", delay_mix)):
                synchronize(); started = time.perf_counter()
                for _ in range(repeats):
                    operation()
                synchronize()
                timings[label] = 1000 * (time.perf_counter() - started) / repeats
        rows.append({"L": length, "device": str(selected_device),
                     "K": min(math.ceil(2 * math.log(length)), max(0, min(48, length - 1) - 7)),
                     **timings})
    table = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.5, 3.5), layout="constrained")
    for column, colour in (("Attention ms", _colour("Raw Attention")),
                           ("Delay mixer ms", _colour("Autoformer-inspired"))):
        axis.plot(table.L, table[column], marker="o", color=colour, label=column[:-3])
    axis.set(xscale="log", yscale="log", xlabel="sequence length L", ylabel="milliseconds",
             title="Measured mixing time (projections excluded)")
    axis.legend()
    return table


def contrasting_forecasts(study, region="validation"):
    """The largest combined-model gain and loss cases -- deliberately extreme, not representative."""
    names = ("Raw Attention", "Raw delay mixer", "Attention + decomposition", "Autoformer-inspired")
    part, predictions = study.region(region), study.predictions(region)
    target = part["target"]
    base = np.mean((predictions["Raw Attention"] - target) ** 2, axis=-1)
    auto = np.mean((predictions["Autoformer-inspired"] - target) ** 2, axis=-1)
    candidates = study.mask("all", region)
    cases = [("largest combined-model gain", np.where(candidates, base - auto, -np.inf)),
             ("largest combined-model loss", np.where(candidates, auto - base, -np.inf))]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), layout="constrained")
    rows = []
    for axis, (case, criterion) in zip(axes, cases):
        sensor, window = np.unravel_index(np.argmax(criterion), criterion.shape)
        product = _product_name(study, part, window)
        forecasts = {name: predictions[name][sensor, window] for name in names}
        _plot_forecast(axis, part["context"][sensor, window], target[sensor, window], forecasts,
                       title=f"{case}: Station {sensor + 1}, {product}")
        for name, forecast in forecasts.items():
            rows.append({"case": case, "station": sensor + 1, "product": product, "model": name,
                         "window RMSE (m/s²)": data.rmse(forecast, target[sensor, window])})
    axes[-1].set_xlabel("time relative to forecast origin (s)")
    axes[0].legend(ncol=3, fontsize=7)
    return pd.DataFrame(rows)


def horizon_figure(study, names, region="validation"):
    table = study.horizon_errors(names, region)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained", sharey=True)
    labels = {"established": "Stations 1--3", "held-out": "Held-out station 4"}
    for axis, population in zip(axes, ("established", "held-out")):
        for name, rows in table[table.population == population].groupby("model"):
            axis.plot(rows.horizon, rows["RMSE (m/s²)"], color=_colour(name), label=name)
        axis.set(xlabel="forecast horizon (samples)", ylabel="RMSE (m/s²)",
                 title=labels[population])
    axes[0].legend(fontsize=7)
    return table


def _causal_pulse(time, width=3.0):
    output = np.zeros_like(time, dtype=float)
    active = time >= 0
    scaled = time[active] / width
    output[active] = scaled * np.exp(1.0 - scaled)
    return output


def spatial_teaser(study):
    """A station-3 warning makes two identical station-4 contexts distinguishable."""
    part = study.region("validation")
    window = int(np.flatnonzero(part["product"] == 1)[len(np.flatnonzero(part["product"] == 1)) // 2])
    focal, upstream = 3, 2  # zero-based: held-out station 4 and physical station 3.
    context = part["context"][focal, window]
    future_without = part["target"][focal, window].copy()
    transit_delay, source_start, amplitude = 12, -8, 1.35
    upstream_time = np.arange(-data.CONTEXT, 0)
    source_warning = amplitude * _causal_pulse(upstream_time - source_start)
    arrival = source_start + transit_delay
    future_difference = amplitude * _causal_pulse(np.arange(data.HORIZON) - arrival)
    future_with = future_without + future_difference
    lower_bound = future_difference ** 2 / 4

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
    axis = axes[0]
    station_x = np.arange(1, 5)
    axis.scatter(station_x, np.zeros(4), s=800, color="#dceaf0", edgecolor="#007c91", zorder=2)
    for station in station_x:
        axis.text(station, 0, f"Station {station}", ha="center", va="center", fontsize=9)
    axis.annotate("", xy=(4 - .34, 0), xytext=(3 + .34, 0),
                  arrowprops=dict(arrowstyle="->", lw=2, color="#d1495b"))
    axis.text(3.5, .18, f"warning travels {transit_delay} samples\n({transit_delay / data.SAMPLING_RATE_HZ:.2f} s)",
              ha="center", color="#d1495b", fontsize=8)
    axis.set(xlim=(.35, 4.65), ylim=(-.45, .5), title="The four-station production line")
    axis.axis("off")
    axes[1].plot(_time(True), part["context"][upstream, window], color="#577590",
                 label="ordinary Station 3 context")
    axes[1].plot(_time(True), part["context"][upstream, window] + source_warning, color="#d1495b",
                 label="Station 3 records incoming disturbance")
    axes[1].axvline(0, color="#555", ls=":", lw=.8)
    axes[1].set(xlabel="time relative to forecast origin (s)", ylabel="vibration acceleration (m/s²)",
                title="Upstream context contains the warning")
    axes[1].legend(fontsize=7)
    axes[2].plot(_time(True), context, color=COLOURS["observed"], label="identical Station 4 context")
    axes[2].plot(_time(False), future_without, color="#577590", label="no disturbance in transit")
    axes[2].plot(_time(False), future_with, color="#d1495b", ls="--",
                 label="disturbance reaches Station 4")
    axes[2].axvline(0, color="#555", ls=":", lw=.8)
    axes[2].set(xlabel="time relative to forecast origin (s)", ylabel="vibration acceleration (m/s²)",
                title="Identical local context, different future")
    axes[2].legend(fontsize=7)
    return pd.DataFrame([{
        "relationship": "Station 3 to held-out station 4",
        "propagation delay (samples)": transit_delay,
        "propagation delay (s)": transit_delay / data.SAMPLING_RATE_HZ,
        "mean equal-probability MSE lower bound (m/s²)²": float(lower_bound.mean()),
        "maximum future separation (m/s²)": float(np.max(np.abs(future_difference))),
    }])
