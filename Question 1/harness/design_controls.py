"""Leakage-safe fitting, caching, and evaluation for the Design assignment."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import design_data as data
import design_models as models

device = torch.device(os.environ.get("PA1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.set_num_threads(int(os.environ.get("PA1_THREADS", "8")))
CHECKPOINTS = Path(os.environ.get(
    "PA1_CHECKPOINTS", Path(__file__).resolve().parent.parent / "checkpoints"))
register_components = models.register_components


@dataclass(frozen=True)
class Preset:
    stride_train: int
    stride_eval: int
    steps: int
    width: int
    layers: int
    batch: int = 128


PRESETS = {
    "smoke": Preset(32, 32, 20, 32, 1),
    "quick": Preset(8, 8, 1500, 64, 2),
    "full": Preset(4, 8, 6000, 64, 2),
}
RIDGE_CANDIDATES = (1e-3, 1e-2, 1e-1, 1.0)
DECOMPOSITION_CANDIDATES = (17, 25, 33, 41)
PERIOD_RANGE = data.PERIOD_SEARCH_RANGE
CACHE_FORMAT = 2
FIT_CONFIGURATION = {
    "optimizer": "AdamW", "learning_rate": 3e-3, "weight_decay": 1e-4,
    "scheduler": "OneCycleLR", "max_learning_rate": 3e-3, "pct_start": 0.15,
    "gradient_clip_norm": 5.0, "best_state_metric": "established validation MSE",
}
_LEGACY_SMOKE_DATA_BEHAVIOR = "8f6f362b03bd239dfaf8b53a1709e491d141711451a70357012da0a656b4ea43"
_LEGACY_SMOKE_MODEL_BEHAVIOR = {
    "Raw Attention": "5654bbdc06c4773727eace342d7b43e17334cda91564e4361356b72785ff7eed",
    "Attention + decomposition": "d56030270469f8a57ed358952772340edfa798d877fff28da0a346b1c29b6002",
    "Raw delay mixer": "1223c0eb2c12ecdc99268dc9b9618c3484d5347c795f670aa07fdcfb381deafe",
    "Autoformer-inspired": "8d4d081e3ca8e2b0951f4c58e02b43c53346ae13b91cf5331e0c0ff04a32ddf1",
}
_LEGACY_SMOKE_TRAINING = {
    "optimizer": "AdamW", "learning_rate": 3e-3, "weight_decay": 1e-4,
    "scheduler": "OneCycleLR", "max_learning_rate": 3e-3, "pct_start": 0.15,
    "gradient_clip_norm": 5.0, "best_state_metric": "established validation MSE",
    "preset": {"stride_train": 32, "stride_eval": 32, "steps": 20, "width": 32,
               "layers": 1, "batch": 128},
}


def _run_decomposition_case(cls, kernel, x, expected_trend, hint):
    x = x.clone().requires_grad_()
    length = x.shape[1]
    returned = cls(kernel)(x)
    assert isinstance(returned, tuple) and len(returned) == 2, (
        f"kernel={kernel}, length={length}: forward must return the pair (remainder, trend).")
    residual, trend = returned
    assert residual.shape == trend.shape == x.shape, (
        f"kernel={kernel}, length={length}: got remainder {list(residual.shape)} and trend "
        f"{list(trend.shape)}, expected both to be {list(x.shape)}. The average is taken "
        "along time (axis 1) and must not change the shape.")
    torch.testing.assert_close(trend, expected_trend, msg=lambda default: (
        f"kernel={kernel}, length={length}: the trend does not match. {hint}\n{default}"))
    torch.testing.assert_close(residual + trend, x, msg=lambda default: (
        f"kernel={kernel}, length={length}: remainder + trend must reconstruct the input "
        f"exactly.\n{default}\nReturn x - trend as the remainder rather than recomputing it, "
        "and make sure the pair is (remainder, trend), not (trend, remainder)."))
    (residual.square().sum() + trend.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0, (
        f"kernel={kernel}, length={length}: gradients did not reach the input. Every step from "
        "x to (remainder, trend) must stay a differentiable tensor op -- no .item(), .numpy(), "
        "or plain Python indexing that detaches the graph.")


def check_decomposition(cls):
    # A: kernel=1 collapses the window to a single point, so trend must equal x exactly.
    _run_decomposition_case(
        cls, 1,
        torch.tensor([[[1.], [2.], [3.], [4.]]], dtype=torch.float64),
        torch.tensor([[[1.], [2.], [3.], [4.]]], dtype=torch.float64),
        "With kernel=1 the averaging window is a single sample, so the trend must equal the "
        "input exactly -- if it doesn't, the window width (2*radius+1) is being built wrong.")

    # B: interior + both endpoints, kernel=3 (radius=1), single batch/feature.
    _run_decomposition_case(
        cls, 3,
        torch.tensor([[[0.], [10.], [20.], [30.], [40.], [50.], [60.]]], dtype=torch.float64),
        torch.tensor([[[10 / 3], [10.], [20.], [30.], [40.], [50.], [170 / 3]]], dtype=torch.float64),
        "Interior points (t=1..5) are a plain centered average of 3 neighbors. The two "
        "endpoints (t=0, t=6) must replicate the nearest in-range sample instead of reading "
        "past the edge or padding with zero -- e.g. t=0 averages x[0], x[0], x[1], not x[-1].")

    # C: heavier edge clipping, kernel=5 (radius=2), single batch/feature.
    _run_decomposition_case(
        cls, 5,
        torch.tensor([[[0.], [10.], [20.], [30.], [40.], [50.]]], dtype=torch.float64),
        torch.tensor([[[6.], [12.], [20.], [30.], [38.], [44.]]], dtype=torch.float64),
        "With radius=2, t=0 and t=1 each need multiple out-of-range window positions clamped "
        "to the same nearest edge sample (not just the single position adjacent to the edge) -- "
        "check that clamping is applied per window position, not once per side.")

    # D: two batches, two features, distinct per-(batch,feature) patterns -- catches axes
    # being averaged, broadcast, or indexed against the wrong dimension.
    x = torch.tensor([
        [[0., 0.], [10., -10.], [20., -20.], [30., -30.], [40., -40.]],
        [[100., 5.], [80., 5.], [60., 5.], [40., 5.], [20., 5.]],
    ], dtype=torch.float64)
    expected = torch.tensor([
        [[10 / 3, -10 / 3], [10., -10.], [20., -20.], [30., -30.], [110 / 3, -110 / 3]],
        [[280 / 3, 5.], [80., 5.], [60., 5.], [40., 5.], [80 / 3, 5.]],
    ], dtype=torch.float64)
    _run_decomposition_case(
        cls, 3, x, expected,
        "Two batches and two features each carry a different pattern (including a constant "
        "channel, whose trend must equal itself everywhere). If any value is off, time is "
        "being mixed with the batch or feature axis somewhere in the averaging.")

    print("Decomposition checks passed.")
    return True


def _direct_circular_scores(queries, keys):
    length = queries.shape[-1]
    q = queries - queries.mean(-1, keepdim=True)
    k = keys - keys.mean(-1, keepdim=True)
    return torch.stack([
        (q * torch.roll(k, shifts=tau, dims=-1)).sum(-1).mean((1, 2))
        for tau in range(length)], dim=-1)


def check_delay_scores(fn):
    generator = torch.Generator().manual_seed(11)
    for batch, heads, features, length in ((2, 1, 1, 7), (3, 2, 4, 16)):
        q = torch.randn(batch, heads, features, length, generator=generator,
                        dtype=torch.float64, requires_grad=True)
        k = torch.randn(batch, heads, features, length, generator=generator,
                        dtype=torch.float64, requires_grad=True)
        scores = fn(q, k)
        assert scores.shape == (batch, length), (
            f"got {list(scores.shape)}, expected [{batch}, {length}]. One score per delay: "
            "average away the head and feature axes, keep batch and time.")
        torch.testing.assert_close(scores, _direct_circular_scores(q, k), msg=lambda default: (
            f"the scores do not match a direct circular correlation.\n{default}\n"
            "Check, in order: both sequences centered along time before transforming; "
            "rfft(q) multiplied by the conjugate of rfft(k), not the other way round; "
            "irfft called with n=L; the mean taken over heads and features, not summed."))
        gradients = torch.autograd.grad(scores.square().sum(), (q, k))
        assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)
    print("FFT delay-score checks passed.")
    return True


def _run_aggregation_case(fn, values, delays, weights, expected, hint):
    values = values.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    actual = fn(values, delays, weights)
    assert actual.shape == values.shape, (
        f"got {list(actual.shape)}, expected {list(values.shape)}. Aggregation returns one "
        "mixed sequence per head and feature, not one per delay.")
    torch.testing.assert_close(actual, expected, msg=lambda default: (
        f"the aggregation does not match the expected circular shifts. {hint}\n{default}"))
    gradients = torch.autograd.grad(actual.square().sum(), (values, weights))
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in gradients), (
        "gradients did not reach both values and weights -- every step from the inputs to "
        "the mixed output must stay a differentiable tensor op.")


def check_aggregation(fn):
    # A: two batches, two heads, two features, each (head, feature) pair scaled by a distinct
    # factor -- catches the batch/head/feature axes being mixed with time or with each other,
    # and catches the delay direction being reversed (batch 0 reuses the notebook's worked
    # [10, 20, 30, 40] example with delays [1, 3], weights [0.75, 0.25] -> [35, 15, 25, 25]).
    base0 = torch.tensor([10., 20., 30., 40.])
    base1 = torch.tensor([1., 2., 3., 4.])
    scales = {(0, 0): 1., (0, 1): 10., (1, 0): 100., (1, 1): 1000.}
    values = torch.zeros(2, 2, 2, 4, dtype=torch.float64)
    for h in range(2):
        for f in range(2):
            values[0, h, f] = base0 * scales[(h, f)]
            values[1, h, f] = base1 * scales[(h, f)]
    delays = torch.tensor([[1, 3], [2, 0]])
    weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]], dtype=torch.float64)
    expected = torch.tensor([
        [[[35., 15., 25., 25.], [350., 150., 250., 250.]],
         [[3500., 1500., 2500., 2500.], [35000., 15000., 25000., 25000.]]],
        [[[1.8, 2.8, 2.2, 3.2], [18., 28., 22., 32.]],
         [[180., 280., 220., 320.], [1800., 2800., 2200., 3200.]]],
    ], dtype=torch.float64)
    _run_aggregation_case(fn, values, delays, weights, expected, (
        "Position t must read the value from t-delay, so the sequence rolls forward: "
        "torch.roll(values, +delay, dims=-1). A reversed shift is the single most common bug "
        "here -- see the worked [10, 20, 30, 40] example above, where z0 = 35."))

    # B: length 5, delays at the two wraparound extremes (delay = length-1 and delay = 0) --
    # catches wraparound being clamped/truncated instead of taken modulo the length.
    values_b = torch.tensor([2., 4., 6., 8., 10.]).reshape(1, 1, 1, 5).to(torch.float64)
    delays_b = torch.tensor([[4, 0]])
    weights_b = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    expected_b = torch.tensor([3., 5., 7., 9., 6.]).reshape(1, 1, 1, 5).to(torch.float64)
    _run_aggregation_case(fn, values_b, delays_b, weights_b, expected_b, (
        "delay=4 on a length-5 sequence must wrap around (equivalent to shifting by -1), not "
        "be clamped to the last index or truncated -- check the shift uses modulo length."))

    print("Delay-aggregation checks passed.")
    return True


def check_world(world):
    """Assert the published four-station, three-product data contract."""
    assert world.values.shape == (4, world.spec.length), (
        "Task 1 must contain four station accelerometers.")
    assert np.array_equal(world.established, np.array((True, True, True, False))), (
        "Stations 1--3 must be established and station 4 must be held out.")
    assert tuple(world.product_names) == ("Product A", "Product B", "Product C"), (
        "Use the three named global products.")
    expected_products = np.arange(len(world.product_index)) % 3
    assert np.array_equal(world.product_index, expected_products), (
        "Product changes must be global and cycle A, B, C.")
    actual_periods = world.primary_periods[0]
    assert np.allclose(world.primary_periods, actual_periods[None, :]), (
        "A batch's operating pace must be shared across the line.")
    bands = world.product_period_bands
    for product, (low, high) in enumerate(bands):
        product_paces = actual_periods[world.product_index == product]
        assert np.all((low <= product_paces) & (product_paces <= high)), (
            "Every batch pace must fall in its named product range.")
        assert np.unique(np.round(product_paces, 6)).size > 1, (
            "Repeated batches of one product must use different operating paces.")
    assert np.allclose(world.values, world.slow + world.primary_cycle + world.noise), (
        "The observed reading must contain only drift, primary vibration, and observation noise.")
    candidate = np.flatnonzero(world.eligible_origin_mask)
    assert np.all(world.regime_index[candidate - data.CONTEXT]
                  == world.regime_index[candidate + data.HORIZON - 1]), (
        "Eligible contexts and forecasts may not cross a product changeover.")
    print("Four-station product-line world checks passed.")
    return True


ALL_COMBINATIONS = (("raw", "attention"), ("raw", "delay"),
                    ("decomposed", "attention"), ("decomposed", "delay"))


def _check_model_output(model, setting):
    x = torch.randn(2, 96, 1, requires_grad=True)
    y = model(x)
    assert isinstance(y, torch.Tensor), (
        f"{setting}: forward returned {type(y).__name__}, not a tensor. Return the forecast itself,")
    assert y.shape == (2, 48), (
        f"{setting}: forward returned {list(y.shape)}, expected [2, 48]. Flatten the encoded "
        "[B, L, d] sequence before the forecast head; do not retain a trailing channel axis.")
    y.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, (
        f"{setting}: no gradient reached the input. Check for .detach(), .item(), or torch.no_grad().")
    starved = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not starved, (
        f"{setting}: these parameters got no gradient, so forward never used them: "
        f"{', '.join(starved)}.")


def check_raw_attention_assembly(cls):
    """Check the Part 1 Raw Attention forecaster before later upgrades exist."""
    model = cls(context=96, horizon=48, d=16, heads=4, layers=1)
    _check_model_output(model, "Raw Attention")
    print("Raw-Attention assembly and gradient checks passed.")
    return True


def check_decomposed_attention_assembly(cls, decomposition_type):
    """Check the supplied Part 2 plumbing after the decomposition TODO passes."""
    model = cls(context=96, horizon=48, d=16, heads=4, layers=1, kernel=25,
                decomposition_type=decomposition_type)
    _check_model_output(model, "Attention + decomposition")
    print("Decomposed-Attention assembly and gradient checks passed.")
    return True


def check_assembly(cls, combinations=ALL_COMBINATIONS):
    """Check the unified Part 4 forecaster on the requested switch settings."""
    for representation, mixing in combinations:
        setting = f"representation={representation!r}, mixing={mixing!r}"
        kwargs = models.components_for(dict(representation=representation, mixing=mixing))
        model = cls(context=96, horizon=48, d=16, heads=4, layers=1,
                    representation=representation, mixing=mixing, **kwargs)
        _check_model_output(model, setting)
    print(f"Model assembly and gradient checks passed ({len(combinations)} setting(s): "
          f"{', '.join(f'{r}/{m}' for r, m in combinations)}).")
    return True


@torch.no_grad()
def _predict(model, x, batch=2048):
    model.eval()
    return np.concatenate([
        model(torch.as_tensor(x[i:i + batch], device=device)).cpu().numpy()
        for i in range(0, len(x), batch)])


def _fit(model, xt, yt, xv, yv, preset, seed, label, scale=1.0, report=4):
    started = time.perf_counter()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIT_CONFIGURATION["learning_rate"],
                                  weight_decay=FIT_CONFIGURATION["weight_decay"])
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=FIT_CONFIGURATION["max_learning_rate"], total_steps=preset.steps,
        pct_start=FIT_CONFIGURATION["pct_start"])
    generator = torch.Generator().manual_seed(2000 + seed)
    X, Y = torch.as_tensor(xt, device=device), torch.as_tensor(yt, device=device)
    every = max(1, preset.steps // report)
    best, state, done, exposure, curve = float("inf"), None, 0, 0, []
    while done < preset.steps:
        for index in torch.randperm(len(X), generator=generator).split(preset.batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(X[index.to(device)]) - Y[index.to(device)]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FIT_CONFIGURATION["gradient_clip_norm"])
            optimizer.step()
            schedule.step()
            done += 1
            exposure += len(index)
            if done % every == 0 or done == preset.steps:
                score = float(((_predict(model, xv) - yv) ** 2).mean())
                curve.append((done, exposure, score))
                if score < best:
                    best = score
                    state = {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()}
            if done >= preset.steps:
                break
    model.load_state_dict(state)
    seconds = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{label}: {done} steps, {seconds:.1f}s, {parameters} parameters, "
          f"best established-validation MSE {best * scale ** 2:.3f} (m/s²)² "
          "-- same physical units as the report tables below", flush=True)
    return model, dict(model=label, steps=done, exposure=exposure, seconds=seconds,
                       curve=curve, best=best, parameters=parameters, device=str(device))


def _array_fingerprint(*arrays):
    """Hash values, shapes, and dtypes without depending on source layout or symbol names."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _model_behavior_fingerprint(name, settings):
    """Fingerprint deterministic model states and probes, not notebook or harness source text."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(913_517)
        model = models.build(name, **settings)
        parameter_digests = []
        for value in model.state_dict().values():
            tensor = value.detach().cpu().contiguous().numpy()
            parameter_digests.append(_array_fingerprint(tensor))

        channels = int(settings.get("channels", 1))
        context = int(settings.get("context", data.CONTEXT))
        probe = torch.linspace(-1.25, 1.75, 2 * context * channels).reshape(2, context, channels)
        with torch.no_grad():
            model.eval()
            evaluation = model(probe).detach().cpu().numpy()
            torch.manual_seed(271_828)
            model.train()
            training = model(probe).detach().cpu().numpy()

    digest = hashlib.sha256()
    for parameter in sorted(parameter_digests):
        digest.update(parameter.encode())
    digest.update(_array_fingerprint(evaluation, training).encode())
    return digest.hexdigest()


class Study:
    """One generated world, frozen preprocessing, and all fitted forecasters."""

    LINEAR = {"Shared ridge": "shared", "Sensor-specific ridge": "sensor",
              "Period-routed ridge": "routed"}

    def __init__(self, preset="quick", seed=0, spec=data.SPEC, model_seed=None):
        self.preset_name, self.preset, self.seed = preset, PRESETS[preset], seed
        self.model_seed = seed if model_seed is None else int(model_seed)
        self.world = data.make_world(spec, seed)
        self.established = self.world.established
        train_values = self.world.values[self.established, :data.TRAIN[1]]
        self.scale = max(float(np.std(np.diff(train_values, axis=1))), 1e-6)
        self._regions, self._linear, self._linear_runs = {}, {}, {}
        self._models, self._runs, self._model_signatures = {}, {}, {}
        self._data_behavior = None
        self._test_selection = None
        self.decomposition_kernel, self.kernel_selection = self._select_kernel()
        self.ridge_lambda, self.ridge_selection = self._select_ridge_lambda()

    def _selection_windows(self):
        origins = data.origins(data.TRAIN, stride=8, eligible=self.world.eligible_origin_mask)
        context, target = data.windows(self.world.values, origins)
        primary, product = data.window_regime_values(self.world, origins)
        return origins, context, target, primary, product

    def _select_kernel(self):
        _, context, _, primary, _ = self._selection_windows()
        rows = []
        keep = self.established[:, None]
        for kernel in DECOMPOSITION_CANDIDATES:
            estimate = data.estimate_period(context[self.established], *PERIOD_RANGE,
                                            detrend_kernel=kernel)
            truth = primary[self.established]
            recovery = float((np.abs(estimate - truth) <= 1.0).mean())
            rows.append({"width": kernel, "training period recovery": recovery})
        table = pd.DataFrame(rows)
        best = int(table.loc[table["training period recovery"].idxmax(), "width"])
        return best, table

    def _select_ridge_lambda(self):
        _, context, target, _, _ = self._selection_windows()
        x, z, _ = data.anchored(context, target)
        x, z = x / self.scale, z / self.scale
        cut = max(1, int(0.8 * x.shape[1]))
        X_fit = x[self.established, :cut].reshape(-1, data.CONTEXT)
        Z_fit = z[self.established, :cut].reshape(-1, data.HORIZON)
        X_choose = x[self.established, cut:].reshape(-1, data.CONTEXT)
        Z_choose = z[self.established, cut:].reshape(-1, data.HORIZON)
        rows = []
        for candidate in RIDGE_CANDIDATES:
            weights = data.ridge_fit(X_fit, Z_fit, candidate)
            score = float(np.mean((X_choose @ weights - Z_choose) ** 2))
            rows.append({"lambda": candidate, "training holdout MSE": score})
        table = pd.DataFrame(rows)
        chosen = float(table.loc[table["training holdout MSE"].idxmin(), "lambda"])
        return chosen, table

    def region(self, name="validation"):
        if name not in self._regions:
            bounds = {"train": data.TRAIN, "validation": data.VALIDATION, "test": data.TEST}[name]
            stride = self.preset.stride_train if name == "train" else self.preset.stride_eval
            origins = data.origins(bounds, stride=stride, eligible=self.world.eligible_origin_mask)
            context, target = data.windows(self.world.values, origins)
            x, z, anchor = data.anchored(context, target)
            primary, product = data.window_regime_values(self.world, origins)
            slow_context, _ = data.windows(self.world.slow, origins)
            estimated = data.estimate_period(context, *PERIOD_RANGE,
                                             detrend_kernel=self.decomposition_kernel)
            self._regions[name] = dict(
                origins=origins, context=context, target=target, anchor=anchor,
                x=x / self.scale, z=z / self.scale, period=estimated,
                route=data.route_period(estimated, self.world.product_period_bands),
                primary_period=primary, product=product,
                slow_change=np.ptp(slow_context, axis=-1), regime=self.world.regime_index[origins])
        return self._regions[name]

    def tensors(self, name):
        part = self.region(name)
        return (part["x"][self.established].reshape(-1, data.CONTEXT, 1).astype(np.float32),
                part["z"][self.established].reshape(-1, data.HORIZON).astype(np.float32))

    def mask(self, which="established", region="validation"):
        part = self.region(region)
        shape = part["period"].shape
        if which == "established":
            return np.broadcast_to(self.established[:, None], shape)
        if which == "held-out":
            return np.broadcast_to((~self.established)[:, None], shape)
        if which in self.world.product_names:
            product = self.world.product_names.index(which)
            return np.broadcast_to((part["product"] == product)[None, :], shape)
        if which == "all":
            return np.ones(shape, dtype=bool)
        raise ValueError(which)

    def score(self, prediction, region="validation", which="established"):
        keep = self.mask(which, region)
        target = self.region(region)["target"][keep]
        return {"MSE": data.mse(prediction[keep], target),
                "RMSE (m/s²)": data.rmse(prediction[keep], target),
                "MAE (m/s²)": data.mae(prediction[keep], target)}

    def _weights(self, kind):
        if kind in self._linear:
            return self._linear[kind]
        started = time.perf_counter()
        train = self.region("train")
        x, z, keep = train["x"], train["z"], self.established
        if kind == "shared":
            out = data.ridge_fit(x[keep].reshape(-1, data.CONTEXT),
                                 z[keep].reshape(-1, data.HORIZON), self.ridge_lambda)
        elif kind == "sensor":
            out = {sensor: data.ridge_fit(x[sensor], z[sensor], self.ridge_lambda)
                   for sensor in np.flatnonzero(keep)}
        elif kind == "routed":
            out = {}
            for route in range(len(data.routing_centres(self.world.product_period_bands))):
                selected = (train["route"] == route) & keep[:, None]
                if selected.sum() >= 24:
                    out[route] = data.ridge_fit(x[selected], z[selected], self.ridge_lambda)
        else:
            raise ValueError(kind)
        self._linear[kind] = out
        self._linear_runs[kind] = time.perf_counter() - started
        return out

    @property
    def period_matrices(self):
        return self._weights("routed")

    def linear_prediction(self, kind, region="validation"):
        part = self.region(region)
        x, anchor = part["x"], part["anchor"]
        if kind == "shared":
            scaled = x @ self._weights(kind)
        elif kind == "sensor":
            scaled = np.full(x.shape[:2] + (data.HORIZON,), np.nan)
            for sensor, matrix in self._weights(kind).items():
                scaled[sensor] = x[sensor] @ matrix
        elif kind == "routed":
            scaled = np.empty(x.shape[:2] + (data.HORIZON,))
            shared = self._weights("shared")
            weights = self._weights(kind)
            for route in np.unique(part["route"]):
                selected = part["route"] == route
                scaled[selected] = x[selected] @ weights.get(int(route), shared)
        else:
            raise ValueError(kind)
        return scaled * self.scale + anchor

    def settings(self, name, **kwargs):
        settings = dict(models.SETTINGS[name])
        settings.update(context=data.CONTEXT, horizon=data.HORIZON, channels=1, heads=4,
                        d=self.preset.width, layers=self.preset.layers, dropout=0.1)
        settings.update(kwargs)
        if settings["representation"] == "decomposed":
            settings.setdefault("kernel", self.decomposition_kernel)
        else:
            settings.pop("kernel", None)
        return settings

    def _data_behavior_fingerprint(self):
        if self._data_behavior is None:
            train_x, train_y = self.tensors("train")
            validation_x, validation_y = self.tensors("validation")
            self._data_behavior = _array_fingerprint(
                np.asarray([self.scale], dtype=np.float64), train_x, train_y,
                validation_x, validation_y)
        return self._data_behavior

    def _cache_identity(self, name, **kwargs):
        settings = self.settings(name, **kwargs)
        return {
            "format": CACHE_FORMAT,
            "model": name,
            "architecture": settings,
            "model_behavior": _model_behavior_fingerprint(name, settings),
            "data_behavior": self._data_behavior_fingerprint(),
            "training": dict(FIT_CONFIGURATION, preset=asdict(self.preset)),
            "data_seed": self.seed,
            "model_seed": self.model_seed,
            "world_spec": asdict(self.world.spec),
        }

    def _checkpoint_path(self, name, identity):
        payload = json.dumps(identity, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return CHECKPOINTS / (f"{slug}-{self.preset_name}-data{self.seed}-"
                              f"model{self.model_seed}-{digest}.pt")

    def checkpoint_path(self, name, **kwargs):
        return self._checkpoint_path(name, self._cache_identity(name, **kwargs))

    @staticmethod
    def _load_state(model, state, name):
        """Accept an equivalent neutral forecast-head rename while requiring an exact tensor schema."""
        target = model.state_dict()
        remapped = dict(state)
        if "forecast_head.weight" in target:
            for suffix in ("weight", "bias"):
                old, new = f"seasonal_head.{suffix}", f"forecast_head.{suffix}"
                if old in remapped and new not in remapped:
                    remapped[new] = remapped.pop(old)
        if set(remapped) != set(target):
            return False
        if any(remapped[key].shape != value.shape for key, value in target.items()):
            return False
        model.load_state_dict(remapped)
        return True

    def _restore(self, model, saved, name, path, signature):
        if not self._load_state(model, saved.get("state", {}), name):
            return None
        record = saved["record"]
        print(f"{name}: loaded {path.name} ({record['steps']} steps, "
              f"{record['seconds']:.1f}s on {record['device']})", flush=True)
        self._models[name], self._runs[name] = model.to(device).eval(), record
        self._model_signatures[name] = signature
        return self._models[name]

    def _compatible_existing_paths(self, name, identity):
        """Find only the shipped smoke checkpoints whose measured behavior still matches exactly."""
        if (self.preset_name != "smoke" or identity["data_behavior"] != _LEGACY_SMOKE_DATA_BEHAVIOR
                or identity["model_behavior"] != _LEGACY_SMOKE_MODEL_BEHAVIOR.get(name)
                or identity["training"] != _LEGACY_SMOKE_TRAINING):
            return ()
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        pattern = f"{slug}-smoke-data{self.seed}-model{self.model_seed}-*.pt"
        return sorted(CHECKPOINTS.glob(pattern))

    def train(self, name="Raw Attention", cache=True, **kwargs):
        identity = self._cache_identity(name, **kwargs)
        path = self._checkpoint_path(name, identity)
        signature = (str(path), identity["model_behavior"], identity["data_behavior"])
        if cache and self._model_signatures.get(name) == signature:
            return self._models[name]
        torch.manual_seed(5000 + self.model_seed)
        model = models.build(name, **self.settings(name, **kwargs))
        if cache and path.exists():
            saved = torch.load(path, map_location=device, weights_only=False)
            if saved.get("cache_identity") == identity:
                restored = self._restore(model, saved, name, path, signature)
                if restored is not None:
                    return restored
            print(f"{name}: cached configuration differs; fitting a new model.", flush=True)
        if cache:
            for existing in self._compatible_existing_paths(name, identity):
                if existing == path:
                    continue
                saved = torch.load(existing, map_location=device, weights_only=False)
                restored = self._restore(model, saved, name, existing, signature)
                if restored is not None:
                    return restored
        xt, yt = self.tensors("train")
        xv, yv = self.tensors("validation")
        model, record = _fit(model, xt, yt, xv, yv, self.preset, self.model_seed, name,
                             scale=self.scale)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": {key: value.cpu() for key, value in model.state_dict().items()},
                        "record": record, "cache_identity": identity}, path)
            print(f"{name}: saved {path.name}", flush=True)
        self._models[name], self._runs[name] = model, record
        self._model_signatures[name] = signature
        return model

    def neural_prediction(self, name, region="validation"):
        part = self.region(region)
        x = part["x"].reshape(-1, data.CONTEXT, 1).astype(np.float32)
        scaled = _predict(self._models[name], x).reshape(part["x"].shape[:2] + (data.HORIZON,))
        return scaled * self.scale + part["anchor"]

    def predictions(self, region="validation"):
        out = {label: self.linear_prediction(kind, region) for label, kind in self.LINEAR.items()}
        out.update({name: self.neural_prediction(name, region) for name in self._models})
        return out

    def _parameters(self, kind):
        size = data.CONTEXT * data.HORIZON
        return {"shared": size, "sensor": size * int(self.established.sum()),
                "routed": size * len(self._weights("routed"))}[kind]

    def model_table(self, region="validation", groups=("established", "held-out")):
        rows = []
        for label, kind in self.LINEAR.items():
            prediction = self.linear_prediction(kind, region)
            for group in groups:
                selected = prediction[self.mask(group, region)]
                rows.append({"model": label, "population": group,
                             "parameters": self._parameters(kind),
                             "fit seconds": self._linear_runs.get(kind, np.nan),
                             "MSE": np.nan if np.isnan(selected).any()
                             else self.score(prediction, region, group)["MSE"],
                             "RMSE (m/s²)": np.nan if np.isnan(selected).any()
                             else self.score(prediction, region, group)["RMSE (m/s²)"],
                             "MAE (m/s²)": np.nan if np.isnan(selected).any()
                             else self.score(prediction, region, group)["MAE (m/s²)"]})
        for name in self._models:
            prediction = self.neural_prediction(name, region)
            for group in groups:
                rows.append({"model": name, "population": group,
                             "parameters": self._runs[name]["parameters"],
                             "fit seconds": self._runs[name]["seconds"],
                             **self.score(prediction, region, group)})
        return pd.DataFrame(rows)

    def grouped_table(self, names, region="validation"):
        rows = []
        for name in names:
            prediction = (self.linear_prediction(self.LINEAR[name], region)
                          if name in self.LINEAR else self.neural_prediction(name, region))
            for population in ("established", "held-out"):
                for product in self.world.product_names:
                    keep = self.mask(population, region) & self.mask(product, region)
                    values = prediction[keep]
                    target = self.region(region)["target"][keep]
                    rows.append({"model": name, "population": population, "product": product,
                                 "MSE": np.nan if np.isnan(values).any() else data.mse(values, target),
                                 "RMSE (m/s²)": (np.nan if np.isnan(values).any()
                                                 else data.rmse(values, target)),
                                 "MAE (m/s²)": (np.nan if np.isnan(values).any()
                                                else data.mae(values, target))})
        return pd.DataFrame(rows)

    def horizon_errors(self, names, region="validation"):
        rows = []
        for name in names:
            prediction = (self.linear_prediction(self.LINEAR[name], region)
                          if name in self.LINEAR else self.neural_prediction(name, region))
            for population in ("established", "held-out"):
                keep = self.mask(population, region)
                curve = np.mean((prediction[keep] - self.region(region)["target"][keep]) ** 2, axis=0)
                rows.extend({"model": name, "population": population, "horizon": h + 1,
                             "MSE": float(value), "RMSE (m/s²)": float(np.sqrt(value))}
                            for h, value in enumerate(curve))
        return pd.DataFrame(rows)

    def select_for_test(self, choices):
        if set(choices) != {"established", "held-out"}:
            raise ValueError("Choose exactly one model for established stations and station 4.")
        eligible = set(self.LINEAR) | set(self._models)
        for population, name in choices.items():
            if name not in eligible:
                raise ValueError(f"Choose a fitted model for {population}: {sorted(eligible)}")
            if population == "held-out" and name == "Sensor-specific ridge":
                raise ValueError("Sensor-specific ridge has no fit for held-out station 4.")
        self._test_selection = dict(choices)
        return pd.DataFrame([{"population": group, "model": name}
                             for group, name in choices.items()])

    def selected_test_table(self):
        if self._test_selection is None:
            raise RuntimeError("Choose both deployment models before reading test errors.")
        rows = []
        for population, name in self._test_selection.items():
            row = {"population": population, "model": name}
            for region in ("validation", "test"):
                prediction = (self.linear_prediction(self.LINEAR[name], region)
                              if name in self.LINEAR else self.neural_prediction(name, region))
                score = self.score(prediction, region, population)
                row[f"{region} MSE"] = score["MSE"]
                row[f"{region} RMSE (m/s²)"] = score["RMSE (m/s²)"]
            rows.append(row)
        return pd.DataFrame(rows)
