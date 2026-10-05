"""A small, physical sensor world for Task 1.

Four accelerometers sit on one production line. The line makes Product A, B, and C in
a repeating global schedule. A product specifies a nominal pace range, and each batch runs
at one pace from that range; a station only changes mounting/calibration details. Station 4
is never used to fit a model or estimate a preprocessing statistic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


SAMPLING_RATE_HZ = 20.0
CONTEXT = 96
HORIZON = 48
TRAIN = (0, 9600)
VALIDATION = (9600, 12480)
TEST = (12480, 15360)

PRODUCT_NAMES = ("Product A", "Product B", "Product C")
PRODUCT_PERIOD_BANDS = np.array(((14.0, 18.0), (21.0, 27.0), (28.0, 36.0)))
PERIOD_SEARCH_RANGE = tuple(PRODUCT_PERIOD_BANDS[[0, -1], [0, 1]])
ROUTING_WIDTH = 1.5


@dataclass(frozen=True)
class WorldSpec:
    n_sensors: int = 4
    length: int = 15360
    sampling_rate_hz: float = SAMPLING_RATE_HZ
    regime_duration: int = 192
    product_names: tuple[str, ...] = PRODUCT_NAMES
    product_period_bands: tuple[tuple[float, float], ...] = (
        (14.0, 18.0), (21.0, 27.0), (28.0, 36.0))
    vibration_amplitudes: tuple[float, ...] = (1.75, 2.05, 1.85, 2.15)
    mounting_phases: tuple[float, ...] = (0.25, 1.10, 2.15, 2.85)
    calibration_offsets: tuple[float, ...] = (-0.30, 0.15, -0.10, 0.35)
    drift_amplitudes: tuple[float, ...] = (4.80, 5.25, 4.55, 5.00)
    drift_phases: tuple[float, ...] = (0.20, 1.15, 2.05, 2.75)
    drift_period: float = 240.0
    noise_sigma: float = 0.09


SPEC = WorldSpec()


@dataclass
class World:
    spec: WorldSpec
    values: np.ndarray
    slow: np.ndarray
    seasonal: np.ndarray
    primary_cycle: np.ndarray
    primary_phase: np.ndarray
    noise: np.ndarray
    regime_boundaries: np.ndarray
    regime_index: np.ndarray
    product_index: np.ndarray
    primary_periods: np.ndarray
    eligible_origin_mask: np.ndarray
    levels: np.ndarray
    drift_amplitudes: np.ndarray
    upstream: np.ndarray
    established: np.ndarray
    seed: int

    @property
    def held_out(self) -> np.ndarray:
        return ~self.established

    @property
    def trend(self) -> np.ndarray:
        return self.slow

    @property
    def product_names(self) -> tuple[str, ...]:
        return self.spec.product_names

    @property
    def product_period_bands(self) -> np.ndarray:
        return np.asarray(self.spec.product_period_bands, dtype=float)


def _regime_boundaries(spec: WorldSpec) -> np.ndarray:
    return np.arange(0, spec.length + spec.regime_duration, spec.regime_duration,
                     dtype=int).clip(max=spec.length)


def _batch_periods(spec: WorldSpec, product_index: np.ndarray, seed: int) -> np.ndarray:
    """One reproducible, continuously valued pace per batch and product range.

    The golden-ratio sequence spreads the finite assignment evenly through every product's
    allowed range. It avoids a second stochastic mechanism while ensuring that repeated
    batches of one product do not reduce to a few memorisable frequencies.
    """
    bands = np.asarray(spec.product_period_bands, dtype=float)
    counts = np.zeros(len(spec.product_names), dtype=int)
    periods = np.empty(len(product_index), dtype=float)
    golden_fraction = (np.sqrt(5.0) - 1.0) / 2.0
    for regime, product in enumerate(product_index):
        occurrence = counts[product]
        counts[product] += 1
        fraction = ((occurrence + 1) * golden_fraction + 0.173 * (product + 1)
                    + 0.071 * seed) % 1.0
        low, high = bands[product]
        periods[regime] = low + (high - low) * fraction
    return periods


def make_world(spec: WorldSpec = SPEC, seed: int = 0) -> World:
    """Generate one line with only vibration, a slow drift, and observation noise."""
    if spec.n_sensors != 4:
        raise ValueError("Task 1 models exactly four stations.")
    station_parameters = (
        spec.vibration_amplitudes, spec.mounting_phases, spec.calibration_offsets,
        spec.drift_amplitudes, spec.drift_phases,
    )
    if any(len(parameter) != spec.n_sensors for parameter in station_parameters):
        raise ValueError("Give one mounting/calibration value for each of the four stations.")
    if len(spec.product_names) != 3 or len(spec.product_period_bands) != 3:
        raise ValueError("Task 1 uses exactly Product A, Product B, and Product C.")
    bands = np.asarray(spec.product_period_bands, dtype=float)
    if bands.shape != (3, 2) or np.any(bands[:, 0] >= bands[:, 1]):
        raise ValueError("Give each product one increasing operating-period range.")

    rng = np.random.default_rng(10_000 + seed)
    n, length = spec.n_sensors, spec.length
    time = np.arange(length, dtype=float)
    boundaries = _regime_boundaries(spec)
    n_regimes = len(boundaries) - 1
    regime_index = np.minimum(
        np.searchsorted(boundaries[1:], np.arange(length), side="right"), n_regimes - 1)
    product_index = np.arange(n_regimes, dtype=int) % len(spec.product_names)
    batch_periods = _batch_periods(spec, product_index, seed)
    primary_periods = np.broadcast_to(batch_periods, (n, n_regimes)).copy()

    # A batch fixes one line-wide rotational pace. Product names set nominal ranges; load and
    # material conditions choose the actual pace. Phase is accumulated so a shaft does not
    # restart at a changeover; stations differ only in mounting phase and amplitude.
    period_at_time = primary_periods[:, regime_index]
    primary_phase = np.empty((n, length), dtype=float)
    primary_phase[:, 0] = np.asarray(spec.mounting_phases, dtype=float)
    primary_phase[:, 1:] = primary_phase[:, :1] + np.cumsum(
        2 * np.pi / period_at_time[:, :-1], axis=1)
    amplitudes = np.asarray(spec.vibration_amplitudes, dtype=float)
    primary_cycle = amplitudes[:, None] * np.sin(primary_phase)
    seasonal = primary_cycle

    # A broad calibration/warm-up drift is locally ramp-like, yet large enough that raw period
    # evidence is visibly less clean than the decomposed residual.
    levels = np.asarray(spec.calibration_offsets, dtype=float)
    drift_amplitudes = np.asarray(spec.drift_amplitudes, dtype=float)
    drift_phase = np.asarray(spec.drift_phases, dtype=float)
    slow = levels[:, None] + drift_amplitudes[:, None] * np.sin(
        2 * np.pi * time[None, :] / spec.drift_period + drift_phase[:, None])

    noise = rng.normal(0.0, spec.noise_sigma, size=(n, length))
    values = slow + seasonal + noise
    established = np.array((True, True, True, False))
    upstream = np.array((-1, 0, 1, 2))  # Station 3 is immediately upstream of held-out station 4.

    eligible = np.zeros(length, dtype=bool)
    candidate = np.arange(CONTEXT, length - HORIZON + 1)
    same_regime = (regime_index[candidate - CONTEXT]
                   == regime_index[candidate + HORIZON - 1])
    eligible[candidate] = same_regime
    return World(spec, values, slow, seasonal, primary_cycle, primary_phase, noise,
                 boundaries, regime_index, product_index, primary_periods, eligible,
                 levels, drift_amplitudes, upstream, established, seed)


def seconds(samples):
    """Convert sample offsets to seconds at the published fixed sampling rate."""
    return np.asarray(samples, dtype=float) / SAMPLING_RATE_HZ


def product_for_regimes(world: World, regimes: np.ndarray) -> np.ndarray:
    return world.product_index[np.asarray(regimes, dtype=int)]


def origins(region: tuple[int, int], context: int = CONTEXT, horizon: int = HORIZON,
            stride: int = 1, eligible: np.ndarray | None = None) -> np.ndarray:
    out = np.arange(max(region[0], context), region[1] - horizon + 1, stride, dtype=int)
    return out if eligible is None else out[eligible[out]]


def windows(values: np.ndarray, origins_: np.ndarray, context: int = CONTEXT,
            horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
    index_context = origins_[:, None] + np.arange(-context, 0)[None, :]
    index_target = origins_[:, None] + np.arange(horizon)[None, :]
    return values[:, index_context], values[:, index_target]


def window_regime_values(world: World, origins_: np.ndarray):
    regimes = world.regime_index[origins_]
    return world.primary_periods[:, regimes], world.product_index[regimes]


def mse(prediction, target, axis=None):
    values = (np.asarray(prediction) - np.asarray(target)) ** 2
    return float(np.mean(values)) if axis is None else np.mean(values, axis=axis)


def rmse(prediction, target, axis=None):
    return np.sqrt(mse(prediction, target, axis=axis))


def mae(prediction, target, axis=None):
    values = np.abs(np.asarray(prediction) - np.asarray(target)
    )
    return float(np.mean(values)) if axis is None else np.mean(values, axis=axis)


def anchored(context: np.ndarray, target: np.ndarray | None = None):
    """Subtract the last observation; level is removed while within-window drift remains."""
    anchor = context[..., -1:]
    return context - anchor, (None if target is None else target - anchor), anchor


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    n, width = X.shape
    return np.linalg.solve(X.T @ X / n + lam * np.eye(width), X.T @ Y / n)


def moving_average(x: np.ndarray, kernel: int) -> np.ndarray:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be positive and odd")
    radius = kernel // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(radius, radius)], mode="edge")
    cumulative = np.cumsum(np.concatenate([np.zeros_like(padded[..., :1]), padded], axis=-1), axis=-1)
    return (cumulative[..., kernel:] - cumulative[..., :-kernel]) / kernel


def dense_period_scores(context: np.ndarray, lo: float = PERIOD_SEARCH_RANGE[0],
                        hi: float = PERIOD_SEARCH_RANGE[1],
                        step: float = 0.25, detrend_kernel: int = 25):
    """Parameter-free sinusoidal projection scores on the three-product period band."""
    signal = np.asarray(context)
    if detrend_kernel:
        signal = signal - moving_average(signal, detrend_kernel)
    signal = signal - signal.mean(-1, keepdims=True)
    grid = np.arange(lo, hi + step / 2, step)
    time = np.arange(signal.shape[-1])
    basis = np.exp(-2j * np.pi * time[:, None] / grid[None, :])
    projection = signal @ basis
    scores = np.abs(projection) ** 2 / (np.sum(signal ** 2, axis=-1, keepdims=True) + 1e-12)
    return grid, scores


def estimate_period(context: np.ndarray, lo: float = PERIOD_SEARCH_RANGE[0],
                    hi: float = PERIOD_SEARCH_RANGE[1],
                    detrend_kernel: int = 25, step: float = 0.25) -> np.ndarray:
    grid, scores = dense_period_scores(context, lo, hi, step, detrend_kernel)
    return grid[np.argmax(scores, axis=-1)]


def routing_centres(period_bands=PRODUCT_PERIOD_BANDS, width: float = ROUTING_WIDTH) -> np.ndarray:
    """Centres of the hand-designed, disjoint operating-pace bands used by ridge routing."""
    if width <= 0:
        raise ValueError("The routing-band width must be positive.")
    bands = np.asarray(period_bands, dtype=float)
    return np.concatenate([
        low + width * (np.arange(int(np.ceil((high - low) / width))) + 0.5)
        for low, high in bands
    ])


def route_period(period: np.ndarray, period_bands=PRODUCT_PERIOD_BANDS,
                 width: float = ROUTING_WIDTH) -> np.ndarray:
    """Route an estimated pace to the nearest fixed operating-pace band."""
    candidates = routing_centres(period_bands, width)
    return np.abs(np.asarray(period)[..., None] - candidates).argmin(axis=-1)


def seasonal_copy(context: np.ndarray, period: np.ndarray, horizon: int = HORIZON):
    integer_period = np.rint(np.broadcast_to(np.asarray(period), context.shape[:-1])).astype(int)
    steps = np.arange(horizon)
    repeats = steps[None, ...] // integer_period[..., None] + 1
    source = np.clip(steps[None, ...] - repeats * integer_period[..., None],
                     -context.shape[-1], -1)
    flat = np.take_along_axis(context.reshape(-1, context.shape[-1]),
                              source.reshape(-1, horizon) + context.shape[-1], axis=1)
    return flat.reshape(*context.shape[:-1], horizon)
