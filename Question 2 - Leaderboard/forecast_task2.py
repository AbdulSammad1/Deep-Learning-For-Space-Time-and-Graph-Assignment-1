from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ROOT / "Data"
RESULTS_FOLDER = ROOT / "outputs"
FORECAST_STEPS = 168
SCALED_EXTERNAL_COLUMNS = 6


@dataclass(frozen=True)
class RunSettings:
    context: int
    epochs: int
    stride: int
    seeds: tuple[int, ...]
    validation_blocks: int
    width: int
    layers: int
    kernel: int
    dropout: float
    batch_size: int
    learning_rate: float
    smoke: bool


@dataclass(frozen=True)
class InputSeries:
    target: np.ndarray
    external: np.ndarray


@dataclass
class WindowBatch:
    past: torch.Tensor
    future_external: torch.Tensor
    daily_pattern: torch.Tensor
    guide: torch.Tensor
    raw_target: torch.Tensor | None

    def move(self, device: torch.device) -> "WindowBatch":
        raw_target = None if self.raw_target is None else self.raw_target.to(device)
        return WindowBatch(
            past=self.past.to(device),
            future_external=self.future_external.to(device),
            daily_pattern=self.daily_pattern.to(device),
            guide=self.guide.to(device),
            raw_target=raw_target,
        )


def read_inputs() -> InputSeries:
    # So here I am loading the training, test, and optional external data files
    train = pd.read_csv(DATA_FOLDER / "student_train.csv")
    test = pd.read_csv(DATA_FOLDER / "student_test.csv")
    external = pd.read_csv(DATA_FOLDER / "optional_external_data.csv")

    y = train["value"].to_numpy(dtype=np.float64)
    x = external.drop(columns=["time_idx"]).to_numpy(dtype=np.float64)

    if test.shape[0] != FORECAST_STEPS:
        raise ValueError(f"student_test.csv must contain exactly {FORECAST_STEPS} rows")
    if x.shape[0] != y.shape[0] + FORECAST_STEPS:
        raise ValueError("optional_external_data.csv does not cover the full train + test period")

    return InputSeries(target=y, external=x)


class ExternalStandardizer:
    def __init__(self, continuous_columns: int = SCALED_EXTERNAL_COLUMNS):
        self.continuous_columns = continuous_columns
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, matrix: np.ndarray, rows: int) -> "ExternalStandardizer":
        # So here I am learning the external feature scaling only from the available history
        observed = matrix[:rows, : self.continuous_columns]
        self.mean_ = observed.mean(axis=0)
        self.std_ = np.maximum(observed.std(axis=0), 1e-6)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ExternalStandardizer must be fitted before transform()")
        result = matrix.copy()
        result[:, : self.continuous_columns] = (
            result[:, : self.continuous_columns] - self.mean_
        ) / self.std_
        return result


def valid_origins(start: int, stop: int, context: int, horizon: int, stride: int) -> np.ndarray:
    first_origin = max(start, context)
    final_origin = stop - horizon
    if final_origin < first_origin:
        return np.empty(0, dtype=np.int64)
    return np.arange(first_origin, final_origin + 1, stride, dtype=np.int64)


class WindowBuilder:
    def __init__(
        self,
        raw_target: np.ndarray,
        external: np.ndarray,
        context: int,
        horizon: int,
        include_external: bool,
        log_mean: float,
        log_std: float,
        day_length: int = 24,
    ):
        self.raw_target = raw_target
        self.log_target = np.log1p(raw_target)
        self.external = external
        self.context = context
        self.horizon = horizon
        self.include_external = include_external
        self.log_mean = log_mean
        self.log_std = log_std
        self.day_length = day_length

    def build(self, origins: np.ndarray | list[int], require_target: bool = True) -> WindowBatch:
        history_rows: list[np.ndarray] = []
        future_rows: list[np.ndarray] = []
        pattern_rows: list[np.ndarray] = []
        guide_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        profile_days = max(1, min(14, self.context // self.day_length))

        for origin in origins:
            # So here I am scaling the past on a fixed log scale, not by the last point alone
            history = self.log_target[origin - self.context : origin]
            normalized = ((history - self.log_mean) / self.log_std)[:, None]
            if self.include_external:
                past_external = self.external[origin - self.context : origin]
                normalized = np.concatenate((normalized, past_external), axis=1)
                future_external = self.external[origin : origin + self.horizon]
            else:
                future_external = np.zeros((self.horizon, 1), dtype=np.float64)

            # So here I am using the average hour-of-day shape, not the last quiet day
            profile_history = self.raw_target[origin - profile_days * self.day_length : origin]
            hour_profile = profile_history.reshape(profile_days, self.day_length).mean(axis=0)
            daily_pattern = np.resize(hour_profile, self.horizon)
            lookback = min(self.horizon, self.context)
            recent_block = self.raw_target[origin - lookback : origin]
            if lookback < self.horizon:
                recent_block = np.resize(recent_block, self.horizon)
            recent_level = float(profile_history.mean())
            guide = np.stack(
                (
                    np.log1p(daily_pattern),
                    daily_pattern / 100.0,
                    recent_block / 100.0,
                    np.full(self.horizon, recent_level / 100.0),
                ),
                axis=1,
            )

            history_rows.append(normalized)
            future_rows.append(future_external)
            pattern_rows.append(daily_pattern)
            guide_rows.append(guide)

            if require_target:
                future_target = self.raw_target[origin : origin + self.horizon]
                if future_target.shape[0] != self.horizon:
                    raise ValueError("A requested training window extends beyond the known target series")
                target_rows.append(future_target)

        raw_target = None
        if require_target:
            raw_target = torch.as_tensor(np.stack(target_rows), dtype=torch.float32)
        return WindowBatch(
            past=torch.as_tensor(np.stack(history_rows), dtype=torch.float32),
            future_external=torch.as_tensor(np.stack(future_rows), dtype=torch.float32),
            daily_pattern=torch.as_tensor(np.stack(pattern_rows), dtype=torch.float32),
            guide=torch.as_tensor(np.stack(guide_rows), dtype=torch.float32),
            raw_target=raw_target,
        )


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # So here I am separating the smoother trend from the faster changing part
        padding = self.kernel_size // 2
        channel_major = sequence.transpose(1, 2)
        padded = F.pad(channel_major, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
        residual = sequence - trend
        return residual, trend


class LagCorrelationMixer(nn.Module):
    def __init__(self, width: int, max_lags: int = 8):
        super().__init__()
        self.max_lags = max_lags
        self.q_projection = nn.Linear(width, width)
        self.k_projection = nn.Linear(width, width)
        self.v_projection = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)

    @staticmethod
    def _demean(sequence: torch.Tensor) -> torch.Tensor:
        return sequence - sequence.mean(dim=1, keepdim=True)

    def _rank_lags(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = query.shape[1]
        # So here I am using FFT correlation to find the strongest repeating time delays
        q_fft = torch.fft.rfft(self._demean(query), dim=1)
        k_fft = torch.fft.rfft(self._demean(key), dim=1)
        correlations = torch.fft.irfft(q_fft * k_fft.conj(), n=steps, dim=1).mean(dim=-1)

        nonzero_lags = correlations[:, 1:]
        k = min(self.max_lags, nonzero_lags.shape[-1])
        strengths, zero_based = torch.topk(nonzero_lags, k=k, dim=-1)
        return zero_based + 1, torch.softmax(strengths, dim=-1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = sequence.shape
        query = self.q_projection(sequence)
        key = self.k_projection(sequence)
        value = self.v_projection(sequence)
        lags, weights = self._rank_lags(query, key)

        timeline = torch.arange(steps, device=sequence.device)
        source_index = (timeline[None, :, None] - lags[:, None, :]) % steps

        # So here I am mixing shifted versions of the sequence using the selected lag weights
        mixed = torch.zeros_like(value)
        for lag_position in range(lags.shape[1]):
            gather_index = source_index[:, :, lag_position].unsqueeze(-1).expand_as(value)
            shifted = torch.gather(value, dim=1, index=gather_index)
            mixed += shifted * weights[:, lag_position].view(batch_size, 1, 1)

        return self.output_projection(mixed)


class LagResidualBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pre_mix_norm = nn.LayerNorm(width)
        self.pre_ff_norm = nn.LayerNorm(width)
        self.lag_mixer = LagCorrelationMixer(width)
        self.decompose = MovingAverageDecomposition(kernel_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state + self.dropout(self.lag_mixer(self.pre_mix_norm(state)))
        state, trend_a = self.decompose(state)
        state = state + self.dropout(self.feed_forward(self.pre_ff_norm(state)))
        state, trend_b = self.decompose(state)
        return state, trend_a + trend_b


class ForecastNetwork(nn.Module):
    def __init__(
        self,
        context: int,
        horizon: int,
        input_channels: int,
        external_channels: int,
        width: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.decompose_input = MovingAverageDecomposition(kernel_size)
        self.input_projection = nn.Linear(input_channels, width)
        self.position_bias = nn.Parameter(torch.empty(1, context, width))
        nn.init.normal_(self.position_bias, mean=0.0, std=0.02)

        self.encoder = nn.ModuleList(
            LagResidualBlock(width, kernel_size, dropout) for _ in range(layers)
        )
        weather_channels = external_channels if external_channels else 1
        self.weather_projection = nn.Linear(weather_channels, width)
        self.pattern_projection = nn.Linear(1, width)
        self.step_embedding = nn.Embedding(horizon, width)
        self.decoder_in = nn.Linear(width * 4, width)
        self.decoder = nn.ModuleList(
            LagResidualBlock(width, kernel_size, dropout) for _ in range(layers)
        )
        # So here I am letting future weather and the daily shape make the main guess
        self.direct_head = nn.Linear(weather_channels + 4, 1)
        nn.init.zeros_(self.direct_head.weight)
        nn.init.constant_(self.direct_head.bias, 70.0)
        # So here I am starting the leftover correction at zero so it cannot drown that guess
        self.correction_head = nn.Linear(width, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(
        self,
        past: torch.Tensor,
        future_external: torch.Tensor,
        daily_pattern: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        residual, _trend = self.decompose_input(past)
        encoded = self.input_projection(residual) + self.position_bias
        for block in self.encoder:
            encoded, _ = block(encoded)

        summary = encoded.mean(dim=1, keepdim=True).expand(-1, daily_pattern.shape[1], -1)
        steps = torch.arange(daily_pattern.shape[1], device=past.device)
        step_mark = self.step_embedding(steps).unsqueeze(0).expand(past.shape[0], -1, -1)
        weather = self.weather_projection(future_external)
        pattern = self.pattern_projection(torch.log1p(daily_pattern).unsqueeze(-1))
        decoded = self.decoder_in(torch.cat((summary, step_mark, weather, pattern), dim=-1))
        for block in self.decoder:
            decoded, _ = block(decoded)

        # So here I am adding a small learned correction on top of the weather guess
        direct = self.direct_head(torch.cat((future_external, guide), dim=-1)).squeeze(-1)
        correction = self.correction_head(decoded).squeeze(-1)
        return F.softplus(direct + correction)


def prime_direct_head(model: ForecastNetwork, windows: WindowBuilder, origins: np.ndarray) -> None:
    # So here I am fitting the weather guess on past weeks before the network trains
    batch = windows.build(origins)
    if batch.raw_target is None:
        raise RuntimeError("The weather guess needs known target values")
    features = torch.cat((batch.future_external, batch.guide), dim=-1)
    features = features.reshape(-1, features.shape[-1]).numpy().astype(np.float64)
    target = batch.raw_target.reshape(-1).numpy().astype(np.float64)
    target = np.clip(target, 1e-3, None)
    pre_activation = target.copy()
    small_values = target <= 20.0
    pre_activation[small_values] = np.log(np.expm1(target[small_values]))

    design = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
    penalty = 5.0
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += penalty
    gram[0, 0] -= penalty
    coefficients = np.linalg.solve(gram, design.T @ pre_activation)

    with torch.no_grad():
        model.direct_head.bias.copy_(torch.tensor(float(coefficients[0]), dtype=torch.float32))
        model.direct_head.weight.copy_(
            torch.tensor(coefficients[1:], dtype=torch.float32).view_as(model.direct_head.weight)
        )


def model_size(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "sMAPE": float(np.mean(200.0 * np.abs(error) / denominator)),
    }


def benchmark_rows(values: np.ndarray, origins: np.ndarray, horizon: int) -> list[dict[str, float | str]]:
    definitions = (
        ("repeat last value", 1),
        ("repeat last 24 steps", 24),
    )
    rows: list[dict[str, float | str]] = []

    for label, lag in definitions:
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        for origin in origins:
            if lag == 1:
                predictions.append(np.full(horizon, values[origin - 1]))
            else:
                predictions.append(np.resize(values[origin - lag : origin], horizon))
            actuals.append(values[origin : origin + horizon])

        metrics = regression_metrics(np.concatenate(actuals), np.concatenate(predictions))
        rows.append({"model": label, **metrics})

    return rows


class Trainer:
    def __init__(
        self,
        model: ForecastNetwork,
        windows: WindowBuilder,
        device: torch.device,
        batch_size: int,
        learning_rate: float,
    ):
        self.model = model.to(device)
        self.windows = windows
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

    def _train_epoch(self, origins: np.ndarray) -> float:
        order = np.random.permutation(origins)
        weighted_loss = 0.0
        samples = 0
        self.model.train()

        for offset in range(0, len(order), self.batch_size):
            # So here I am training on a shuffled mini batch of historical windows
            chunk = order[offset : offset + self.batch_size]
            batch = self.windows.build(chunk).move(self.device)
            assert batch.raw_target is not None

            self.optimizer.zero_grad(set_to_none=True)
            # So here I am scoring the guess against the real measurements, which is what the leaderboard uses
            prediction = self.model(
                batch.past, batch.future_external, batch.daily_pattern, batch.guide
            )
            loss = F.mse_loss(prediction, batch.raw_target) / 10000.0
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            weighted_loss += float(loss.detach()) * len(chunk)
            samples += len(chunk)

        return weighted_loss / max(samples, 1)

    def evaluate(self, origins: np.ndarray) -> dict[str, float]:
        predicted_blocks: list[np.ndarray] = []
        actual_blocks: list[np.ndarray] = []

        # So here I am checking the model on validation origins in the original value scale
        self.model.eval()
        with torch.no_grad():
            for origin in origins:
                batch = self.windows.build([int(origin)]).move(self.device)
                prediction = self.model(
                    batch.past, batch.future_external, batch.daily_pattern, batch.guide
                )
                predicted_blocks.append(prediction.cpu().numpy()[0])
                actual_blocks.append(self.windows.raw_target[origin : origin + self.windows.horizon])

        return regression_metrics(np.concatenate(actual_blocks), np.concatenate(predicted_blocks))

    def fit_with_early_stopping(
        self,
        train_origins: np.ndarray,
        validation_origins: np.ndarray,
        epochs: int,
    ) -> tuple[int, float]:
        best_rmse = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            train_mse = self._train_epoch(train_origins)
            validation = self.evaluate(validation_origins)
            print(
                f"  epoch {epoch:02d}  train MSE {train_mse:.4f}  "
                f"val RMSE {validation['RMSE']:.3f}"
            )

            # So here I am keeping the best model and stopping if validation stops improving
            if validation["RMSE"] < best_rmse - 1e-4:
                best_rmse = validation["RMSE"]
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if epoch >= 5 and stale_epochs >= 4:
                    break

        if best_state is None:
            raise RuntimeError("Training finished without producing a model state")
        self.model.load_state_dict(best_state)
        return best_epoch, best_rmse

    def fit_fixed_epochs(self, origins: np.ndarray, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            self._train_epoch(origins)
            print(f"  final epoch {epoch:02d}")

    def forecast(self, origin: int) -> np.ndarray:
        self.model.eval()
        batch = self.windows.build([origin], require_target=False).move(self.device)
        with torch.no_grad():
            prediction = self.model(
                batch.past, batch.future_external, batch.daily_pattern, batch.guide
            )
        return prediction.cpu().numpy()[0]


def build_model(
    settings: RunSettings,
    context: int,
    external_width: int,
    include_external: bool,
) -> ForecastNetwork:
    input_channels = 1 + (external_width if include_external else 0)
    future_channels = external_width if include_external else 0
    return ForecastNetwork(
        context=context,
        horizon=FORECAST_STEPS,
        input_channels=input_channels,
        external_channels=future_channels,
        width=settings.width,
        layers=settings.layers,
        kernel_size=settings.kernel,
        dropout=settings.dropout,
    )


def resolve_settings(args: argparse.Namespace) -> RunSettings:
    return RunSettings(
        context=96 if args.smoke else args.context,
        epochs=1 if args.smoke else args.epochs,
        stride=400 if args.smoke else 24,
        seeds=(0,) if args.smoke else tuple(int(item) for item in args.seeds.split(",")),
        validation_blocks=1 if args.smoke else 4,
        width=args.width,
        layers=args.layers,
        kernel=args.kernel,
        dropout=args.dropout,
        batch_size=args.batch,
        learning_rate=args.lr,
        smoke=args.smoke,
    )


def run_experiment(args: argparse.Namespace) -> None:
    settings = resolve_settings(args)
    data = read_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # So here I am reserving the last few 168-step blocks for validation
    validation_start = len(data.target) - settings.validation_blocks * FORECAST_STEPS
    validation_origins = np.arange(validation_start, len(data.target), FORECAST_STEPS)
    train_origins = valid_origins(
        0,
        validation_start,
        settings.context,
        FORECAST_STEPS,
        settings.stride,
    )

    print(
        f"device={device}  context={settings.context}  horizon={FORECAST_STEPS}  "
        f"epochs={settings.epochs}"
    )
    print(
        f"train windows={len(train_origins)}  "
        f"validation blocks={len(validation_origins)}"
    )
    print("simple baselines on the same validation blocks:")
    for row in benchmark_rows(data.target, validation_origins, FORECAST_STEPS):
        print(
            f"  {row['model']}: RMSE {row['RMSE']:.3f}  "
            f"MAE {row['MAE']:.3f}  sMAPE {row['sMAPE']:.2f}"
        )

    validation_scaler = ExternalStandardizer().fit(data.external, validation_start)
    validation_external = validation_scaler.transform(data.external)
    logged_history = np.log1p(data.target[:validation_start])
    log_mean = float(logged_history.mean())
    log_std = max(float(logged_history.std()), 1e-3)
    records: list[dict[str, float | int | bool]] = []

    # So here I am comparing the model with and without the optional external features
    for include_external in (False, True):
        for seed in settings.seeds:
            print(f"\nextras={include_external} seed={seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            windows = WindowBuilder(
                data.target,
                validation_external,
                settings.context,
                FORECAST_STEPS,
                include_external,
                log_mean,
                log_std,
            )
            model = build_model(
                settings,
                settings.context,
                validation_external.shape[1],
                include_external,
            )
            trainer = Trainer(
                model,
                windows,
                device,
                settings.batch_size,
                settings.learning_rate,
            )
            prime_direct_head(model, windows, train_origins)
            best_epoch, _ = trainer.fit_with_early_stopping(
                train_origins,
                validation_origins,
                settings.epochs,
            )
            scores = trainer.evaluate(validation_origins)
            records.append(
                {
                    **scores,
                    "use_extra": include_external,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "parameters": model_size(model),
                }
            )
            print(
                f"  kept epoch {best_epoch}  RMSE {scores['RMSE']:.3f}  "
                f"MAE {scores['MAE']:.3f}  sMAPE {scores['sMAPE']:.2f}"
            )

    validation_table = pd.DataFrame(records)
    summary = (
        validation_table.groupby("use_extra")[["RMSE", "MAE", "sMAPE", "best_epoch"]]
        .mean()
        .reset_index()
        .sort_values("RMSE")
    )
    print("\nmean over seeds:")
    print(summary.round(3).to_string(index=False))

    # So here I am selecting the better validation setup and its average best epoch count
    chosen_external = bool(summary.iloc[0]["use_extra"])
    final_epochs = max(1, int(round(float(summary.iloc[0]["best_epoch"]))))
    print(f"chosen extras={chosen_external}  final epochs={final_epochs}")

    # So here I am fitting one model per seed on the whole history, then averaging their guesses
    final_scaler = ExternalStandardizer().fit(data.external, len(data.target))
    final_external = final_scaler.transform(data.external)
    final_logged = np.log1p(data.target)
    final_log_mean = float(final_logged.mean())
    final_log_std = max(float(final_logged.std()), 1e-3)
    final_origins = valid_origins(
        0,
        len(data.target),
        settings.context,
        FORECAST_STEPS,
        stride=24 if not settings.smoke else settings.stride,
    )
    member_forecasts: list[np.ndarray] = []
    parameter_total = 0
    for seed in settings.seeds:
        print(f"\nfinal seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        final_windows = WindowBuilder(
            data.target,
            final_external,
            settings.context,
            FORECAST_STEPS,
            chosen_external,
            final_log_mean,
            final_log_std,
        )
        final_model = build_model(
            settings,
            settings.context,
            final_external.shape[1],
            chosen_external,
        )
        final_trainer = Trainer(
            final_model,
            final_windows,
            device,
            settings.batch_size,
            settings.learning_rate,
        )
        prime_direct_head(final_model, final_windows, final_origins)
        final_trainer.fit_fixed_epochs(final_origins, final_epochs)
        member_forecasts.append(final_trainer.forecast(len(data.target)))
        parameter_total += model_size(final_model)

    forecast = np.mean(member_forecasts, axis=0)
    epoch_total = final_epochs * len(settings.seeds)

    if forecast.shape[0] != FORECAST_STEPS:
        raise RuntimeError(f"Expected {FORECAST_STEPS} forecast values, got {forecast.shape[0]}")

    # So here I am saving the validation results, final forecast, and run information
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    run_label = "smoke" if settings.smoke else "full"
    validation_table.to_csv(RESULTS_FOLDER / f"validation_{run_label}.csv", index=False)
    forecast_text = ", ".join(f"{value:.6f}" for value in forecast)
    (RESULTS_FOLDER / f"forecast_{run_label}.txt").write_text(forecast_text + "\n")

    notes = (
        f"parameters={parameter_total}\n"
        f"epochs_that_produced_this_forecast={epoch_total}\n"
        f"used_optional_file={chosen_external}\n"
        f"seeds_used_for_the_choice={list(settings.seeds)}\n"
        f"context={settings.context}\n"
    )
    (RESULTS_FOLDER / f"submission_notes_{run_label}.txt").write_text(notes)

    print("\n" + notes)
    print(f"wrote {RESULTS_FOLDER / f'forecast_{run_label}.txt'}")
    print("Paste that line only after a non-smoke run. You get 5 leaderboard submissions.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the lag-correlation forecaster")
    parser.add_argument("--smoke", action="store_true", help="run a minimal training pass")
    parser.add_argument("--context", type=int, default=336)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--kernel", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_arguments())
