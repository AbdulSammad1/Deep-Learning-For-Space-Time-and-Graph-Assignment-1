"""Registration and construction for the notebook-defined forecasting model."""
from __future__ import annotations

_DECOMPOSITION = None
_DELAY_SCORES = None
_AGGREGATE = None
_RAW_FORECASTER = None
_DECOMPOSED_ATTENTION_FORECASTER = None
_FORECASTER = None


def register_components(decomposition_type=None, delay_scores=None, aggregate=None,
                        raw_forecaster_type=None, decomposed_attention_forecaster_type=None,
                        forecaster_type=None):
    global _DECOMPOSITION, _DELAY_SCORES, _AGGREGATE, _RAW_FORECASTER
    global _DECOMPOSED_ATTENTION_FORECASTER, _FORECASTER
    if decomposition_type is not None:
        _DECOMPOSITION = decomposition_type
    if delay_scores is not None:
        _DELAY_SCORES = delay_scores
    if aggregate is not None:
        _AGGREGATE = aggregate
    if raw_forecaster_type is not None:
        _RAW_FORECASTER = raw_forecaster_type
    if decomposed_attention_forecaster_type is not None:
        _DECOMPOSED_ATTENTION_FORECASTER = decomposed_attention_forecaster_type
    if forecaster_type is not None:
        _FORECASTER = forecaster_type


def _require(component, name):
    if component is None:
        raise RuntimeError(f"Complete and register `{name}` before running this model.")
    return component


SETTINGS = {
    "Raw Attention": dict(mixing="attention", representation="raw"),
    "Raw delay mixer": dict(mixing="delay", representation="raw"),
    "Attention + decomposition": dict(mixing="attention", representation="decomposed"),
    "Autoformer-inspired": dict(mixing="delay", representation="decomposed"),
}


def components_for(settings):
    return dict(
        decomposition_type=(_require(_DECOMPOSITION, "SeriesDecomposition")
                            if settings["representation"] == "decomposed" else None),
        delay_score_fn=(_require(_DELAY_SCORES, "delay_scores")
                        if settings["mixing"] == "delay" else None),
        aggregate_fn=(_require(_AGGREGATE, "aggregate_delays")
                      if settings["mixing"] == "delay" else None),
    )


def _forecaster_for(name):
    if name == "Raw Attention":
        return _RAW_FORECASTER, "RawAttentionForecaster"
    if name == "Attention + decomposition":
        return _DECOMPOSED_ATTENTION_FORECASTER, "DecomposedAttentionForecaster"
    return _FORECASTER, "AutoformerInspiredForecaster"


def build(name="Autoformer-inspired", **kwargs):
    settings = dict(SETTINGS[name])
    settings.update(kwargs)
    forecaster, forecaster_name = _forecaster_for(name)
    if name == "Raw Attention":
        raw_keys = ("context", "horizon", "channels", "d", "heads", "layers", "dropout")
        return _require(forecaster, forecaster_name)(
            **{key: settings[key] for key in raw_keys if key in settings})
    if name == "Attention + decomposition":
        attention_keys = ("context", "horizon", "channels", "d", "heads", "layers", "kernel",
                          "dropout")
        decomposition = _require(_DECOMPOSITION, "SeriesDecomposition")
        return _require(forecaster, forecaster_name)(
            **{key: settings[key] for key in attention_keys if key in settings},
            decomposition_type=decomposition)
    components = components_for(settings)
    return _require(forecaster, forecaster_name)(**settings, **components)
