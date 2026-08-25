"""Foundation-model construction and optional reversible instance scaling."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .baselines import (
    ExpectedBaseline,
    LookbackBaseline,
    PersistenceBaseline,
    RepeatBaseline,
)
from external_models import (
    Chronos2,
    ChronosBolt,
    TabPFNTS,
    TiRex2Forecaster,
    TSICLForecaster,
)


FOUNDATION_MODEL_ALIASES = (
    "chronos2",
    "chronos_bolt",
    "ts_icl",
    "tirex2",
    "tabpfn_ts",
)


class ForecastModel(nn.Module):
    """Apply target-only instance normalization around a frozen forecaster."""

    def __init__(self, base: nn.Module, *, instance_normalize: bool, eps: float):
        super().__init__()
        self.base = base
        self.instance_normalize = bool(instance_normalize)
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_input = x
        mean = std = None
        if self.instance_normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            model_input = (x - mean) / (std + self.eps)
        covariates = None
        if past_covariates is not None or future_covariates is not None:
            covariates = {"past": past_covariates, "future": future_covariates}
        prediction = self.base(model_input, covariates=covariates)
        return prediction if mean is None else prediction * (std + self.eps) + mean


def _plain_dict(value: Any) -> dict[str, Any]:
    return {} if value is None else dict(value)


def build_forecaster(
    model_config: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    *,
    lags: int,
    horizon: int,
) -> ForecastModel:
    """Build exactly one non-trainable evaluation model."""
    raw_name = str(model_config.get("name", "repeat"))
    name = raw_name.lower()
    if name in FOUNDATION_MODEL_ALIASES and raw_name != name:
        raise ValueError(f"foundation model aliases are case-sensitive: {raw_name!r}")
    weights_path = model_config.get("weights_path")
    device = str(model_config.get("device", "cuda"))
    if name == "persistence":
        base: nn.Module = PersistenceBaseline(horizon)
    elif name == "expected":
        base = ExpectedBaseline(horizon)
    elif name == "repeat":
        base: nn.Module = RepeatBaseline(horizon)
    elif name == "lookback":
        period = model_config.get("lookback_period")
        if period is None:
            raise ValueError("lookback baseline requires a resolved model.lookback_period")
        base = LookbackBaseline(horizon, int(period))
    elif name == "chronos2":
        base = Chronos2(
            lags=lags,
            dim=1,
            horizon=horizon,
            weights_path=weights_path,
            device=device,
            local_files_only=bool(model_config.get("local_files_only", True)),
            cross_learning=bool(model_config.get("cross_learning", False)),
            quantile_index=model_config.get("quantile_index"),
        )
    elif name == "chronos_bolt":
        base = ChronosBolt(
            lags,
            horizon=horizon,
            weights_path=weights_path,
            device=device,
            local_files_only=bool(model_config.get("local_files_only", True)),
            quantile_level=float(model_config.get("quantile_level", 0.5)),
        )
    elif name == "ts_icl":
        base = TSICLForecaster(
            lags=lags,
            dim=1,
            horizon=horizon,
            weights_path=weights_path,
            device=device,
            local_files_only=bool(model_config.get("local_files_only", True)),
            quantile_level=float(model_config.get("quantile_level", 0.5)),
        )
    elif name == "tirex2":
        base = TiRex2Forecaster(
            lags=lags,
            dim=1,
            horizon=horizon,
            weights_path=weights_path,
            device=device,
            quantile_level=float(model_config.get("quantile_level", 0.5)),
        )
    elif name == "tabpfn_ts":
        base = TabPFNTS(
            lags=lags,
            dim=1,
            horizon=horizon,
            weights_path=weights_path,
            device=device,
            seasonal_periods=model_config.get("seasonal_periods") or [24, 168],
            use_time_features=bool(model_config.get("use_time_features", True)),
            **_plain_dict(model_config.get("kwargs")),
        )
    else:
        raise ValueError(f"unknown inference model {name!r}")
    return ForecastModel(
        base,
        instance_normalize=bool(preprocessing.get("instance_normalize", True)),
        eps=float(preprocessing.get("eps", 1e-8)),
    )
