from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

import external_models.chronos2 as chronos_module
import external_models.chronos_bolt as chronos_bolt_module
import external_models.chronos_t5 as chronos_t5_module
import external_models.tabpfn as tabpfn_module
import external_models.ts_icl as tsicl_module
from evaluation.evaluator import _weekly_period_steps
from model_loading import build_forecaster
from model_loading.forecast import FOUNDATION_MODEL_ALIASES
from model_loading.baselines import (
    ExpectedBaseline,
    LookbackBaseline,
    PersistenceBaseline,
    RepeatBaseline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fits = []

    def fit(self, features, targets):
        self.fits.append((features, targets))
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=np.float32)


class FakeChronosPipeline:
    def __init__(self):
        self.inputs = None

    def predict(self, *, inputs, prediction_length, cross_learning):
        self.inputs = inputs
        assert cross_learning is False
        return [torch.ones(1, 3, prediction_length) for _ in inputs]


class FakeChronosBase:
    pipeline = FakeChronosPipeline()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls.pipeline


class FakeChronosBoltPipeline:
    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
        assert quantile_levels == [0.5]
        return None, torch.ones(inputs.shape[0], prediction_length)


class FakeChronosBoltBase:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeChronosBoltPipeline()


class FakeChronosT5Pipeline:
    def predict(self, inputs, *, prediction_length, num_samples):
        assert num_samples == 5
        return torch.ones(inputs.shape[0], num_samples, prediction_length)


class FakeChronosT5Base:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeChronosT5Pipeline()


class FakeTSICL:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.forecast_kwargs = None
        FakeTSICL.instance = self

    def forecast(self, **kwargs):
        self.forecast_kwargs = kwargs
        inputs = kwargs["inputs"]
        horizon = kwargs["prediction_length"]
        point = torch.full(
            (inputs.shape[0], inputs.shape[2], horizon, 1),
            2.0,
            device=inputs.device,
        )
        return point, point.clone()


def test_legacy_baselines() -> None:
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    assert torch.equal(PersistenceBaseline(2)(x), torch.tensor([[[7.0, 7.0]]]))
    assert torch.equal(ExpectedBaseline(2)(x), torch.tensor([[[3.5, 3.5]]]))
    assert torch.equal(RepeatBaseline(2)(x), torch.tensor([[[6.0, 7.0]]]))
    weekly = torch.arange(336, dtype=torch.float32).view(1, 1, 336)
    assert torch.equal(LookbackBaseline(24, 168)(weekly), weekly[..., 168:192])
    long_horizon = torch.arange(504, dtype=torch.float32).view(1, 1, 504)
    assert torch.equal(
        LookbackBaseline(200, 168)(long_horizon),
        long_horizon[..., 168:368],
    )


def test_weekly_period_resolution_and_removed_indexed_names() -> None:
    hourly = np.datetime64("2026-01-01") + np.arange(10).astype("timedelta64[h]")
    daily = np.datetime64("2026-01-01") + np.arange(10).astype("timedelta64[D]")
    assert _weekly_period_steps(hourly) == 168
    assert _weekly_period_steps(daily) == 7
    preprocessing = {"instance_normalize": False, "eps": 1e-8}
    model = build_forecaster(
        {"name": "lookback", "lookback_period": 168},
        preprocessing,
        lags=336,
        horizon=24,
    )
    x = torch.arange(336, dtype=torch.float32).view(1, 1, 336)
    assert torch.equal(model(x), x[..., 168:192])
    for removed_name in ("lookback0", "lookback168"):
        try:
            build_forecaster(
                {"name": removed_name},
                preprocessing,
                lags=336,
                horizon=24,
            )
        except ValueError as exc:
            assert "unknown inference model" in str(exc)
        else:
            raise AssertionError(f"removed model name {removed_name} was accepted")


def test_tabpfn_feature_protocol() -> None:
    original = tabpfn_module._import_regressor
    tabpfn_module._import_regressor = lambda: FakeRegressor
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            checkpoint = Path(directory) / "weights.ckpt"
            checkpoint.touch()
            model = tabpfn_module.TabPFNTS(
                lags=4,
                dim=1,
                horizon=2,
                weights_path=checkpoint,
                device="cpu",
                seasonal_periods=[2, 4],
            )
            prediction = model(torch.arange(4, dtype=torch.float32).view(1, 1, 4))
            assert prediction.shape == (1, 1, 2)
            features, targets = model.model.fits[0]
            assert features.shape == (4, 5)
            assert targets.shape == (4,)

            covariate_only = tabpfn_module.TabPFNTS(
                lags=4,
                dim=1,
                horizon=2,
                weights_path=checkpoint,
                device="cpu",
                use_time_features=False,
            )
            context = {
                "past": torch.ones(1, 2, 4),
                "future": torch.ones(1, 2, 2),
            }
            covariate_only(torch.ones(1, 1, 4), context)
            assert covariate_only.model.fits[0][0].shape == (4, 2)
    finally:
        tabpfn_module._import_regressor = original


def test_chronos_structured_covariates() -> None:
    original = chronos_module._import_pipeline
    chronos_module._import_pipeline = lambda: FakeChronosBase
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            model = chronos_module.Chronos2(
                lags=4,
                dim=1,
                horizon=2,
                weights_path=directory,
                device="cpu",
            )
            context = {
                "past": torch.ones(1, 2, 4),
                "future": torch.ones(1, 2, 2),
            }
            prediction = model(torch.zeros(1, 1, 4), context)
            assert prediction.shape == (1, 1, 2)
            item = FakeChronosBase.pipeline.inputs[0]
            assert set(item["past_covariates"]) == {"covariate_0", "covariate_1"}
            assert set(item["future_covariates"]) == {"covariate_0", "covariate_1"}
    finally:
        chronos_module._import_pipeline = original


def test_chronos_bolt_official_pipeline_protocol() -> None:
    original = chronos_bolt_module._import_pipeline
    chronos_bolt_module._import_pipeline = lambda: FakeChronosBoltBase
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            model = chronos_bolt_module.ChronosBolt(
                4,
                horizon=2,
                weights_path=directory,
                device="cpu",
            )
            prediction = model(torch.zeros(2, 1, 4))
            assert prediction.shape == (2, 1, 2)
            assert torch.equal(prediction, torch.ones(2, 1, 2))
            for kwargs in (
                {"covariates": {"past": torch.ones(2, 1, 4), "future": None}},
                {"past_covariates": torch.ones(2, 1, 4)},
            ):
                try:
                    model(torch.zeros(2, 1, 4), **kwargs)
                except ValueError as error:
                    assert "does not consume covariates" in str(error)
                else:
                    raise AssertionError("Chronos-Bolt must reject covariates")
    finally:
        chronos_bolt_module._import_pipeline = original


def test_chronos_t5_official_pipeline_protocol() -> None:
    original = chronos_t5_module._import_pipeline
    chronos_t5_module._import_pipeline = lambda: FakeChronosT5Base
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            model = chronos_t5_module.ChronosT5(
                4,
                horizon=2,
                weights_path=directory,
                device="cpu",
                num_samples=5,
            )
            prediction = model(torch.zeros(2, 1, 4))
            assert torch.equal(prediction, torch.ones(2, 1, 2))
            try:
                model(
                    torch.zeros(2, 1, 4),
                    past_covariates=torch.ones(2, 1, 4),
                )
            except ValueError as error:
                assert "does not consume covariates" in str(error)
            else:
                raise AssertionError("Chronos-T5 must reject covariates")
    finally:
        chronos_t5_module._import_pipeline = original


def test_foundation_aliases_are_unique() -> None:
    assert FOUNDATION_MODEL_ALIASES == (
        "chronos2",
        "chronos_bolt",
        "chronos_t5",
        "ts_icl",
    )
    for removed in (
        "chronos",
        "chronos-2",
        "chronos-bolt",
        "chronos-t5",
        "tsicl",
        "ts-icl",
        "tirex2",
        "tirex_2",
        "tirex-2",
        "tyrex2",
        "tabpfn_ts",
        "tabpfn",
        "tabpfn-ts",
    ):
        try:
            build_forecaster(
                {"name": removed, "device": "cpu"},
                {"instance_normalize": False},
                lags=4,
                horizon=2,
            )
        except ValueError as error:
            assert "unknown inference model" in str(error)
        else:
            raise AssertionError(f"removed alias {removed!r} must be rejected")


def test_tsicl_official_forecast_protocol() -> None:
    original = tsicl_module._import_tsicl
    tsicl_module._import_tsicl = lambda: FakeTSICL
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            checkpoint = Path(directory) / "tsicl-v1.ckpt"
            checkpoint.touch()
            model = tsicl_module.TSICLForecaster(
                lags=4,
                dim=1,
                horizon=2,
                weights_path=checkpoint,
                device="cpu",
            )
            context = {
                "past": torch.ones(2, 2, 4),
                "future": torch.ones(2, 2, 2),
            }
            prediction = model(torch.zeros(2, 1, 4), context)
            assert prediction.shape == (2, 1, 2)
            assert torch.equal(prediction, torch.full((2, 1, 2), 2.0))
            call = FakeTSICL.instance.forecast_kwargs
            assert call["inputs"].shape == (2, 4, 1)
            assert call["covars"].shape == (2, 6, 2)
            assert call["quantile_levels"] == [0.5]
            assert call["point_estimator"] == "median"
            assert call["denormalize"] is True
            assert call["allow_auto_complete"] is False
            assert call["allow_covar_forecast"] is False
            assert call["squeeze_output"] is False
    finally:
        tsicl_module._import_tsicl = original


if __name__ == "__main__":
    test_legacy_baselines()
    test_weekly_period_resolution_and_removed_indexed_names()
    test_tabpfn_feature_protocol()
    test_chronos_structured_covariates()
    test_chronos_bolt_official_pipeline_protocol()
    test_chronos_t5_official_pipeline_protocol()
    test_foundation_aliases_are_unique()
    test_tsicl_official_forecast_protocol()
