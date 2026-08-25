"""Non-trainable forecasting baselines from the original TimeTensor project."""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceBaseline(nn.Module):
    """Repeat the last observed value over the forecast horizon."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor, covariates=None) -> torch.Tensor:
        del covariates
        return x[..., -1:].repeat_interleave(self.horizon, dim=-1)


class ExpectedBaseline(nn.Module):
    """Repeat the lookback mean over the forecast horizon."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor, covariates=None) -> torch.Tensor:
        del covariates
        return x.mean(dim=-1, keepdim=True).repeat_interleave(self.horizon, dim=-1)


class RepeatBaseline(nn.Module):
    """Repeat the final horizon-sized segment of the lookback."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor, covariates=None) -> torch.Tensor:
        del covariates
        if x.shape[-1] < self.horizon:
            raise ValueError("repeat baseline requires lags >= horizon")
        return x[..., -self.horizon :]


class LookbackBaseline(nn.Module):
    """Return the latest periodically aligned, non-overlapping history block."""

    def __init__(self, horizon: int, period: int):
        super().__init__()
        self.horizon = int(horizon)
        self.period = int(period)
        if self.period < 1:
            raise ValueError("lookback period must be positive")

    def forward(self, x: torch.Tensor, covariates=None) -> torch.Tensor:
        del covariates
        periods_back = (self.horizon + self.period - 1) // self.period
        start = x.shape[-1] - periods_back * self.period
        stop = start + self.horizon
        if start < 0:
            raise ValueError(
                f"length-{x.shape[-1]} input cannot hold a horizon-{self.horizon} "
                f"window aligned {periods_back} period(s) of {self.period} steps back"
            )
        return x[..., start:stop]
