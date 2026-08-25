"""Hydra entry point for one pretrained-model evaluation."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from evaluation import evaluate


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    plain = OmegaConf.to_container(config, resolve=True)
    level = str(plain.get("misc", {}).get("log_level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")
    evaluate(plain)


if __name__ == "__main__":
    main()
