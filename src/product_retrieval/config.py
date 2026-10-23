"""Experiment configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    required = {"model", "retrieval", "verification", "paths"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(sorted(missing))}")
    return config


def make_encoder(config: dict):
    from product_retrieval.models import DinoV2Encoder

    model = config["model"]
    preprocessing = config.get("preprocessing", {})
    return DinoV2Encoder(
        model_name=model.get("name", "dinov2_vits14"),
        image_size=int(model.get("image_size", 224)),
        device=model.get("device", "auto"),
        feature_type=model.get("feature_type", "cls"),
        background_color=tuple(preprocessing.get("background_color", (255, 255, 255))),
        interpolation=preprocessing.get("interpolation", "bicubic"),
    )


def make_retriever(config: dict, gallery):
    from product_retrieval.retrieval import ProductRetriever

    retrieval = config["retrieval"]
    return ProductRetriever(
        gallery,
        aggregation=retrieval.get("aggregation", "top_mean"),
        references_per_code=int(retrieval.get("references_per_code", 3)),
    )

