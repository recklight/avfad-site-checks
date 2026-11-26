"""Arms on any feature columns, for the sensitivity and exploratory runs.

``harness.evaluate_arm`` is under the first freeze and fixed to the 78 handcrafted columns.
These do the same with other columns, and with the HuBERT layers as blocks the inner loop
chooses from.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from voice_severity_bench import models as vsb_models
from voice_severity_bench.comparison import compare_feature_sets
from voice_severity_bench.dataset import feature_set_from_table
from harness import ROOT, Arm, check_splits

HUBERT_DIR = ROOT / "features" / "out_hubert"
# 1536 columns against a few hundred speakers: the grid of the frozen detector stops at
# C = 0.01, which can be too little regularisation here, so it goes two decades lower.
HUBERT_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALL_LAYERS = tuple(range(13))


def layer_columns(layer: int) -> list[str]:
    return [f"l{layer:02d}_{i:04d}" for i in range(1536)]


def load_hubert(condition: str, layers: Sequence[int], material: str = "avfad_a") -> pd.DataFrame:
    """One row per file with the mean and std columns of the chosen layers."""
    data = np.load(HUBERT_DIR / f"{material}_{condition}.npz")
    pooled = data["pooled"]
    columns = [c for layer in layers for c in layer_columns(layer)]
    values = np.concatenate([pooled[:, layer, :] for layer in layers], axis=1)
    table = pd.DataFrame(values.astype(np.float64), columns=columns)
    table.insert(0, "file_id", data["file_id"].astype(str))
    return table


def merge_features(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> pd.DataFrame:
    merged = meta.merge(features, on="file_id", how="left", validate="one_to_one")
    missing = merged[features.columns[1]].isna()
    if missing.any():
        raise ValueError(f"no features for {merged.loc[missing, 'file_id'].tolist()[:5]}")
    merged = merged.assign(
        path=merged["file_id"], group=merged["speaker"], label=merged[target].astype(str)
    )
    return merged.sort_values("file_id").reset_index(drop=True)


def _use_hubert_grid() -> None:
    # Set in the process that fits: joblib workers import the library afresh.
    vsb_models.SEARCH_SPACES[("classification", "logistic")] = {"model__C": list(HUBERT_C_GRID)}


def _arm_from(name, result, worst, key) -> Arm:
    summary = result.summary.iloc[0]
    chosen = Counter(str(json.loads(p).get(key)) if key else "none" for p in result.folds["params"])
    return Arm(
        name=name,
        y_true=result.y_true,
        groups=result.groups,
        predictions=result.prediction_matrix(name),
        scores=result.score_matrix(name),
        uar=float(summary["uar"]),
        roc_auc=float(summary["roc_auc"]),
        sensitivity=float(summary["sensitivity"]),
        specificity=float(summary["specificity"]),
        worst_share_deviation=worst,
        chosen_c=dict(sorted(chosen.items())),
    )


def evaluate_columns(name: str, table: pd.DataFrame, columns: Sequence[str], config) -> Arm:
    """The repeated CV on the given columns with the frozen detector's grid."""
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        feature_columns=list(columns),
        task="classification",
    )
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    worst = check_splits(y, groups, config)
    result = compare_feature_sets({name: features}, config, n_jobs=1)
    return _arm_from(name, result, worst, "model__C")


def evaluate_layers(name: str, table: pd.DataFrame, layers: Sequence[int], config) -> Arm:
    """HuBERT: the inner loop chooses the layer and C; a single layer just chooses C."""
    _use_hubert_grid()
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in layers}
    columns = [c for block in blocks.values() for c in block]
    many = len(blocks) > 1
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        feature_columns=None if many else columns,
        blocks=blocks if many else None,
        task="classification",
    )
    tuned = config.override({"model.block_selection": True}) if many else config
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    worst = check_splits(y, groups, config)
    result = compare_feature_sets({name: features}, tuned, n_jobs=1)
    arm = _arm_from(name, result, worst, "model__C")
    chosen_layers = Counter(str(json.loads(p).get("select__block")) for p in result.folds["params"])
    return Arm(
        **{
            **arm.__dict__,
            "chosen_c": {"C": arm.chosen_c, "layer": dict(sorted(chosen_layers.items()))},
        }
    )


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
