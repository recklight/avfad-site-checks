"""Pieces every experiment shares: config, matching, the split check, one arm of a comparison."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from voice_severity_bench.comparison import compare_feature_sets
from voice_severity_bench.config import Config
from voice_severity_bench.dataset import FeatureSet, feature_set_from_table
from voice_severity_bench.permutation import permute_group_labels
from voice_severity_bench.splitting import repeated_split_indices
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))

from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402

PLAN_PATH = ROOT / "analysis_plan.yaml"
AGE_BAND_YEARS = 5


class SplitCheckError(RuntimeError):
    """A fold shares a speaker with the rest, or its class shares drift too far."""


def load_plan(path: Path = PLAN_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_config(plan: dict, positive_class: str) -> Config:
    stats = plan["statistics"]
    classifier = plan["detector"]["classifier"]
    return Config().override(
        {
            "seed": stats["seeds"]["cv"],
            "data.task": "classification",
            "data.target": "label",
            "model.name": classifier["name"],
            "model.tune": True,
            "model.class_weight": classifier["class_weight"],
            "evaluation.n_splits": stats["cv"]["folds"],
            "evaluation.n_inner_splits": classifier["inner_splits"],
            "evaluation.n_repeats": stats["cv"]["repeats"],
            "evaluation.positive_class": positive_class,
            # Only the per-arm interval in the summary; the tests draw their own resamples.
            "analysis.n_resamples": 1000,
        }
    )


def age_sex_cell(table: pd.DataFrame, years: int = AGE_BAND_YEARS) -> pd.Series:
    return (table["age"] // years).astype(int).astype(str) + "_" + table["sex"].astype(str)


def match_cells(
    reference: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw from ``pool`` the age by sex distribution of ``reference``.

    A cell with too few speakers is taken whole. Returns the draw and one row per cell
    with how many were wanted and how many were found.
    """
    wanted = age_sex_cell(reference).value_counts().sort_index()
    cells = age_sex_cell(pool)
    picked, rows = [], []
    for cell, n in wanted.items():
        members = pool[cells == cell].sort_values("speaker")
        k = min(int(n), len(members))
        rows.append({"cell": cell, "wanted": int(n), "found": k})
        if k:
            chosen = np.sort(rng.choice(len(members), size=k, replace=False))
            picked.append(members.iloc[chosen])
    empty = pool.iloc[:0]
    return pd.concat(picked) if picked else empty, pd.DataFrame(rows)


def to_feature_set(table: pd.DataFrame) -> FeatureSet:
    return feature_set_from_table(
        table[["path", "group", "label", *FEATURE_COLUMNS]],
        target="label",
        feature_columns=FEATURE_COLUMNS,
        task="classification",
    )


def analysis_table(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per speaker of ``meta`` with its features; every speaker must have them."""
    merged = meta.merge(
        features[["file_id", *FEATURE_COLUMNS]], on="file_id", how="left", validate="one_to_one"
    )
    missing = merged[FEATURE_COLUMNS[0]].isna()
    if missing.any():
        raise ValueError(f"no features for {merged.loc[missing, 'file_id'].tolist()[:5]}")
    merged = merged.assign(
        path=merged["file_id"], group=merged["speaker"], label=merged[target].astype(str)
    )
    return merged.sort_values("file_id").reset_index(drop=True)


def check_folds(folds: list, y: np.ndarray, groups: np.ndarray) -> float:
    """Worst deviation of a class share in one repeat; raises on a broken split."""
    classes = np.unique(y)
    speaker_labels = pd.Series(y).groupby(groups).first()
    overall = {c: float(np.mean(speaker_labels == c)) for c in classes}
    worst = 0.0
    tested: list[np.ndarray] = []
    for train, test in folds:
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise SplitCheckError(f"speakers on both sides of a fold: {sorted(shared)[:5]}")
        tested.append(test)
        in_fold = speaker_labels.loc[np.unique(groups[test])]
        limit = max(0.02, 1.5 / len(in_fold))
        for c in classes:
            deviation = abs(float(np.mean(in_fold == c)) - overall[c])
            if deviation > limit:
                raise SplitCheckError(
                    f"class {c}: share {np.mean(in_fold == c):.3f} against {overall[c]:.3f} "
                    f"in a fold of {len(in_fold)} speakers, limit {limit:.3f}"
                )
            worst = max(worst, deviation)
    together = np.concatenate(tested)
    if np.unique(together).size != y.size or together.size != y.size:
        raise SplitCheckError("the test folds of a repeat do not cover every item once")
    return worst


def check_splits(y: np.ndarray, groups: np.ndarray, config: Config) -> float:
    """Every fold of every repeat; returns the worst class-share deviation."""
    splits = repeated_split_indices(
        y,
        groups,
        task="classification",
        n_splits=config.evaluation.n_splits,
        n_repeats=config.evaluation.n_repeats,
        seed=config.seed,
    )
    return max(check_folds(folds, y, groups) for folds in splits)


def _check_one_permutation(
    child: np.random.SeedSequence,
    y: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray | None,
    config: Config,
) -> float:
    permuted = permute_group_labels(y, groups, strata, np.random.default_rng(child))
    return check_splits(permuted, groups, config)


def check_permuted_splits(
    y: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray | None,
    config: Config,
    n_permutations: int,
    seed: int,
    n_jobs: int = 1,
) -> float:
    """The same check on the labels ``permutation_test`` will draw with this seed."""
    children = np.random.SeedSequence(seed).spawn(n_permutations)
    worst = Parallel(n_jobs=n_jobs)(
        delayed(_check_one_permutation)(child, y, groups, strata, config) for child in children
    )
    return float(max(worst))


@dataclass(frozen=True, eq=False)
class Arm:
    """One feature table through the repeated CV."""

    name: str
    y_true: np.ndarray
    groups: np.ndarray
    predictions: np.ndarray  # (n_speakers, n_repeats) class names
    scores: np.ndarray  # (n_speakers, n_repeats) positive-class score
    uar: float
    roc_auc: float
    sensitivity: float
    specificity: float
    worst_share_deviation: float
    chosen_c: dict[str, int]

    def row(self) -> dict:
        return {
            "arm": self.name,
            "n_speakers": int(self.y_true.size),
            **{str(c): int(np.sum(self.y_true == c)) for c in np.unique(self.y_true)},
            "uar": self.uar,
            "roc_auc": self.roc_auc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "worst_share_deviation": self.worst_share_deviation,
            "chosen_c": json.dumps(self.chosen_c, sort_keys=True),
        }


def evaluate_arm(name: str, table: pd.DataFrame, config: Config) -> Arm:
    features = to_feature_set(table)
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    worst = check_splits(y, groups, config)
    result = compare_feature_sets({name: features}, config, n_jobs=1)
    summary = result.summary.iloc[0]
    chosen = Counter(json.loads(p)["model__C"] for p in result.folds["params"])
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
        chosen_c={str(k): int(v) for k, v in sorted(chosen.items())},
    )


def predictions_table(arms: list[Arm]) -> pd.DataFrame:
    """Long table of the out-of-fold predictions, one row per arm, speaker and repeat."""
    parts = []
    for arm in arms:
        n, repeats = arm.predictions.shape
        parts.append(
            pd.DataFrame(
                {
                    "arm": arm.name,
                    "speaker": np.repeat(arm.groups, repeats),
                    "label": np.repeat(arm.y_true, repeats),
                    "repeat": np.tile(np.arange(repeats), n),
                    "predicted": arm.predictions.ravel(),
                    "score": arm.scores.ravel(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
