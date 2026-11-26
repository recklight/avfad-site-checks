"""The C1 gap seen by a detector that never trained on AVFAD, and what the AVFAD one learned.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before the outside gap on the C1 draws was computed; /i/ and /u/ were added after /a/, before
their outside detectors were trained. The inside scores are the stored HuBERT C1 test folds; the
outside ones come from HuBERT trained on the same vowel of SVD.

    score i|u     train on SVD i_n or u_n once, score every AVFAD speaker's /i/ or /u/
    analyse a|i|u writes results/outside_view/summary.json (/a/) or summary_{i,u}.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/outside_view.py analyse a
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
from arms import write_json  # noqa: E402

RESULTS = HERE / "results" / "outside_view"
INSIDE = {
    "a": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
    "i": HERE / "results" / "vowel_fusion" / "predictions_i.csv.gz",
    "u": HERE / "results" / "vowel_fusion" / "predictions_u.csv.gz",
}
OUTSIDE = {
    "a": HERE / "results" / "cross_database" / "reverse_scores_hubert.csv.gz",
    "i": RESULTS / "outside_scores_i.csv.gz",
    "u": RESULTS / "outside_scores_u.csv.gz",
}
SEEDS = {"a": 20261123, "i": 20261124, "u": 20261125}
HANDCRAFTED_INSIDE = HERE / "results" / "first_batch" / "c1_predictions.csv.gz"
HANDCRAFTED_SEED = 20261127
ERASED_INSIDE = HERE / "results" / "erasure_folds" / "c1_predictions.csv.gz"
ERASED_SEED = 20261131
# The outside detector trained and scoring with the AVFAD place direction taken out (added later).
OUTSIDE_ERASED = {v: RESULTS / f"outside_scores_{v}_erased.csv.gz" for v in ("a", "i", "u")}
OUTSIDE_ERASED_SEEDS = {"a": 20261134, "i": 20261137, "u": 20261138}
# Phonation spans only, on both sides (added later).
OUTSIDE_PHONATION = RESULTS / "outside_scores_a_phonation.csv.gz"
INSIDE_PHONATION = HERE / "results" / "phonation_c1" / "c1_predictions_hubert.csv.gz"
PHONATION_SEED = 20261136
SVD_SPEAKERS = 1457
N_RESAMPLES = 10_000
# Readings, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10
RANK_POINTS = 0.10


def spearman_weighted(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Spearman correlation per row of ``weights`` (counts), ranks taken once on all patients.

    With repeated patients the ranks are not recomputed; the weighted Pearson correlation of
    the full-sample ranks is close and keeps 10,000 resamples cheap.
    """
    rx, ry = rankdata(x), rankdata(y)
    w = np.atleast_2d(weights)
    total = w.sum(axis=1, keepdims=True)
    mx = (w * rx).sum(axis=1, keepdims=True) / total
    my = (w * ry).sum(axis=1, keepdims=True) / total
    dx, dy = rx[None, :] - mx, ry[None, :] - my
    cov = (w * dx * dy).sum(axis=1)
    return cov / np.sqrt((w * dx**2).sum(axis=1) * (w * dy**2).sum(axis=1))


def gap_reading(gap: dict) -> str:
    if gap["ci_high"] < 0:
        return "the (i) pairing is harder for the outside detector"
    if gap["ci_low"] <= 0:
        return "the outside detector sees no difference between the two sets of normals"
    if gap["difference"] >= GAP_POINTS:
        return "the test populations differ this much"
    return "the test populations differ a little"


def learned_reading(learned: dict) -> str:
    if learned["difference"] >= GAP_POINTS and learned["ci_low"] > 0:
        return "most of the gap is learned in training"
    return "as measured"


def rank_reading(diff: dict) -> str:
    if diff["ci_high"] < 0:
        return "(i) agrees less"
    if diff["ci_low"] <= 0:
        return "the same"
    if diff["difference"] >= RANK_POINTS:
        return "(i) ranks patients more like the outside detector"
    return "small"


def interval(point: float, samples: np.ndarray, key: str = "difference") -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {key: float(point), "ci_low": float(low), "ci_high": float(high)}


def gap_of(
    scores: dict[str, pd.Series], patients, hsa_normals, speakers, w_pos, w_normal
) -> tuple[dict, np.ndarray]:
    """(i) minus (ii) AUC for one detector, with its resampled differences."""
    ii = scores["ii"]
    kernel = db.comparisons(ii.loc[patients].to_numpy(), ii.loc[hsa_normals].to_numpy())
    ii_point = kernel.mean()
    ii_boot = db.weighted_auc(kernel, w_pos, w_normal.loc[:, hsa_normals].to_numpy())
    draws = sorted(a for a in scores if a.startswith("i_"))
    points, boot = [], np.zeros(w_pos.shape[0])
    for name in draws:
        arm = scores[name]
        normals = arm.index[speakers.loc[arm.index, "label"] == "normal"]
        kernel = db.comparisons(arm.loc[patients].to_numpy(), arm.loc[normals].to_numpy())
        points.append(kernel.mean())
        boot += db.weighted_auc(kernel, w_pos, w_normal.loc[:, normals].to_numpy())
    i_point, i_boot = float(np.mean(points)), boot / len(draws)
    record = {
        "ii": interval(ii_point, ii_boot, "auc"),
        "i": interval(i_point, i_boot, "auc") | {"draw_range": [min(points), max(points)]},
        "gap": interval(i_point - ii_point, i_boot - ii_boot),
    }
    return record, i_boot - ii_boot


def score_outside(vowel: str) -> None:
    """Train on the same vowel of SVD once and score every AVFAD speaker's file."""
    import cross_database as cd
    import cross_vowels as cv
    import run_first_batch as batch
    from arms import layer_columns, load_hubert
    from harness import load_plan, make_config
    from sensitivity_avfad import CHOSEN_LAYERS

    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    material = cv.SVD_MATERIAL[vowel]
    sessions = cv.svd_sessions()
    sessions = sessions.assign(file_id=[f"svd__{s}-{material}" for s in sessions["session"]])
    train = sessions.merge(
        load_hubert("full", CHOSEN_LAYERS, material=f"svd_{material}"),
        on="file_id",
        how="inner",
        validate="one_to_one",
    )
    if len(train) != SVD_SPEAKERS:
        raise RuntimeError(f"{len(train)} SVD speakers, expected {SVD_SPEAKERS}")
    speakers = batch.load_speakers()
    test = speakers.assign(file_id=[cv.avfad_file_id(s, vowel) for s in speakers["speaker"]])
    test = test.merge(
        load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{vowel}"),
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    if test[columns[0]].isna().any():
        raise ValueError("an AVFAD speaker has no features")
    scores = cd.score_on_svd(train, test, columns, blocks, config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "place", "label", "age", "sex"]].assign(score=scores)
    frame.to_csv(OUTSIDE[vowel], index=False)


def erased_reading(diff: dict) -> str:
    if diff["difference"] >= RANK_POINTS and diff["ci_low"] > 0:
        return "with the place direction out, (i) ranks patients more like the outside detector"
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return "the ranking does not change"
    return "as measured"


def score_outside_phonation() -> None:
    """SVD a_n phonation spans train once; every AVFAD speaker's /a/ phonation spans are scored."""
    import cross_database as cd
    import run_first_batch as batch
    from arms import layer_columns, load_hubert
    from harness import load_plan, make_config
    from sensitivity_avfad import CHOSEN_LAYERS

    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    svd_features = load_hubert("phonation", CHOSEN_LAYERS, material="svd_a_n")
    train = cd.svd_test_set().merge(svd_features, on="file_id", how="left", validate="one_to_one")
    test = batch.load_speakers().merge(
        load_hubert("phonation", CHOSEN_LAYERS), on="file_id", how="left", validate="one_to_one"
    )
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    if train[columns[0]].isna().any() or test[columns[0]].isna().any():
        raise ValueError("a speaker has no features")
    scores = cd.score_on_svd(train, test, columns, blocks, config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "place", "label", "age", "sex"]].assign(score=scores)
    frame.to_csv(OUTSIDE_PHONATION, index=False)


def score_outside_erased(vowel: str = "a") -> None:
    """SVD trains once, AVFAD is scored, same vowel; both with the place direction projected out."""
    import cross_database as cd
    import cross_vowels as cv
    import run_first_batch as batch
    import site_erasure as se
    import vowel_fusion as vf
    from arms import layer_columns, load_hubert
    from harness import load_plan, make_config
    from sensitivity_avfad import CHOSEN_LAYERS

    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    speakers = batch.load_speakers()
    if vowel == "a":
        avfad, svd_features, _ = se.features()
        train = cd.svd_test_set().merge(
            svd_features, on="file_id", how="left", validate="one_to_one"
        )
        test = speakers.merge(avfad, on="file_id", how="left", validate="one_to_one")
    else:
        avfad = load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{vowel}")
        normals = vf.with_vowel(se.normal_speakers(), vowel).merge(
            avfad, on="file_id", how="left", validate="one_to_one"
        )
        essua = (normals["place"] == "ESSUA").to_numpy()
        directions = se.place_directions(se.layer_blocks(normals), essua, ~essua)
        material = cv.SVD_MATERIAL[vowel]
        svd_features = se.erased(
            load_hubert("full", CHOSEN_LAYERS, material=f"svd_{material}"), directions
        )
        sessions = cv.svd_sessions()
        sessions = sessions.assign(file_id=[f"svd__{s}-{material}" for s in sessions["session"]])
        train = sessions.merge(svd_features, on="file_id", how="inner", validate="one_to_one")
        if len(train) != SVD_SPEAKERS:
            raise RuntimeError(f"{len(train)} SVD speakers, expected {SVD_SPEAKERS}")
        test = vf.with_vowel(speakers, vowel).merge(
            se.erased(avfad, directions), on="file_id", how="left", validate="one_to_one"
        )
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    if train[columns[0]].isna().any() or test[columns[0]].isna().any():
        raise ValueError("a speaker has no features")
    scores = cd.score_on_svd(train, test, columns, blocks, config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "place", "label", "age", "sex"]].assign(score=scores)
    frame.to_csv(OUTSIDE_ERASED[vowel], index=False)


def analyse(
    vowel: str, kind: str = "hubert", outside_kind: str = "plain", phonation: bool = False
) -> None:
    speakers = db.load_speakers()
    hsa = speakers["place"] == "HSA"
    patients = speakers.index[hsa & (speakers["label"] == "pathological")]
    hsa_normals = speakers.index[hsa & (speakers["label"] == "normal")]
    path = {"hubert": INSIDE[vowel], "handcrafted": HANDCRAFTED_INSIDE, "erased": ERASED_INSIDE}
    inside, _ = db.speaker_means(INSIDE_PHONATION if phonation else path[kind])
    outside_path = OUTSIDE_ERASED[vowel] if outside_kind == "erased" else OUTSIDE[vowel]
    if phonation:
        outside_path = OUTSIDE_PHONATION
    outside_scores = pd.read_csv(outside_path, dtype={"speaker": str}).set_index("speaker")["score"]
    draws = sorted(a for a in inside if a.startswith("i_"))
    # The outside detector scores everyone once, so every arm reads the same scores.
    outside = {name: outside_scores.loc[inside[name].index] for name in ("ii", *draws)}

    seed = {"hubert": SEEDS[vowel], "handcrafted": HANDCRAFTED_SEED, "erased": ERASED_SEED}[kind]
    if outside_kind == "erased":
        seed = OUTSIDE_ERASED_SEEDS[vowel]
    if phonation:
        seed = PHONATION_SEED
    rng = np.random.default_rng(seed)
    w_pos = db.resample_counts(len(patients), N_RESAMPLES, rng)
    parts = []
    for at_hsa in (True, False):
        members = speakers.index[
            (speakers["label"] == "normal") & ((speakers["place"] == "HSA") == at_hsa)
        ]
        parts.append(
            pd.DataFrame(db.resample_counts(len(members), N_RESAMPLES, rng), columns=members)
        )
    w_normal = pd.concat(parts, axis=1)

    out_record, out_samples = gap_of(outside, patients, hsa_normals, speakers, w_pos, w_normal)
    in_record, in_samples = gap_of(inside, patients, hsa_normals, speakers, w_pos, w_normal)
    learned = interval(
        in_record["gap"]["difference"] - out_record["gap"]["difference"], in_samples - out_samples
    )

    # Ranking among HSA patients; new resamples of the patients, after the ones above.
    w_rank = db.resample_counts(len(patients), N_RESAMPLES, rng)
    reference = outside_scores.loc[patients].to_numpy()
    ii_scores = inside["ii"].loc[patients].to_numpy()
    rho_ii = float(spearman_weighted(reference, ii_scores, np.ones(len(patients)))[0])
    boot_ii = spearman_weighted(reference, ii_scores, w_rank)
    rho_i, boot_i = [], np.zeros(N_RESAMPLES)
    for name in draws:
        values = inside[name].loc[patients].to_numpy()
        rho_i.append(float(spearman_weighted(reference, values, np.ones(len(patients)))[0]))
        boot_i += spearman_weighted(reference, values, w_rank)
    boot_i /= len(draws)
    rank = {
        "rho_ii": interval(rho_ii, boot_ii, "rho"),
        "rho_i": interval(float(np.mean(rho_i)), boot_i, "rho")
        | {"draw_range": [min(rho_i), max(rho_i)]},
        "i_minus_ii": interval(float(np.mean(rho_i)) - rho_ii, boot_i - boot_ii),
    }
    if kind == "erased":
        # The same patients' weights, now on the plain (i) scores.
        plain, _ = db.speaker_means(INSIDE["a"])
        rho_plain, boot_plain = [], np.zeros(N_RESAMPLES)
        for name in draws:
            values = plain[name].loc[patients].to_numpy()
            rho_plain.append(float(spearman_weighted(reference, values, np.ones(len(patients)))[0]))
            boot_plain += spearman_weighted(reference, values, w_rank)
        boot_plain /= len(draws)
        rank["rho_i_plain"] = interval(float(np.mean(rho_plain)), boot_plain, "rho")
        rank["i_erased_minus_i_plain"] = interval(
            float(np.mean(rho_i)) - float(np.mean(rho_plain)), boot_i - boot_plain
        )

    summary = {
        "vowel": vowel,
        "inside_detector": kind,
        "outside_detector": outside_kind,
        "phonation_only": phonation,
        "seed": seed,
        "n_resamples": N_RESAMPLES,
        "outside": out_record,
        "inside": in_record,
        "learned": learned,
        "ranking_among_hsa_patients": rank,
        "readings": {
            "outside_gap": gap_reading(out_record["gap"]),
            "learned": learned_reading(learned),
            "ranking": rank_reading(rank["i_minus_ii"]),
        },
    }
    if kind == "erased":
        summary["readings"]["erased_vs_plain"] = erased_reading(rank["i_erased_minus_i_plain"])
    essua_normals = speakers.index[(speakers["place"] == "ESSUA") & (speakers["label"] == "normal")]
    summary["outside_hsa_normals_vs_essua_normals"] = float(
        db.comparisons(
            outside_scores.loc[hsa_normals].to_numpy(), outside_scores.loc[essua_normals].to_numpy()
        ).mean()
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    if phonation:
        name = f"summary_{vowel}_phonation"
    elif outside_kind == "erased":
        name = f"summary_{vowel}_outside_erased"
    elif kind != "hubert":
        name = f"summary_{vowel}_{kind}"
    else:
        name = "summary" if vowel == "a" else f"summary_{vowel}"
    write_json(RESULTS / f"{name}.json", summary)
    print(vowel, summary["readings"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("score", "score-erased", "analyse"))
    parser.add_argument("vowel", choices=("a", "i", "u"))
    parser.add_argument("--inside", choices=("hubert", "handcrafted", "erased"), default="hubert")
    parser.add_argument("--outside", choices=("plain", "erased"), default="plain")
    parser.add_argument("--phonation", action="store_true", help="phonation spans only (/a/)")
    args = parser.parse_args()
    if args.step == "score-erased":
        score_outside_erased(args.vowel)
    elif args.step == "score" and args.phonation:
        if args.vowel != "a":
            parser.error("the phonation features exist for /a/ only")
        score_outside_phonation()
    elif args.step == "score":
        if args.vowel == "a":
            parser.error("the /a/ outside scores come from cross_database.py reverse")
        score_outside(args.vowel)
    else:
        if args.inside != "hubert" and args.vowel != "a":
            parser.error(f"the {args.inside} C1 scores exist for /a/ only")
        if args.outside == "erased" and args.inside != "hubert":
            parser.error("the erased outside detector goes with the HuBERT C1 only")
        if args.phonation and (
            args.vowel != "a" or args.inside != "hubert" or args.outside != "plain"
        ):
            parser.error("the phonation view is /a/, HuBERT C1, plain outside detector")
        analyse(args.vowel, args.inside, args.outside, args.phonation)


if __name__ == "__main__":
    main()
