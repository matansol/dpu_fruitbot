"""
Mixed-effects logistic regression replacing the round-level one-sided
Mann-Whitney U tests in the "Deterioration sensitivity" and "Local correct
choice stratified by update direction" analyses.

Model (per environment, per outcome, restricted to Same + Salient-Contrast):
    outcome ~ condition * update_direction + (1 | participant_id)

Fitted with statsmodels BinomialBayesMixedGLM (variational Bayes). We
reparameterise the fixed effects with two nested dummies so the within-
direction Same-vs-Salient-Contrast contrasts are read directly off two
coefficients (algebraically equivalent to the condition*direction model).

Directional predictions (matched to the original Mann-Whitney tails):
    correct_local:       Same > Salient-Contrast   (log-OR positive)
    correct_generalized: Salient-Contrast > Same   (log-OR negative)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "mixed_effects_results"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = ["Same", "Salient-Contrast"]

# outcome_col, direction_col, name printed in paper, hypothesis direction on
# beta_same_vs_SC (positive log-OR = Same > Salient-Contrast).
ANALYSES = [
    {
        "outcome": "good_choice_local",
        "outcome_label": "correct_local",
        "direction_col": "improved_from_prev_feedback",
        "hyp": "greater",  # Same > Salient-Contrast
    },
    {
        "outcome": "good_choice_global",
        "outcome_label": "correct_generalized",
        "direction_col": "improved_from_prev_mean",
        "hyp": "less",     # Salient-Contrast > Same (beta negative)
    },
]


def load_data() -> dict[str, pd.DataFrame]:
    mg = pd.read_csv(BASE_DIR / "choice_results_for_combined_minigrid_df.csv")
    fb = pd.read_csv(BASE_DIR / "choice_results_for_combined_fruitbot_df.csv")
    fb["group_label"] = fb["group_label"].replace({"Contrast": "Salient-Contrast"})
    return {"MiniGrid": mg, "FruitBot": fb}


def one_sided_p(z: float, hyp: str) -> float:
    """hyp='greater' tests beta>0; hyp='less' tests beta<0."""
    return float(1.0 - norm.cdf(z)) if hyp == "greater" else float(norm.cdf(z))


def fit_mem(df: pd.DataFrame, outcome_col: str, direction_col: str):
    d = df[df["group_label"].isin(CONDITIONS)].copy()
    d = d.dropna(subset=[outcome_col, direction_col, "user_id"])
    d["y"] = d[outcome_col].astype(int)
    d["is_pos"] = (d[direction_col].astype(float) > 0.5).astype(int)
    d["is_same"] = (d["group_label"] == "Same").astype(int)
    d["same_x_neg"] = d["is_same"] * (1 - d["is_pos"])
    d["same_x_pos"] = d["is_same"] * d["is_pos"]

    # Fixed-effect design: Intercept, is_pos, same_x_neg, same_x_pos.
    # Two last columns are the within-direction log-ORs (Same vs Salient-Contrast).
    X = np.column_stack([
        np.ones(len(d)),
        d["is_pos"].values,
        d["same_x_neg"].values,
        d["same_x_pos"].values,
    ]).astype(float)
    fe_names = [
        "Intercept",
        "direction_pos",
        "same_vs_SC_within_neg",
        "same_vs_SC_within_pos",
    ]

    # Random intercept per participant via indicator matrix.
    pids, pid_idx = np.unique(d["user_id"].values, return_inverse=True)
    Z = np.zeros((len(d), len(pids)))
    Z[np.arange(len(d)), pid_idx] = 1.0
    ident = np.zeros(len(pids), dtype=int)  # one variance component

    model = BinomialBayesMixedGLM(
        endog=d["y"].values.astype(float),
        exog=X,
        exog_vc=Z,
        ident=ident,
        vcp_p=3.0,
        fe_p=3.0,
    )
    result = model.fit_vb()

    info = {
        "n_rows": int(len(d)),
        "n_participants": int(len(pids)),
        "fe_names": fe_names,
        "fe_mean": [float(x) for x in result.fe_mean],
        "fe_sd": [float(x) for x in result.fe_sd],
        "vcp_mean": [float(x) for x in result.vcp_mean],
        "vcp_sd": [float(x) for x in result.vcp_sd],
        "cell_means": (
            d.groupby(["is_pos", "is_same"])["y"].mean().to_dict()
        ),
        "cell_counts": (
            d.groupby(["is_pos", "is_same"])["y"].size().to_dict()
        ),
        "converged": bool(getattr(result, "converged", True)),
    }
    return result, info, d


def extract_contrast(result, fe_names, key, hyp):
    idx = fe_names.index(key)
    b = float(result.fe_mean[idx])
    se = float(result.fe_sd[idx])
    z = b / se if se > 0 else float("nan")
    return {
        "logOR": b,
        "SE": se,
        "z": z,
        "OR": float(np.exp(b)),
        "CI_low": float(np.exp(b - 1.96 * se)),
        "CI_high": float(np.exp(b + 1.96 * se)),
        "p_one_sided": one_sided_p(z, hyp),
    }


def mw_user_level(d: pd.DataFrame, outcome_col: str, direction_val: int, mw_alt: str):
    """Reproduce the original user-level one-sided Mann-Whitney test."""
    sub = d[d["is_pos"] == direction_val]
    user_means = (
        sub.groupby(["user_id", "group_label"], as_index=False)[outcome_col].mean()
    )
    g1 = user_means[user_means["group_label"] == "Salient-Contrast"][outcome_col].dropna()
    g2 = user_means[user_means["group_label"] == "Same"][outcome_col].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return {"U": float("nan"), "p": float("nan"), "n_SC": len(g1), "n_Same": len(g2)}
    # mw_alt is on (Salient-Contrast vs Same) in alphabetical order.
    u, p = mannwhitneyu(g1, g2, alternative=mw_alt)
    return {"U": float(u), "p": float(p), "n_SC": int(len(g1)), "n_Same": int(len(g2))}


def run():
    envs = load_data()
    rows, sentences, summaries = [], [], {}

    for env_name, env_df in envs.items():
        for a in ANALYSES:
            outcome = a["outcome"]
            outcome_label = a["outcome_label"]
            direction_col = a["direction_col"]
            hyp = a["hyp"]
            # Matching Mann-Whitney tail: hyp on (Same - SC) beta.
            # mannwhitneyu sorts groups alphabetically -> (Salient-Contrast, Same).
            # "greater" on beta_same_vs_SC ↔ Same > SC ↔ mannwhitneyu alternative="less"
            #   (SC stochastically less than Same).
            # "less" on beta_same_vs_SC ↔ SC > Same ↔ mannwhitneyu alternative="greater".
            mw_alt = "less" if hyp == "greater" else "greater"

            result, info, d = fit_mem(env_df, outcome, direction_col)
            fe_names = info["fe_names"]

            for dir_name, key, dir_val in [
                ("negative", "same_vs_SC_within_neg", 0),
                ("positive", "same_vs_SC_within_pos", 1),
            ]:
                c = extract_contrast(result, fe_names, key, hyp)
                mw = mw_user_level(d, outcome, dir_val, mw_alt)

                # Sign concordance: log-OR and MW deviation from 0.5 should agree.
                # MW p small under same tail as our hyp if MEM and MW agree.
                mem_sig = c["p_one_sided"] < 0.05
                mw_sig = (not np.isnan(mw["p"])) and mw["p"] < 0.05
                diverges = mem_sig != mw_sig

                rows.append({
                    "environment": env_name,
                    "outcome": outcome_label,
                    "direction": dir_name,
                    "OR": round(c["OR"], 3),
                    "CI_low": round(c["CI_low"], 3),
                    "CI_high": round(c["CI_high"], 3),
                    "logOR": round(c["logOR"], 4),
                    "SE": round(c["SE"], 4),
                    "z": round(c["z"], 3),
                    "p_one_sided_MEM": round(c["p_one_sided"], 4),
                    "MW_U": mw["U"],
                    "MW_p_one_sided": round(mw["p"], 4) if not np.isnan(mw["p"]) else mw["p"],
                    "n_participants_SC": mw["n_SC"],
                    "n_participants_Same": mw["n_Same"],
                    "n_rows": info["n_rows"],
                    "converged": info["converged"],
                    "divergence_flag": diverges,
                })

                # Paper-style sentence.
                if hyp == "greater":  # local: expecting Same > SC
                    verb = "exceeded" if c["p_one_sided"] < 0.05 and c["OR"] > 1 else "did not differ from"
                    group_a, group_b = "Same", "Salient-Contrast"
                else:  # generalized: expecting SC > Same; report SC vs Same with OR inverted
                    inv_OR = 1.0 / c["OR"]
                    inv_lo = 1.0 / c["CI_high"]
                    inv_hi = 1.0 / c["CI_low"]
                    verb = "exceeded" if c["p_one_sided"] < 0.05 and inv_OR > 1 else "did not differ from"
                    group_a, group_b = "Salient-Contrast", "Same"
                    # Re-bind reporting OR/CI to the Salient-Contrast-over-Same direction.
                    c_report = {"OR": inv_OR, "CI_low": inv_lo, "CI_high": inv_hi, "p_one_sided": c["p_one_sided"]}
                if hyp == "greater":
                    c_report = {"OR": c["OR"], "CI_low": c["CI_low"], "CI_high": c["CI_high"], "p_one_sided": c["p_one_sided"]}

                sentences.append(
                    f"In {env_name}, {group_a} {verb} {group_b} on {dir_name} updates "
                    f"(OR = {c_report['OR']:.2f}, 95% CI [{c_report['CI_low']:.2f}, {c_report['CI_high']:.2f}], "
                    f"p = {c_report['p_one_sided']:.3f})."
                )

            summaries[f"{env_name}__{outcome_label}"] = {
                "n_rows": info["n_rows"],
                "n_participants": info["n_participants"],
                "fe_names": fe_names,
                "fe_mean": info["fe_mean"],
                "fe_sd": info["fe_sd"],
                "vcp_mean": info["vcp_mean"],
                "vcp_sd": info["vcp_sd"],
                "cell_means": {str(k): float(v) for k, v in info["cell_means"].items()},
                "cell_counts": {str(k): int(v) for k, v in info["cell_counts"].items()},
                "converged": info["converged"],
                "hypothesis": (
                    "Same > Salient-Contrast (local)"
                    if hyp == "greater"
                    else "Salient-Contrast > Same (generalized)"
                ),
            }

    contrast_table = pd.DataFrame(rows)
    contrast_csv = OUT_DIR / "update_direction_contrasts.csv"
    contrast_table.to_csv(contrast_csv, index=False)

    sentences_path = OUT_DIR / "paper_replacement_sentences.txt"
    sentences_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

    summaries_path = OUT_DIR / "model_summaries.json"
    summaries_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    return contrast_table, sentences, summaries


if __name__ == "__main__":
    table, sentences, summaries = run()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Mixed-effects logistic regression contrasts ===")
    print(table.to_string(index=False))

    print("\n=== Paper replacement sentences ===")
    for s in sentences:
        print(s)

    print("\n=== Convergence / diagnostics ===")
    for key, info in summaries.items():
        print(
            f"{key}: n_rows={info['n_rows']}, n_participants={info['n_participants']}, "
            f"converged={info['converged']}, random-intercept SD (VB posterior mean on log scale) "
            f"vcp_mean={info['vcp_mean']}"
        )

    print("\n=== Divergences from Mann-Whitney ===")
    div = table[table["divergence_flag"]]
    if len(div) == 0:
        print("None: every MEM contrast matches the MW significance decision at alpha=0.05 (same tail).")
    else:
        print(div.to_string(index=False))

    print(f"\nSaved to: {OUT_DIR}")
