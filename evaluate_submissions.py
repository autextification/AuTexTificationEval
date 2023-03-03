"""
Module for evaluating submissions. See:
https://sites.google.com/view/autextification/submit-results
https://sites.google.com/view/autextification/evaluation
"""

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import classification_report, f1_score


def main(submissions_path: Path, ground_truth_path: Path, subtask: str, language: str):
    results = {
        "team": [],
        "run": [],
        "all_metrics": [],
        "mf1": [],
        "mf1_cinterval": [],
    }
    ground_truth_df = pd.read_csv(
        next(ground_truth_path.glob(f"{subtask}/{language}/truth.tsv")),
        delimiter="\t",
        index_col="id",
    )

    for run in submissions_path.glob(f"*/{subtask}/{language}/*"):
        team = str(run.parents[2]).split("/")[-1]
        run_name = run.stem
        run_df = pd.read_csv(run, delimiter="\t", index_col="id")
        if len(run_df) != len(ground_truth_df):
            print(
                f"The number of predicted examples does not match with the reference: {team}"
            )
            continue
        if not len(
            set(run_df.index.tolist()).intersection(set(ground_truth_df.index.tolist()))
        ) == len(ground_truth_df):
            print(
                f"There is a mismatch between the ids of team {team} and the ground truth."
            )
            continue

        results["team"].append(team)
        results["run"].append(run_name)

        run_df = run_df.join(
            ground_truth_df,
            on="id",
            how="left",
            lsuffix="_pred",
            rsuffix="_true",
        )

        y_true = run_df["label_true"]
        y_pred = run_df["label_pred"]
        results["all_metrics"].append(
            classification_report(
                y_true=y_true,
                y_pred=y_pred,
                digits=4,
                output_dict=True,
            )
        )
        results["mf1"].append(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

        mf1_cinterval = bootstrap(
            data=[y_true, y_pred],
            statistic=partial(f1_score, average="macro"),
            n_resamples=100,
            paired=True,
            confidence_level=0.95,
            method="basic",
        )
        results["mf1_cinterval"].append(
            (
                mf1_cinterval.confidence_interval.low,
                mf1_cinterval.confidence_interval.high,
            )
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mf1", ascending=False)
    print(results_df)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "subtask",
        type=str,
        help="Subtask to evaluate",
        choices=["subtask_1", "subtask_2"],
    )
    parser.add_argument(
        "language",
        type=str,
        help="Language to evaluate",
        choices=["es", "en"],
    )
    parser.add_argument(
        "--submissions_path",
        type=str,
        required=False,
        help="Path to the submissions folder",
        default="../task_submissions/submissions",
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=False,
        help="Path to the ground_truth folder",
        default="../task_submissions/ground_truth",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        submissions_path=Path(args.submissions_path),
        ground_truth_path=Path(args.ground_truth_path),
        subtask=args.subtask,
        language=args.language,
    )
