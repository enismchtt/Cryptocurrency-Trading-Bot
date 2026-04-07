import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models import XGBModel

import config
from data.binance_data import fetchData
from data.csv_data import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated XGBoost experiment runner for crypto forecasting."
    )
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--timeframes", nargs="+", default=["1d"])
    parser.add_argument("--fetch-fresh-data", action="store_true")
    parser.add_argument("--fetch-amount", type=int, default=1)
    parser.add_argument("--data-path", default="experiment_dataset")
    parser.add_argument("--output-path", default="experiment_output")
    parser.add_argument("--baseline-output-path", default="new_output")
    parser.add_argument("--target", default="log_ret_close")
    parser.add_argument(
        "--feature-pool",
        nargs="+",
        default=[
            "open",
            "high",
            "low",
            "close",
            "log_ret_vol",
            "volatility",
            "rsi",
            "macd",
            "bollinger_bands",
            "atr",
        ],
    )
    parser.add_argument("--max-optional-features", type=int, default=3)
    parser.add_argument("--test-percentage", type=float, default=0.25)
    parser.add_argument(
        "--fees-bps",
        type=float,
        default=5.0,
        help="Transaction fee in basis points for simple strategy metric.",
    )
    return parser.parse_args()


def set_runtime_config(args: argparse.Namespace) -> None:
    config.data_path = args.data_path
    config.output_path = args.output_path
    config.model_name = "XGBOOST"
    config.pred = args.target
    config.test_percentage = args.test_percentage


def fetch_new_data(args: argparse.Namespace) -> None:
    for coin in args.coins:
        for timeframe in args.timeframes:
            os.makedirs(f"{config.data_path}/{coin}", exist_ok=True)
            fetchData(
                symbol=coin,
                amount=args.fetch_amount,
                timeframe=timeframe,
                as_csv=True,
            )


def build_feature_combos(feature_pool: list[str], target: str, max_optional: int) -> list[list[str]]:
    combos = []
    max_optional = max(0, min(max_optional, len(feature_pool)))
    for r in range(max_optional + 1):
        for combo in itertools.combinations(feature_pool, r):
            combos.append(list(combo) + [target])
    return combos


def build_param_grid() -> list[dict]:
    return [
        {
            "run_tag": "base_l7_cov7",
            "lags": 7,
            "cov_lags": 7,
            "max_depth": 4,
            "max_leaves": 10,
            "min_child_weight": 7,
            "subsample": 0.89,
            "colsample_bytree": 0.93,
            "gamma": 0.005,
        },
        {
            "run_tag": "deeper_l14_cov14",
            "lags": 14,
            "cov_lags": 14,
            "max_depth": 6,
            "max_leaves": 16,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.01,
        },
        {
            "run_tag": "regularized_l7_cov14",
            "lags": 7,
            "cov_lags": 14,
            "max_depth": 3,
            "max_leaves": 8,
            "min_child_weight": 9,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
        },
    ]


def get_train_test_full(coin: str, time_frame: str, cols: list[str]) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    data = read_csv(coin, time_frame, cols).dropna()
    data["date"] = data.index
    ts = TimeSeries.from_dataframe(data, "date", cols)
    split_idx = int(len(ts) * (1 - config.test_percentage))
    return ts[:split_idx], ts[split_idx:], ts


def simple_strategy_return(actual: np.ndarray, pred: np.ndarray, fees_bps: float) -> float:
    if len(actual) == 0:
        return np.nan
    pos = np.sign(pred)
    gross = pos * actual
    turns = np.abs(np.diff(pos, prepend=0))
    fee = (fees_bps / 10_000.0) * turns
    net = gross - fee
    return float(np.nansum(net))


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return np.nan
    return float(np.mean(np.sign(actual) == np.sign(pred)) * 100.0)


def regime_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    n = len(actual)
    if n < 4:
        return {
            "rmse_early": np.nan,
            "rmse_late": np.nan,
            "diracc_early": np.nan,
            "diracc_late": np.nan,
        }
    m = n // 2
    return {
        "rmse_early": float(np.sqrt(np.mean((actual[:m] - pred[:m]) ** 2))),
        "rmse_late": float(np.sqrt(np.mean((actual[m:] - pred[m:]) ** 2))),
        "diracc_early": directional_accuracy(actual[:m], pred[:m]),
        "diracc_late": directional_accuracy(actual[m:], pred[m:]),
    }


def run_experiment(args: argparse.Namespace) -> Path:
    set_runtime_config(args)
    if args.fetch_fresh_data:
        fetch_new_data(args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_path) / "runs" / run_id
    pred_root = run_root / "model_predictions" / "XGBOOST"
    metrics_root = run_root / "metrics"
    plots_root = run_root / "plots"
    metrics_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    feature_combos = build_feature_combos(args.feature_pool, args.target, args.max_optional_features)
    param_grid = build_param_grid()
    all_needed_cols = sorted({c for combo in feature_combos for c in combo})
    records: list[dict] = []

    for coin in args.coins:
        for timeframe in args.timeframes:
            train_full, test_full, full_series = get_train_test_full(coin, timeframe, all_needed_cols)
            train_target = train_full[args.target]
            test_target = test_full[args.target]
            full_target = full_series[args.target]

            for combo in feature_combos:
                combo_name = "__".join(combo)
                cov_cols = [c for c in combo if c != args.target]
                train_cov = train_full[cov_cols] if cov_cols else None
                full_cov = full_series[cov_cols] if cov_cols else None

                for params in param_grid:
                    cov_lags = params["cov_lags"] if cov_cols else None
                    model = XGBModel(
                        lags=params["lags"],
                        lags_past_covariates=cov_lags,
                        output_chunk_length=1,
                        random_state=42,
                        max_depth=params["max_depth"],
                        max_leaves=params["max_leaves"],
                        min_child_weight=params["min_child_weight"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        gamma=params["gamma"],
                    )

                    model.fit(series=train_target, past_covariates=train_cov)
                    pred = model.historical_forecasts(
                        series=full_target,
                        past_covariates=full_cov,
                        start=len(train_target),
                        forecast_horizon=1,
                        stride=1,
                        retrain=False,
                        verbose=False,
                    )

                    pred_df = pred.to_dataframe().reset_index()
                    actual_df = test_target.to_dataframe().reset_index()
                    actual_vals = actual_df.iloc[:, -1].values
                    pred_vals = pred_df.iloc[:, -1].values
                    n = min(len(actual_vals), len(pred_vals))
                    actual_vals = actual_vals[-n:]
                    pred_vals = pred_vals[-n:]

                    ts_actual = TimeSeries.from_dataframe(
                        actual_df.tail(n), time_col=actual_df.columns[0], value_cols=actual_df.columns[-1]
                    )
                    ts_pred = TimeSeries.from_dataframe(
                        pred_df.tail(n), time_col=pred_df.columns[0], value_cols=pred_df.columns[-1]
                    )

                    rmse_score = float(rmse(ts_actual, ts_pred))
                    mae_score = float(mae(ts_actual, ts_pred))
                    dir_acc = directional_accuracy(actual_vals, pred_vals)
                    strat_ret = simple_strategy_return(actual_vals, pred_vals, args.fees_bps)
                    regimes = regime_metrics(actual_vals, pred_vals)

                    save_dir = pred_root / f"input_{combo_name}" / params["run_tag"] / args.target / coin / timeframe
                    save_dir.mkdir(parents=True, exist_ok=True)
                    pred_df.to_csv(save_dir / "pred.csv", index=False)
                    actual_df.to_csv(save_dir / "actual.csv", index=False)
                    (save_dir / "features.txt").write_text(str(combo), encoding="utf-8")
                    (save_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

                    records.append(
                        {
                            "run_id": run_id,
                            "coin": coin,
                            "timeframe": timeframe,
                            "target": args.target,
                            "feature_combo": combo_name,
                            "n_features": len(cov_cols),
                            "param_tag": params["run_tag"],
                            "lags": params["lags"],
                            "cov_lags": params["cov_lags"],
                            "max_depth": params["max_depth"],
                            "max_leaves": params["max_leaves"],
                            "min_child_weight": params["min_child_weight"],
                            "subsample": params["subsample"],
                            "colsample_bytree": params["colsample_bytree"],
                            "gamma": params["gamma"],
                            "rmse": rmse_score,
                            "mae": mae_score,
                            "direction_acc_pct": dir_acc,
                            "strategy_return_net": strat_ret,
                            **regimes,
                            "out_dir": str(save_dir),
                        }
                    )

    df = pd.DataFrame(records).sort_values(["coin", "rmse"], ascending=[True, True])
    df.to_csv(metrics_root / "leaderboard_all.csv", index=False)

    for coin in df["coin"].unique():
        coin_df = df[df["coin"] == coin].sort_values("rmse")
        coin_df.to_csv(metrics_root / f"leaderboard_{coin}.csv", index=False)

    write_factor_analysis(df, metrics_root)
    compare_with_baseline(df, args.baseline_output_path, metrics_root)
    make_plots(df, plots_root)
    write_readme(run_root, args)
    return run_root


def write_factor_analysis(df: pd.DataFrame, metrics_root: Path) -> None:
    feature_rows = []
    all_features = set()
    for combo in df["feature_combo"].unique():
        all_features.update([x for x in combo.split("__") if x])

    for feat in sorted(all_features):
        has_feat = df["feature_combo"].str.contains(fr"(?:^|__){feat}(?:__|$)", regex=True)
        with_feat = df[has_feat]
        without_feat = df[~has_feat]
        if with_feat.empty or without_feat.empty:
            continue
        feature_rows.append(
            {
                "feature": feat,
                "mean_rmse_with": with_feat["rmse"].mean(),
                "mean_rmse_without": without_feat["rmse"].mean(),
                "rmse_delta_without_minus_with": without_feat["rmse"].mean() - with_feat["rmse"].mean(),
                "mean_diracc_with": with_feat["direction_acc_pct"].mean(),
                "mean_diracc_without": without_feat["direction_acc_pct"].mean(),
            }
        )
    pd.DataFrame(feature_rows).sort_values(
        "rmse_delta_without_minus_with", ascending=False
    ).to_csv(metrics_root / "factor_feature_impact.csv", index=False)

    hyper_cols = ["lags", "cov_lags", "max_depth", "max_leaves", "min_child_weight", "subsample", "colsample_bytree", "gamma"]
    for col in hyper_cols:
        impact = (
            df.groupby(col, dropna=False)
            .agg(
                runs=("rmse", "count"),
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_diracc=("direction_acc_pct", "mean"),
                mean_strategy_return=("strategy_return_net", "mean"),
            )
            .reset_index()
            .sort_values("mean_rmse")
        )
        impact.to_csv(metrics_root / f"factor_hyper_{col}.csv", index=False)


def compare_with_baseline(exp_df: pd.DataFrame, baseline_root: str, metrics_root: Path) -> None:
    baseline_root_path = Path(baseline_root) / "model_predictions" / "XGBOOST"
    rows = []
    if not baseline_root_path.exists():
        pd.DataFrame(
            [{"note": f"Baseline path not found: {baseline_root_path}"}]
        ).to_csv(metrics_root / "baseline_comparison.csv", index=False)
        return

    for pred_path in baseline_root_path.rglob("pred.csv"):
        actual_path = pred_path.parent / "actual.csv"
        if not actual_path.exists():
            continue
        try:
            pred_df = pd.read_csv(pred_path)
            act_df = pd.read_csv(actual_path)
            if "date" not in pred_df.columns:
                pred_df.rename(columns={pred_df.columns[0]: "date"}, inplace=True)
            if "date" not in act_df.columns:
                act_df.rename(columns={act_df.columns[0]: "date"}, inplace=True)
            ts_pred = TimeSeries.from_dataframe(pred_df, time_col="date", value_cols=pred_df.columns[-1])
            ts_act = TimeSeries.from_dataframe(act_df, time_col="date", value_cols=act_df.columns[-1])
            score = float(rmse(ts_act, ts_pred))
            parts = pred_path.parts
            # Expected layout: .../<target>/<coin>/<timeframe>/pred.csv
            coin = parts[-3]
            timeframe = parts[-2]
            rows.append({"coin": coin, "timeframe": timeframe, "baseline_rmse": score, "baseline_pred_path": str(pred_path)})
        except Exception:
            continue

    baseline_df = pd.DataFrame(rows)
    if baseline_df.empty:
        pd.DataFrame([{"note": "No baseline predictions discovered."}]).to_csv(
            metrics_root / "baseline_comparison.csv", index=False
        )
        return

    best_baseline = baseline_df.groupby(["coin", "timeframe"], as_index=False)["baseline_rmse"].min()
    best_exp = exp_df.groupby(["coin", "timeframe"], as_index=False)["rmse"].min().rename(columns={"rmse": "best_experiment_rmse"})
    cmp_df = best_exp.merge(best_baseline, on=["coin", "timeframe"], how="left")
    cmp_df["rmse_improvement_vs_baseline"] = cmp_df["baseline_rmse"] - cmp_df["best_experiment_rmse"]
    cmp_df.to_csv(metrics_root / "baseline_comparison.csv", index=False)


def make_plots(df: pd.DataFrame, plots_root: Path) -> None:
    for coin in df["coin"].unique():
        top = df[df["coin"] == coin].nsmallest(10, "rmse").copy()
        if top.empty:
            continue
        plt.figure(figsize=(12, 5))
        labels = [f"{a}|{b}" for a, b in zip(top["feature_combo"], top["param_tag"])]
        plt.bar(range(len(top)), top["rmse"].values)
        plt.xticks(range(len(top)), labels, rotation=60, ha="right")
        plt.title(f"Top-10 RMSE Models - {coin}")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(plots_root / f"top10_rmse_{coin}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(top["rmse"], top["direction_acc_pct"])
        for _, row in top.iterrows():
            plt.annotate(row["param_tag"], (row["rmse"], row["direction_acc_pct"]), fontsize=8)
        plt.xlabel("RMSE")
        plt.ylabel("Directional Accuracy (%)")
        plt.title(f"RMSE vs Direction Accuracy - {coin} (Top 10)")
        plt.tight_layout()
        plt.savefig(plots_root / f"rmse_vs_diracc_{coin}.png", dpi=200)
        plt.close()


def write_readme(run_root: Path, args: argparse.Namespace) -> None:
    txt = f"""# XGBoost Experiment Run

Run path: {run_root}
Created at: {datetime.now().isoformat(timespec="seconds")}

## Configuration
- coins: {args.coins}
- timeframes: {args.timeframes}
- fetch_fresh_data: {args.fetch_fresh_data}
- fetch_amount: {args.fetch_amount}
- data_path: {args.data_path}
- output_path: {args.output_path}
- baseline_output_path: {args.baseline_output_path}
- target: {args.target}
- feature_pool: {args.feature_pool}
- max_optional_features: {args.max_optional_features}
- test_percentage: {args.test_percentage}
- fees_bps: {args.fees_bps}

## Outputs
- model predictions: model_predictions/XGBOOST/...
- global leaderboard: metrics/leaderboard_all.csv
- per-coin leaderboards: metrics/leaderboard_*.csv
- factor analysis: metrics/factor_*.csv
- baseline comparison: metrics/baseline_comparison.csv
- plots: plots/*.png
"""
    (run_root / "README.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    cli_args = parse_args()
    run_folder = run_experiment(cli_args)
    print(f"Experiment completed. Results are in: {run_folder}")
