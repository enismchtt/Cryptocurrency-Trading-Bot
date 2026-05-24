# Cryptocurrency Trading Bot

Hacettepe University BBM479 design project: cryptocurrency price-direction
forecasting with multiple machine-learning models, plus a full-stack web
dashboard that visualizes the predictions.

The repository has two parts:

| Part                | Purpose                                                                        |
| ------------------- | ------------------------------------------------------------------------------ |
| `src/`              | Offline research pipeline — fetches data, trains models, builds leaderboards.  |
| `webapp/`           | Live web app (FastAPI + React) that runs the same XGBoost recipe interactively. |

> **A separate, more detailed setup guide for the web app lives in
> [`webapp/README.md`](webapp/README.md).** This file covers the overall
> repository and the offline research code.

## Quick demo

```bash
# 1. Backend (port 8000)
./webapp/start_backend.sh

# 2. Frontend (port 5173) — in another terminal
./webapp/start_frontend.sh
```

Open <http://localhost:5173>. Pick one of the 20 supported coins, choose a
date range, and click **Compare** — the model fetches Binance candles on
demand, trains XGBoost (lags=7) and visualises the forecast.

## Repository layout

```
Cryptocurrency-Trading-Bot/
├── src/                                Offline research pipeline
│   ├── main.py                         Fetch -> forecast -> RMSE loop
│   ├── forecast.py                     darts XGBModel / BlockRNNModel wrapper
│   ├── calc_rmse.py                    Per-coin RMSE & direction-accuracy leaderboard
│   ├── config.py                       Feature combinations, target, paths
│   └── data/                           Binance fetcher + CSV reader
│
├── paper_dataset/                      Raw OHLCV CSVs (BTC, ETH) for the paper
├── paper_output_*/                     Backtest predictions per experiment
├── best_for_all/                       Aggregated leaderboards per feature set
├── best_for_all_timeframe_results.xlsx Final benchmark workbook
│
├── webapp/                             Live web dashboard
│   ├── backend/                        FastAPI + xgboost + Gemini
│   ├── frontend/                       Vite + React + Tailwind + Recharts
│   ├── start_backend.sh                One-liner: install + uvicorn
│   ├── start_frontend.sh               One-liner: npm install + vite dev
│   └── README.md                       Detailed web-app setup
│
└── README.md                           ← you are here
```

## Models compared in the project

During the research phase we evaluated four families of models, all with the
target `log_ret_close = log(close[t]) - log(close[t-1])` and a 7-step lag
window:

1. **XGBoost** (`darts.models.XGBModel`) — best overall, used in the web app.
2. **LSTM** (`darts.models.BlockRNNModel`, kind=LSTM) — strong on lower-frequency data.
3. **TCN** (Temporal Convolutional Network) — competitive at 1d.
4. **CNN-LSTM** — combined convolutional + recurrent.

All four were run across 20 coins × 4 timeframes (1m / 15m / 4h / 1d) and 64
feature combinations of:

```
log_ret_vol · volatility · rsi · macd · bollinger_bands · atr · log_ret_close
```

Aggregated leaderboards live in `best_for_all/`. The 1d feature sets with the
lowest RMSE for the most coins are `['log_ret_close']` (target only) and
`['rsi', 'macd', 'log_ret_close']`. The web app currently runs
`['rsi', 'macd', 'log_ret_close']` (configurable in `webapp/backend/config.py`).

## Running the offline research pipeline

The research code uses the same `.venv` as the web app.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-research.txt   # if not present, see src/data/binance_data.py imports
.venv/bin/python src/main.py
```

`src/main.py` will:

1. Fetch Binance OHLCV for every coin in `config.coins_to_fetch` × every
   timeframe in `config.time_frames` (skips if `config.data_path` exists).
2. Train the model defined by `config.model_name` for every feature
   combination in `config.selected_feature_combinations`.
3. Dump predictions to
   `<output_path>/model_predictions/<MODEL>/input_<combo>/<target>/<coin>/<timeframe>/`.
4. Build per-coin RMSE / direction-accuracy leaderboards in
   `<output_path>/rmse/`.

Knobs in `src/config.py`:

| Variable                          | Purpose                                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| `coins_to_fetch`                  | Coins to download. Defaults to `BTC`, `ETH`.                           |
| `time_frames`                     | `["1d", "4h", "15m", "1m"]`                                            |
| `input_types`                     | Pool of optional features used to build 2^N combinations.              |
| `pred`                            | Target column. Always `log_ret_close` in our experiments.              |
| `model_name`                      | `"XGBOOST"` or `"LSTM"`.                                               |
| `test_percentage` / `val_percentage` | Train/validation/test split sizes (default 75/10/15).               |
| `xgb_lags`, `xgb_n_estimators`, … | Hyperparameters consumed by `forecast.get_model`.                      |

## Data sources

* **OHLCV** — pulled directly from the Binance REST API
  (`https://api.binance.com/api/v3/klines`). Both the research script and the
  web app honour `BINANCE_API_KEY` / `BINANCE_API_SECRET` env vars if set, but
  public market data does not require them.

* **Cached candles for the web app** live in `webapp/backend/cache/`. Delete
  any CSV there to force a refetch.

## License / disclaimer

Educational project for Hacettepe BBM479. The forecasts are **not financial
advice** and the model is not suitable for live trading.
