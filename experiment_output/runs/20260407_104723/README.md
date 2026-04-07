# XGBoost Experiment Run

Run path: experiment_output/runs/20260407_104723
Created at: 2026-04-07T10:47:50

## Configuration
- coins: ['BTC', 'ETH']
- timeframes: ['1d']
- fetch_fresh_data: True
- fetch_amount: 1
- data_path: experiment_dataset
- output_path: experiment_output
- baseline_output_path: new_output
- target: log_ret_close
- feature_pool: ['open', 'high', 'low', 'close', 'log_ret_vol', 'volatility', 'rsi', 'macd', 'bollinger_bands', 'atr']
- max_optional_features: 3
- test_percentage: 0.25
- fees_bps: 5.0

## Outputs
- model predictions: model_predictions/XGBOOST/...
- global leaderboard: metrics/leaderboard_all.csv
- per-coin leaderboards: metrics/leaderboard_*.csv
- factor analysis: metrics/factor_*.csv
- baseline comparison: metrics/baseline_comparison.csv
- plots: plots/*.png
