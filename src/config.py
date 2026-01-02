# Running config

data_path = "latest_dataset"

coins_to_fetch = ["BTC","ETH"]

time_frames = ["1d"]

input_types = ['log_ret_vol', 'volatility', 'rsi', 'macd', 'bollinger_bands', 'atr']

pred = "log_ret_close"

output_path = "new_output"
rmse_dir = f"{output_path}/rmse"
model_output_dir = f"{output_path}/model_predictions"

model_name = "XGBOOST"


# FORECAST
# Use 25% of the data for testing, the rest for training
test_percentage = 0.25
# Use 10% of the training data for validation
val_percentage = 0.1

