# Running config

data_path = "baseline_dataset"

coins_to_fetch = ["BTC","ETH"]

time_frames = ["1d"]


pred = "log_returns"

output_path = "output"
rmse_dir = f"{output_path}/rmse"
model_output_dir = f"{output_path}/model_predictions"

model_name = "LSTM"


# FORECAST
# Use 25% of the data for testing, the rest for training
test_percentage = 0.25
# Use 10% of the training data for validation
val_percentage = 0.1
# Split the data in periods of 5
n_periods = 5