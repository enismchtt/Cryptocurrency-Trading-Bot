# Running config

data_path = "paper_dataset_8.05.26"

coins_to_fetch = [ "BTC","ETH"]

"""coins_to_fetch = [
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "MATIC",
    "LINK",
    "ETC",
    "XLM",
    "LTC",
    "TRX",
    "ATOM",
    "XMR",
    "VET",
    "ALGO",
    "EOS",
    "CHZ",
    "IOTA",
    "NEO",
    "XTZ"
]"""

time_frames = ["1d","4h","15m","1m"]

isPaperSet = True

input_types = ['log_ret_vol', 'volatility', 'rsi', 'macd', 'bollinger_bands', 'atr']

pred = "log_ret_close"

output_path = "paper_output_8.05.26"
rmse_dir = f"{output_path}/rmse"
model_output_dir = f"{output_path}/model_predictions"

model_name = "LSTM"

selected_feature_combinations = [
    ['log_ret_close'],
    ['bollinger_bands', 'log_ret_close'],
    ['log_ret_vol', 'log_ret_close'],
    ['volatility', 'log_ret_close'],
    ['log_ret_vol', 'rsi', 'bollinger_bands', 'log_ret_close'],
    ['log_ret_vol', 'volatility', 'rsi', 'bollinger_bands', 'log_ret_close']
]

# FORECAST
# Use 25% of the data for testing, the rest for training
test_percentage = 0.25
# Use 10% of the training data for validation
val_percentage = 0.1

