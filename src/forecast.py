import config
import os
from darts import TimeSeries
from darts.models import( BlockRNNModel,XGBModel)

from data.csv_data import read_csv
from tqdm import tqdm




def get_train_test(
    coin: str = "BTC", 
    time_frame: str = "1d", 
    col: str | list = "log_returns", # Updated type hint to allow lists of features
    ):
    
    # 1. Read data
    # Ensure 'col' is a list for read_csv, even if a single string is passed
    cols_to_read = [col] if isinstance(col, str) else col
    
    data = read_csv(coin, time_frame, cols_to_read).dropna()
    data["date"] = data.index

    # 2. Create Darts TimeSeries
    # This handles both single-variate (target) and multi-variate (covariates)
    time_series = TimeSeries.from_dataframe(data, "date", col)

    # 3. Simple Split (Train / Test)
    # Calculate the index to split at (e.g., 80% mark)
    split_index = int(len(time_series) * (1 - config.test_percentage))
    
    train = time_series[:split_index]
    test = time_series[split_index:]
    full = time_series

    # 4. Return Single TimeSeries objects
    # We return single objects now, not lists of periods.
    return train, test, full
    
    

def get_model(model_type,forecasting_model_name, coin, time_frame , lags_past_covariates):
    # Returns Darts TimeForecast Model 
    if model_type == "LSTM":
        return BlockRNNModel(
            model="LSTM",
            input_chunk_length=14,    # Look back 14 steps (matches your RSI window)
            output_chunk_length=1,    # Predict 1 step into the future
            hidden_dim=64,            # Reduced from 128 to prevent overfitting
            n_rnn_layers=1,
            dropout=0.2,
            batch_size=16,
            n_epochs=25,
            optimizer_kwargs={"lr": 1e-3},
            model_name=forecasting_model_name,   # Unique name for Tensorboard logs
            log_tensorboard=True,
            random_state=42,
            force_reset=True,
            save_checkpoints=True,
        )
    elif model_type == "XGBOOST":
        return XGBModel(
            # 1. Structure Parameters
            lags=7,                   # Your "Best Param" for target lookback
            lags_past_covariates=lags_past_covariates,  
            output_chunk_length=1,    # Predict 1 step ahead (Fair comparison)
            # 2. Hyperparameters (From your optimization)
            colsample_bytree=0.93,
            gamma=0.0050,
            max_depth=None,
            max_leaves=10,            # This constrains the complexity significantly
            min_child_weight=7,
            subsample=0.89,
            # 3. Settings
            random_state=42,
            # XGBoost doesn't use Tensorboard or Checkpoints folder the same way
        )


def forecast_model(
    input_combinations: list,   # List of lists (e.g. [['rsi', 'log_ret'], ['macd', 'log_ret']])
    model_name: str = "LSTM", 
    forecast_type: str = "log_ret_close", 
    coin: str = "BTC", 
    time_frame: str = "1d"
):
    """
    Trains and forecasts multiple models based on a list of input feature combinations.
    """

    # 1. OPTIMIZATION: Identify ALL unique columns needed across all combinations
    # This flattens the list of lists into one set of unique column names
    # e.g. {'rsi', 'macd', 'log_ret_close', 'volatility'}
    all_needed_cols = sorted(list(set(sum(input_combinations, []))))

    # 2. Fetch Data ONCE (Contains all possible columns)
    # We rename variables to '_full' to indicate they hold everything
    train_full, test_full, series_full = get_train_test(
        coin=coin, 
        time_frame=time_frame, 
        col=all_needed_cols
    )

    # 3. Loop through each Feature Combination
    for feature_combo in tqdm(input_combinations, desc="Testing Feature Combinations"):
        
        # --- A. Setup Paths ---
        # Join features with double underscore for readability (e.g., "rsi__macd__log_ret_close")
        input_name = "__".join(feature_combo)
        
        save_dir = (
            f"{config.output_path}/model_predictions/{model_name}/input_{input_name}/{forecast_type}/{coin}/{time_frame}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # --- B. Data Preparation (Slicing) ---
        # Target is ALWAYS config.pred
        target_col = config.pred
        
        # Covariates are everything in this combo EXCEPT the target
        covariate_cols = [c for c in feature_combo if c != target_col]
        
        # Extract Target Series (What we predict)
        train_target = train_full[target_col]
        full_target  = series_full[target_col]
        
        # Extract Covariate Series (What helps us predict)
        # If list is empty, set to None so Darts knows there are no covariates
        train_cov = train_full[covariate_cols] if covariate_cols else None
        full_cov  = series_full[covariate_cols] if covariate_cols else None

        # --- C. Model Setup ---
        # Create a unique name for this specific run so logs don't mix
        # e.g. "LSTM_BTC_1d_rsi__macd"
        unique_model_name = f"{model_name}_{coin}_{time_frame}_{input_name}"

        current_lags_past = 7 if train_cov is not None else None
        
        forecasting_model = get_model(model_name,unique_model_name, coin, time_frame, lags_past_covariates =current_lags_past )

        # --- D. Train ---
        forecasting_model.fit(
            series=train_target, 
            past_covariates=train_cov
        )

        # --- E. Forecast ---
        # Predict over the Test period using full history
        pred = forecasting_model.historical_forecasts(
            series=full_target,
            past_covariates=full_cov,
            start=len(train_target), # Start exactly where training ended
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False,
        )

        # --- F. Save Results ---
        pred.to_dataframe().to_csv(f"{save_dir}/pred.csv")
        
        # Save actual test values for comparison
        # (We slice test_full just to be safe, though target is same for all)
        test_full[target_col].to_dataframe().to_csv(f"{save_dir}/actual.csv")
        
        # Save the feature list for this run just in case
        with open(f"{save_dir}/features.txt", "w") as f:
            f.write(str(feature_combo))




    

    
    
