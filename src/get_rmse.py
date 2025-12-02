import config
import os
import pandas as pd

from darts.metrics import rmse
from darts.timeseries import TimeSeries
from darts import concatenate


def get_rmse_scores():
    rmse_outpath = f"{config.rmse_dir}/{config.pred}"

    os.makedirs(rmse_outpath, exist_ok=True)

    print(f"Building {rmse_outpath} ...")

    # Data will be added to this DataFrame
    rmse_df = pd.DataFrame()

    for coin in config.coins_to_fetch:
        # Get the predictions
        _, rmse_df_coin = all_model_predictions(
            model=config.pred, model_name = config.model_name, coin=coin
        )
        # Convert the dataframe to a list of lists
        rmse_df_list = pd.DataFrame(
            {col: [rmse_df_coin[col].tolist()] for col in rmse_df_coin}
        )
        # Add the coin to the index
        rmse_df_list.index = [coin]
        # Add the data to the dataframe
        rmse_df = pd.concat([rmse_df, rmse_df_list])

    # Save the dataframe to a csv
    rmse_df.to_csv(f"{config.rmse_dir}/{config.pred}/rmse_1d.csv", index=True)

    # Print number on Nan values
    nan_values = rmse_df.isna().sum().sum()
    if nan_values > 0:
        print(f"Number of NaN values in 1d for {config.pred}: {nan_values}")



def all_model_predictions(
    model: str, model_name:str ,coin: str
) -> (dict, pd.DataFrame):
    # Save the predictions and tests for each model
    model_predictions = {}
  
    preds, _, tests, rmses = get_predictions(
        model=model, # log_returns 
        forecasting_model=model_name, # LSTM
        coin=coin,
        time_frame="1d",
    )
    # If the model does not exist, skip it
    if preds is not None:
        model_predictions[model_name] = (preds, tests, rmses)

    # Only use the third value in the tuple (the rmse) and convert to a dict
    rmses = {model: rmse for model, (_, _, rmse) in model_predictions.items()}
    rmse_df = pd.DataFrame(rmses)

    return model_predictions, rmse_df





def get_predictions(
    model: str = "log_returns",
    forecasting_model: str = "LSTM",
    coin: str = "BTC",
    time_frame: str = "1d",
    concatenated: bool = True,
) -> (TimeSeries, TimeSeries, list):
    """
    Gets the predictions for a given model.

    Parameters
    ----------
    model_dir : str
        Options are: "models" or "raw_models"
    forecasting_model : str
        Options are the models that were trained, for instance "ARIMA"
    coin : str
        This can be any of the 21 coins that were trained on
    time_frame : str
        Options are: "1m", "15m", "4h", and "1d"

    Returns
    -------
    preds, tests, rmses
        The forecast (prediction), actual values, and the rmse for each period
    """
    preds = []
    trains = []
    tests = []
    rmses = []

    for period in range(config.n_periods):
        file_loc = (
            f"{config.model_output_dir}/{model}/{forecasting_model}/{coin}/{time_frame}"
        )
        pred_path = f"{file_loc}/pred_{period}.csv"
        train_path = f"{file_loc}/train_{period}.csv"
        test_path = f"{file_loc}/test_{period}.csv"

        if not os.path.exists(pred_path):
            print(f"Warning the following file does not exist: {pred_path}")
            return None, None, None, None

        # Create the prediction TimeSeries
        pred_df = pd.read_csv(pred_path)
        # The value_cols is always the 2nd column (that is not time)
        """pred = TimeSeries.from_dataframe(
            pred_df, time_col="time", value_cols=pred_df.columns[1]
        )"""
        pred = TimeSeries.from_dataframe(
            pred_df, time_col="date", value_cols=pred_df.columns[1]
        )
        try:
            train_df = pd.read_csv(train_path)
            train = TimeSeries.from_dataframe(
                train_df, time_col="date", value_cols=train_df.columns[1]
            )
        except Exception:
            train = None
        test_df = pd.read_csv(test_path)
        test = TimeSeries.from_dataframe(
            test_df, time_col="date", value_cols=test_df.columns[1]
        )

        # Calculate the RMSE for this period and add it to the list
        rmses.append(rmse(test, pred))

        # Add it to list
        preds.append(pred)
        trains.append(train)
        tests.append(test)

    # Make it one big TimeSeries
    if (concatenated):
        preds = concatenate(preds, axis=0)
        # trains = concatenate(trains, axis=0)
        tests = concatenate(tests, axis=0)

    return preds, trains, tests, rmses