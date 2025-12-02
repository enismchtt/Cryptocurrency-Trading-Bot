import config
import os
from darts import TimeSeries
from darts.models import RNNModel

from data.csv_data import read_csv
from tqdm import tqdm




def get_train_test(
    coin:str = "BTC", 
    time_frame:str = "1d", 
    col:str = "log_returns",
    ) -> (list, list, list):
    
    # Read data from a CSV file
    data = read_csv(coin, time_frame, [col]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", col)  # (df , timecolumn , value column's)

    # Set parameters for sliding window and periods
    test_size = int(
        len(time_series) / (1 / config.test_percentage - 1 + config.n_periods)
    )
    train_size = int(test_size * (1 / config.test_percentage - 1))

    # Save the training and test sets as lists of TimeSeries
    train_set = []
    test_set = []
    full_set = []

    for i in range(config.n_periods):
        # The train start shifts by the test size each period
        train_start = i * test_size
        train_end = train_start + train_size

        train = time_series[train_start:train_end]
        test = time_series[train_end : train_end + test_size]
        full = time_series[train_start : train_end + test_size]

       
        train_set.append(train)
        test_set.append(test)

        # The whole timeseries of this period
        full_set.append(full)

    return train_set, test_set, full_set
    
    

def get_model(forecasting_model_name, coin, time_frame):
    # Returns Darts TimeForecast Model 
    if forecasting_model_name == "LSTM":
        return RNNModel(
                input_chunk_length=12,
                training_length=25,
                model="LSTM",
                hidden_dim=128,
                dropout=0.01,
                batch_size=16,
                n_rnn_layers=1,
                n_epochs=25,
                optimizer_kwargs={"lr": 1e-3},
                model_name= forecasting_model_name,
                log_tensorboard=True,
                random_state=42,
                force_reset=True,
                save_checkpoints=True,
        )

def forecast_model(
    model_name:str = "LSTM", 
    forecast_type:str = "log_returns" , 
    coin:str = "BTC", 
    time_frame:str = "1d") :
    
    # Create directories
    save_dir = (
        f"{config.output_path}/model_predictions/{forecast_type}/{model_name}/{coin}/{time_frame}"
    )
    os.makedirs(save_dir, exist_ok=True)


    if forecast_type == "log_returns":
        col_name = "log returns"


    # Get the training and testing data for each period
    train_set, test_set, time_series = get_train_test(
        coin=coin, time_frame=time_frame, col=col_name
    )



    #forecast
    for period in tqdm(
        range(config.n_periods),
        desc=f"Forecasting periods for {model_name}/{coin}/{time_frame}",
        leave=False,
    ):
        # Reset the model
        forecasting_model = get_model(model_name, coin, time_frame)   # Model has to be type of Darts Forecasting Model

        # Fit on the training data
        forecasting_model.fit(series=train_set[period])

        # Generate the historical forecast
        pred = forecasting_model.historical_forecasts(
            time_series[period],
            start=len(train_set[period]),
            forecast_horizon=1,  # 1 step ahead forecasting
            stride=1,  # 1 step ahead forecasting
            retrain=False,
            train_length=None,  # only necessary when you want to retrain while forecasting 
            verbose=False,
        )

        # Save all important information
    
        pred.to_dataframe().to_csv(f"{save_dir}/pred_{period}.csv")
        train_set[period].to_dataframe().to_csv(f"{save_dir}/train_{period}.csv")
        test_set[period].to_dataframe().to_csv(f"{save_dir}/test_{period}.csv")


    

    
    
