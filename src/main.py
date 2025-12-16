from data.binance_data import fetchData
import config
import os

from forecast import forecast_model
from get_rmse import get_rmse_scores



if __name__ == "__main__" :

    if not config.data_path == "baseline_dataset" : 
        # fetches current dataset
        for coin in config.coins_to_fetch :
            for timeframe in config.time_frames:
                
                os.makedirs(f"{config.data_path}/{coin}",exist_ok=True)
                fetchData(symbol=coin, timeframe=timeframe , as_csv=True)
    else:
        # dataset path is baseline_dataset
        for coin in config.coins_to_fetch :
            forecast_model(model_name="LSTM" , forecast_type=config.pred, coin=coin , time_frame="1d")
    
    #get_rmse_scores()




    


    



    

