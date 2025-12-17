from data.binance_data import fetchData
import config
import os
import itertools
from forecast import forecast_model
#from get_rmse import get_rmse_scores
from calc_rmse import evaluate_all_models



if __name__ == "__main__" :

    if not config.data_path == "new_dataset" : 
        # fetches current dataset
        for coin in config.coins_to_fetch :
            for timeframe in config.time_frames:
                
                os.makedirs(f"{config.data_path}/{coin}",exist_ok=True)
                fetchData(symbol=coin, timeframe=timeframe , as_csv=True)


    # The feature that MUST be in every list
  
    """all_combinations = []

    # Generate combinations of length 0 up to length 6
    for r in range(len(config.input_types) + 1):
        # itertools.combinations returns tuples, so we convert to list
        for combo in itertools.combinations(config.input_types, r):
            # Combine the optional parts with the mandatory part
            full_combo = list(combo) + [config.pred]
            all_combinations.append(full_combo)


    # dataset path is baseline_dataset
    for coin in config.coins_to_fetch :
        forecast_model(input_combinations=all_combinations,model_name="XGBOOST",forecast_type=config.pred, coin=coin , time_frame="1d")"""
    


    evaluate_all_models()
    #get_rmse_scores()




    


    









    

