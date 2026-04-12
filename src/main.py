from data.binance_data import fetchData
import config
import os
import itertools
from forecast import forecast_model
from calc_rmse import evaluate_all_models




if __name__ == "__main__" :



    if not os.path.exists(config.data_path): # use dataset naming with date for ex:  13.03.26
        # fetches current dataset
        os.makedirs(config.data_path)
        for coin in config.coins_to_fetch :
            for timeframe in config.time_frames:
                
                os.makedirs(f"{config.data_path}/{coin}",exist_ok=True)
                fetchData(symbol=coin, timeframe=timeframe , as_csv=True)


    # The feature that MUST be in every list
  
    all_combinations = []

    # Generate combinations of length 0 up to length 6
    for r in range(len(config.input_types) + 1):
        for combo in itertools.combinations(config.input_types, r):
            # Combine the optional parts with the mandatory part
            full_combo = list(combo) + [config.pred]
            all_combinations.append(full_combo)

    for combo in all_combinations:
        print(combo)


    # dataset path is baseline_dataset
    for coin in config.coins_to_fetch :
        forecast_model(input_combinations=all_combinations,model_name="LSTM",forecast_type=config.pred, coin=coin , time_frame="1d")
    


    evaluate_all_models()





    


    









    

