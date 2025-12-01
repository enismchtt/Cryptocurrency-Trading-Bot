from data.binance_data import fetchData
import config
import os











if __name__ == "__main__" :

    for coin in config.coins_to_fetch :
        for timeframe in config.time_frames:
            
            os.makedirs(f"{config.data_path}/{coin}",exist_ok=True)
            fetchData(symbol=coin, timeframe=timeframe , as_csv=True)
    

