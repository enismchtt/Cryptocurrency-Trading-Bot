import os
import pandas as pd
from darts import TimeSeries
from darts.metrics import rmse
from tqdm import tqdm
import config  # Ensure this points to your config file
import numpy as np

def evaluate_all_models():
    # 1. FIX: Point to the main 'model_predictions' folder
    # We scan from here because 'input_...' folders are now at the top level
    base_dir = f"{config.output_path}/model_predictions/XGBOOST" 
    
    results = []
    
    print(f"Scanning directories in: {base_dir} ...")
    
    for root, dirs, files in os.walk(base_dir):
        if "pred.csv" in files and "actual.csv" in files:
            try:
                # --- A. Load Data ---
                df_pred = pd.read_csv(os.path.join(root, "pred.csv"))
                df_actual = pd.read_csv(os.path.join(root, "actual.csv"))
                
                # --- FIX: Ensure 'date' column exists ---
                # If the CSV saved the date as "Unnamed: 0" or no header, rename the first column
                if 'date' not in df_pred.columns:
                    df_pred.rename(columns={df_pred.columns[0]: 'date'}, inplace=True)
                
                if 'date' not in df_actual.columns:
                    df_actual.rename(columns={df_actual.columns[0]: 'date'}, inplace=True)

                # Convert 'date' column to datetime objects
                df_pred['date'] = pd.to_datetime(df_pred['date'])
                df_actual['date'] = pd.to_datetime(df_actual['date'])
                
                ts_pred = TimeSeries.from_dataframe(df_pred, time_col='date', value_cols=df_pred.columns[-1])
                ts_actual = TimeSeries.from_dataframe(df_actual, time_col='date', value_cols=df_actual.columns[-1])
                
                # --- B. Calculate Metrics ---
                # 1. RMSE
                score_rmse = rmse(ts_actual, ts_pred)
                
                # 2. Directional Accuracy (FIXED)
                # We use .to_dataframe() instead of .pd_dataframe()
                df_check = pd.DataFrame()
                
                # We extract values safely
                df_check['actual'] = ts_actual.to_dataframe().iloc[:, 0].values
                df_check['pred'] = ts_pred.to_dataframe().iloc[:, 0].values
                
                # Calculate simple accuracy %
                df_check['correct'] = np.sign(df_check['actual']) == np.sign(df_check['pred'])
                dir_acc = df_check['correct'].mean() * 100

                # --- C. Get Metadata & FIX PATHS ---
                # Get features
                feature_file = os.path.join(root, "features.txt")
                features = open(feature_file, "r").read().strip() if os.path.exists(feature_file) else "Unknown"

                # ROBUST PATH PARSING
                # 1. Normalize path to fix Windows backslash issues
                norm_root = os.path.normpath(root)
                # 2. Split into parts
                parts = norm_root.split(os.sep)
                
                # Structure: .../input_NAME/TYPE/MODEL_NAME/COIN/TIMEFRAME
                # We grab from the END of the list to be safe
                time_frame = parts[-1]  # e.g. "1d"
                coin = parts[-2]        # e.g. "BTC"
                model_id = parts[-3]    # e.g. "LSTM_BTC_1d_input..." (The unique ID)

                results.append({
                    "Model_ID": model_id,
                    "Coin": coin,
                    "TimeFrame": time_frame,
                    "RMSE": score_rmse,
                    "Direction_Acc": f"{dir_acc:.2f}%",
                    "Features": features
                })
                
            except Exception as e:
                print(f"Skipping {root}: {e}")

    # 3. Save Results
    if not results:
        print("No models found! Check if 'config.output_path' is correct.")
        return

    df_results = pd.DataFrame(results)
    
    # Save directory
    save_dir = f"{config.output_path}/rmse/{config.model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # --- 4. Separate Files per Coin ---
    for coin in df_results['Coin'].unique():
        # Filter & Sort
        coin_df = df_results[df_results['Coin'] == coin].copy()
        coin_df = coin_df.sort_values(by="RMSE", ascending=True)
        
        # Save
        output_file = f"{save_dir}/leaderboard_{coin}.csv"
        coin_df.to_csv(output_file, index=False)
        
        # Print
        print("\n" + "="*80)
        print(f" TOP 5 PERFORMING MODELS FOR {coin} ")
        print("="*80)
        print(coin_df[['Features', 'RMSE', 'Direction_Acc']].head(5).to_string(index=False))
        print(f"\nSaved to: {output_file}")
