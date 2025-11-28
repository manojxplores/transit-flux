import numpy as np
import pandas as pd
import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import warnings
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

GDRIVE_BTP_PATH = '../urbanbus-forecast'
DATA_BASE_DIR = os.path.join(GDRIVE_BTP_PATH, 'urbanbus_data')
OUTPUT_FILENAME = os.path.join(GDRIVE_BTP_PATH, 'lgbm_model_metrics_new.csv')
FORECAST_OUTPUT_DIR = os.path.join(GDRIVE_BTP_PATH, 'forecasts_lgbm_new')
os.makedirs(FORECAST_OUTPUT_DIR, exist_ok=True)

def feature_engineering(df):
    # Basic datetime features
    df['hour'] = df['Ride_start_datetime'].dt.hour
    df['minute'] = df['Ride_start_datetime'].dt.minute
    df['day'] = df['Ride_start_datetime'].dt.day
    df['dayofweek'] = df['Ride_start_datetime'].dt.dayofweek
    df['month'] = df['Ride_start_datetime'].dt.month
    df['year'] = df['Ride_start_datetime'].dt.year
    df['week_of_year'] = df['Ride_start_datetime'].dt.isocalendar().week

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Weekend and holiday flag
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    china_holidays = holidays.country_holidays('CN')
    df['is_holiday'] = df['Ride_start_datetime'].dt.date.apply(lambda x: x in china_holidays).astype(int)

    cutoff_date = df['Ride_start_datetime'].max() - pd.Timedelta(days=28)
    train_mask = df['Ride_start_datetime'] <= cutoff_date
    test_mask = df['Ride_start_datetime'] > cutoff_date

    peak_hours = df[train_mask].groupby('hour')['Passenger_Count'].sum().nlargest(2).index.tolist()
    df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)

    # Lag features
    for lag in [1, 2, 3, 4, 8, 12, 24]:
        df[f'lag_{lag}'] = df['Passenger_Count'].shift(lag)

    # Rolling features
    for window in [4, 8, 12, 24]:
        shifted_data = df['Passenger_Count'].shift(1)
        df[f'rolling_mean_{window}'] = shifted_data.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = shifted_data.rolling(window=window, min_periods=1).std()

    # Drop rows with NaNs from lag/rolling features
    lag_roll_cols = [col for col in df.columns if col.startswith(('lag_', 'rolling_'))]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)
    return df


# Helper Functions
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def evaluate_model(y_true, y_pred, set_name="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{set_name} Performance:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  sMAPE: {smape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'sMAPE': smape}

def get_fit_status(train_r2, test_r2):
    r2_gap = train_r2 - test_r2
    if train_r2 < 0.5:
        return "Underfitting"
    elif r2_gap > 0.15:
        return "Overfitting"
    elif r2_gap < -0.05:
        return "Check Data (Unusual)"
    else:
        return "Good Fit"

all_results = []
processed_routes = set()

if os.path.exists(OUTPUT_FILENAME):
    print(f"Found existing results file: {OUTPUT_FILENAME}")
    try:
        results_df = pd.read_csv(OUTPUT_FILENAME)
        all_results = results_df.to_dict('records')
        processed_routes = set(results_df['route_name'])
        print(f"Loaded {len(processed_routes)} previously processed routes.")
    except Exception as e:
        print(f"Warning: Could not read results file. Starting from scratch. Error: {e}")
        all_results = []
        processed_routes = set()
else:
    print("No existing results file found. Starting a new run.")


N_TRIALS = 30

csv_files = [f for f in os.listdir(DATA_BASE_DIR) if f.endswith('.csv')]
print(f"Found {len(csv_files)} routes to process...")

for file in csv_files:
    if file in processed_routes:
        print(f"Skipping {file}: Already processed.")
        continue

    print(f"\n{'='*20} Processing Route: {file} {'='*20}")
    file_path = os.path.join(DATA_BASE_DIR, file)

    try:
        df = pd.read_csv(file_path)
        df = df.groupby(["Ride_start_datetime", "Bus_Service_Number", "Direction"], as_index=False)["Passenger_Count"].sum()
        df['Ride_start_datetime'] = pd.to_datetime(df['Ride_start_datetime'], errors='coerce')

        if df['Ride_start_datetime'].isnull().all():
            print(f"Skipping {file}: All dates are invalid after coercion.")
            continue

        df = df.sort_values('Ride_start_datetime').reset_index(drop=True)
        df = feature_engineering(df)

        cutoff_date = df['Ride_start_datetime'].max() - pd.Timedelta(days=28)
        train_mask = df['Ride_start_datetime'] <= cutoff_date
        test_mask = df['Ride_start_datetime'] > cutoff_date

        X = df.drop(columns=["Ride_start_datetime", "Bus_Service_Number", "Direction", "Passenger_Count"])
        y = df["Passenger_Count"]

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")


        # LightGBM
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                    "objective": "regression",
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 30),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 20.0, log=True),
                    "random_state": 42,
                    "verbose": -1
                }

            fold_mae = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)

                preds = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                fold_mae.append(mae)

            return np.mean(fold_mae)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print(f"\nBest MAE: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = -1

        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_train, y_train)

        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)

        train_metrics = evaluate_model(y_train, y_pred_train, "Train")
        test_metrics = evaluate_model(y_test, y_pred_test, "Test")

        fit_status = get_fit_status(train_metrics['R2'], test_metrics['R2'])

        forecast_df = pd.DataFrame({
            'Ride_start_datetime': df[test_mask]['Ride_start_datetime'],
            'Actual_Passenger_Count': y_test,
            'Predicted_Passenger_Count': y_pred_test
        })
        forecast_filename = f"forecast_{file}"
        forecast_save_path = os.path.join(FORECAST_OUTPUT_DIR, forecast_filename)
        forecast_df.to_csv(forecast_save_path, index=False)

        res_lgbm = {
                'route_name': file,
                'model': 'LightGBM',
                'Fit_Status': fit_status,
                'train_R2': train_metrics['R2'],
                'test_R2': test_metrics['R2'],
                'train_MAE': train_metrics['MAE'],
                'test_MAE': test_metrics['MAE'],
                'train_RMSE': train_metrics['RMSE'],
                'test_RMSE': test_metrics['RMSE'],
                'test_MAPE': test_metrics['MAPE'],
                'test_sMAPE': test_metrics['sMAPE']
            }
        all_results.append(res_lgbm)

    except Exception as e:
        print(f"FAILED to process {file}. Error: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = [
                'route_name', 'model', 'Fit_Status',
                'train_R2', 'test_R2', 'train_MAE', 'test_MAE',
                'train_RMSE', 'test_RMSE', 'test_MAPE', 'test_sMAPE'
            ]
        existing_cols = [c for c in cols if c in results_df.columns]
        results_df = results_df[existing_cols]

        results_df.to_csv(OUTPUT_FILENAME, index=False)


# Conformal Predictions
# import numpy as np
# import pandas as pd
# import holidays
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
# import lightgbm as lgb
# import optuna
# import matplotlib.pyplot as plt
# import warnings
# import os
# from datetime import timedelta

# optuna.logging.set_verbosity(optuna.logging.WARNING)
# warnings.filterwarnings('ignore')

# GDRIVE_BTP_PATH = '../urbanbus-forecast'
# DATA_BASE_DIR = os.path.join(GDRIVE_BTP_PATH, 'urbanbus_data')
# OUTPUT_FILENAME = os.path.join(GDRIVE_BTP_PATH, 'conformal_preds_metrics.csv')
# CONFORMAL_PREDS_DIR = os.path.join(GDRIVE_BTP_PATH, 'conformal_preds')
# os.makedirs(CONFORMAL_PREDS_DIR, exist_ok=True)

# def feature_engineering(df):
#     # Basic datetime features
#     df['hour'] = df['Ride_start_datetime'].dt.hour
#     df['minute'] = df['Ride_start_datetime'].dt.minute
#     df['day'] = df['Ride_start_datetime'].dt.day
#     df['dayofweek'] = df['Ride_start_datetime'].dt.dayofweek
#     df['month'] = df['Ride_start_datetime'].dt.month
#     df['year'] = df['Ride_start_datetime'].dt.year
#     df['week_of_year'] = df['Ride_start_datetime'].dt.isocalendar().week

#     # Cyclic encoding
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
#     df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
#     df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
#     df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
#     df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
#     df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

#     # Weekend and holiday flag
#     df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
#     china_holidays = holidays.country_holidays('CN')
#     df['is_holiday'] = df['Ride_start_datetime'].dt.date.apply(lambda x: x in china_holidays).astype(int)

#     cutoff_date = df['Ride_start_datetime'].max() - pd.Timedelta(days=28)
#     train_mask = df['Ride_start_datetime'] <= cutoff_date
#     test_mask = df['Ride_start_datetime'] > cutoff_date

#     peak_hours = df[train_mask].groupby('hour')['Passenger_Count'].sum().nlargest(2).index.tolist()
#     df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)

#     # Lag features
#     for lag in [1, 2, 3, 4, 8, 12, 24]:
#         df[f'lag_{lag}'] = df['Passenger_Count'].shift(lag)

#     # Rolling features
#     for window in [4, 8, 12, 24]:
#         shifted_data = df['Passenger_Count'].shift(1)
#         df[f'rolling_mean_{window}'] = shifted_data.rolling(window=window, min_periods=1).mean()
#         df[f'rolling_std_{window}'] = shifted_data.rolling(window=window, min_periods=1).std()

#     # Drop rows with NaNs from lag/rolling features
#     lag_roll_cols = [col for col in df.columns if col.startswith(('lag_', 'rolling_'))]
#     df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)
#     return df

# def evaluate_conformal_predictions(results_df, alpha=0.1):
#     df = results_df.copy()
#     df['interval_width'] = df['Upper_Bound'] - df['Lower_Bound']
#     df['covered'] = ((df['Actual'] >= df['Lower_Bound']) &
#                      (df['Actual'] <= df['Upper_Bound'])).astype(int)

#     coverage = df['covered'].mean()                    
#     avg_width = df['interval_width'].mean()        
#     mae = mean_absolute_error(df['Actual'], df['Prediction'])
#     mape = mean_absolute_percentage_error(df['Actual'], df['Prediction'])
#     r2 = r2_score(df['Actual'], df['Prediction'])

#     metrics = {
#         'empirical_coverage': round(coverage, 3)*100,
#         'coverage_error': round(coverage - (1 - alpha), 3),
#         'avg_interval_width': round(avg_width, 3),
#         'MAE': round(mae, 3),
#         'MAPE': round(mape, 3),
#         'R2_Score': round(r2, 3)
#     }

#     print("\nCP evaluation metrics:")
#     for k, v in metrics.items():
#         print(f"{k:25s}: {v}")

#     return metrics

# all_results = []
# processed_routes = set()

# if os.path.exists(OUTPUT_FILENAME):
#     print(f"Found existing results file: {OUTPUT_FILENAME}")
#     try:
#         results_df = pd.read_csv(OUTPUT_FILENAME)
#         all_results = results_df.to_dict('records')
#         processed_routes = set(results_df['route_name'])
#         print(f"Loaded {len(processed_routes)} previously processed routes.")
#     except Exception as e:
#         print(f"Warning: Could not read results file. Starting from scratch. Error: {e}")
#         all_results = []
#         processed_routes = set()
# else:
#     print("No existing results file found. Starting a new run.")


# N_TRIALS = 30

# csv_files = [f for f in os.listdir(DATA_BASE_DIR) if f.endswith('.csv')]
# print(f"Found {len(csv_files)} routes to process...")

# for file in csv_files:
#     if file in processed_routes:
#         print(f"Skipping {file}: Already processed.")
#         continue

#     print(f"\n{'='*20} Processing Route: {file} {'='*20}")
#     file_path = os.path.join(DATA_BASE_DIR, file)

#     try:
#         df = pd.read_csv(file_path)
#         df = df.groupby(["Ride_start_datetime", "Bus_Service_Number", "Direction"], as_index=False)["Passenger_Count"].sum()
#         df['Ride_start_datetime'] = pd.to_datetime(df['Ride_start_datetime'], errors='coerce')

#         if df['Ride_start_datetime'].isnull().all():
#             print(f"Skipping {file}: All dates are invalid after coercion.")
#             continue

#         df = df.sort_values('Ride_start_datetime').reset_index(drop=True)
#         df = feature_engineering(df)

#         train_calib_df = df[df['Ride_start_datetime'] < df['Ride_start_datetime'].max() - timedelta(days=28)].copy()
#         test_df = df[df['Ride_start_datetime'] >= df['Ride_start_datetime'].max() - timedelta(days=28)].copy()

#         train_df = train_calib_df[train_calib_df['Ride_start_datetime'] < train_calib_df['Ride_start_datetime'].max() - timedelta(days=21)].copy()
#         calib_df = train_calib_df[train_calib_df['Ride_start_datetime'] >= train_calib_df['Ride_start_datetime'].max() - timedelta(days=21)].copy()

#         X = df.drop(columns=["Ride_start_datetime", "Bus_Service_Number", "Direction", "Passenger_Count"])
#         y = df["Passenger_Count"]

#         cat_cols = ['Bus_Service_Number', 'Direction']
#         num_cols = [
#             'hour', 'minute', 'day', 'dayofweek', 'month', 'year', 'week_of_year',
#             'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos',
#             'month_sin', 'month_cos', 'is_weekend', 'is_holiday', 'is_peak_hour'
#         ]
#         lag_roll_cols = [col for col in train_df.columns if col.startswith(('lag_', 'rolling_'))]
#         features = num_cols + lag_roll_cols

#         X_train = train_df[features].copy()
#         y_train = train_df['Passenger_Count'].copy()
#         X_calib = calib_df[features].copy()
#         y_calib = calib_df['Passenger_Count'].copy()
#         X_test = test_df[features].copy()
#         y_test = test_df['Passenger_Count'].copy()


#         tscv = TimeSeriesSplit(n_splits=3)

#         def objective(trial):
#             params = {
#                 "objective": "regression",
#                 "n_estimators": trial.suggest_int("n_estimators", 100, 500),
#                 "max_depth": trial.suggest_int("max_depth", 3, 8),
#                 "num_leaves": trial.suggest_int("num_leaves", 15, 40),
#                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
#                 "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
#                 "subsample": trial.suggest_float("subsample", 0.5, 0.9),
#                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
#                 "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
#                 "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
#                 "random_state": 42,
#                 "verbose": -1
#             }
            
#             fold_mae = []
#             for train_idx, val_idx in tscv.split(X_train):
#                 X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#                 y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
#                 model = lgb.LGBMRegressor(**params)
#                 model.fit(X_tr, y_tr)
                
#                 preds = model.predict(X_val)
#                 fold_mae.append(mean_absolute_error(y_val, preds))
            
#             return np.mean(fold_mae)

#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=30)
#         best_params = study.best_params

#         print(f"\nBest MAE: {study.best_value:.4f}")
#         print(f"Best parameters: {study.best_params}")

#         models = {}
#         for alpha in [0.1, 0.5, 0.9]:
#             quantile_params = best_params.copy()
#             quantile_params.update({"objective": "quantile", "alpha": alpha})
#             model = lgb.LGBMRegressor(**quantile_params)
#             model.fit(X_train, y_train)
#             models[alpha] = model

#         cal_lower = models[0.1].predict(X_calib)
#         cal_upper = models[0.9].predict(X_calib)
#         scores = np.maximum(0, np.maximum(cal_lower - y_calib, y_calib - cal_upper))

#         alpha_cqr = 0.1
#         q_hat = np.quantile(scores, 1 - alpha_cqr)
#         print(f"q_hat: {q_hat:.3f}")

#         y_pred_lower = models[0.1].predict(X_test) - q_hat
#         y_pred_upper = models[0.9].predict(X_test) + q_hat
#         y_pred_median = models[0.5].predict(X_test)


#         conformal_preds_df = pd.DataFrame({
#             'Ride_start_datetime': test_df['Ride_start_datetime'].values,
#             'Actual': y_test.values,
#             'Prediction': y_pred_median,
#             'Lower_Bound': y_pred_lower,
#             'Upper_Bound': y_pred_upper
#         })

#         conformal_preds_filename = f"cp_{file}"
#         cp_save_path = os.path.join(CONFORMAL_PREDS_DIR, conformal_preds_filename)
#         conformal_preds_df.to_csv(cp_save_path, index=False)

#         metrics = evaluate_conformal_predictions(conformal_preds_df, alpha=0.1)
#         res_cp = {
#                 'route_name': file,
#                 'model': 'LightGBM',
#                 'empirical_coverage': metrics['empirical_coverage'],
#                 'coverage_error': metrics['coverage_error'],
#                 'avg_interval_width': metrics['avg_interval_width'],
#                 'MAE': metrics['MAE'],
#                 'MAPE': metrics['MAPE'],
#                 'R2_Score': metrics['R2_Score']
#                 }
#         all_results.append(res_cp)

#     except Exception as e:
#         print(f"FAILED to process {file}. Error: {e}")

#     if all_results:
#         results_df = pd.DataFrame(all_results)
#         cols = [
#                 'route_name', 'model', 'emperical_coverage',
#                 'coverage_error', 'avg_interval_width', 'MAE',
#                 'MAPE', 'R2_Score'
#             ]
#         existing_cols = [c for c in cols if c in results_df.columns]
#         results_df = results_df[existing_cols]

#         results_df.to_csv(OUTPUT_FILENAME, index=False)