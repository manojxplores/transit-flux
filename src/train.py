import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna
import warnings
import os
from transit_flux.features import feature_engineering
from transit_flux.utils import evaluate_model, get_fit_status

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

BASE_DIR = '../urbanbus-forecast'
DATA_DIR = os.path.join(BASE_DIR, 'urbanbus_data')
OUTPUT_FILENAME = os.path.join(BASE_DIR, 'lgbm_model_metrics_new.csv')
FORECAST_OUTPUT_DIR = os.path.join(BASE_DIR, 'forecasts_lgbm_new')
os.makedirs(FORECAST_OUTPUT_DIR, exist_ok=True)

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

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"Found {len(csv_files)} routes to process...")

for file in csv_files:
    if file in processed_routes:
        print(f"Skipping {file}: Already processed.")
        continue

    print(f"\n{'='*20} Processing Route: {file} {'='*20}")
    file_path = os.path.join(DATA_DIR, file)

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
