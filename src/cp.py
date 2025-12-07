import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna
import os
from datetime import timedelta
from transit_flux.features import feature_engineering
from transit_flux.utils import evaluate_conformal_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = '../urbanbus-forecast'
DATA_DIR = os.path.join(BASE_DIR, 'urbanbus_data')
OUTPUT_FILENAME = os.path.join(BASE_DIR, 'conformal_preds_metrics.csv')
CONFORMAL_PREDS_DIR = os.path.join(BASE_DIR, 'conformal_preds')
os.makedirs(CONFORMAL_PREDS_DIR, exist_ok=True)

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

        train_calib_df = df[df['Ride_start_datetime'] < df['Ride_start_datetime'].max() - timedelta(days=28)].copy()
        test_df = df[df['Ride_start_datetime'] >= df['Ride_start_datetime'].max() - timedelta(days=28)].copy()

        train_df = train_calib_df[train_calib_df['Ride_start_datetime'] < train_calib_df['Ride_start_datetime'].max() - timedelta(days=21)].copy()
        calib_df = train_calib_df[train_calib_df['Ride_start_datetime'] >= train_calib_df['Ride_start_datetime'].max() - timedelta(days=21)].copy()

        X = df.drop(columns=["Ride_start_datetime", "Bus_Service_Number", "Direction", "Passenger_Count"])
        y = df["Passenger_Count"]

        cat_cols = ['Bus_Service_Number', 'Direction']
        num_cols = [
            'hour', 'minute', 'day', 'dayofweek', 'month', 'year', 'week_of_year',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos', 'is_weekend', 'is_holiday', 'is_peak_hour'
        ]
        lag_roll_cols = [col for col in train_df.columns if col.startswith(('lag_', 'rolling_'))]
        features = num_cols + lag_roll_cols

        X_train = train_df[features].copy()
        y_train = train_df['Passenger_Count'].copy()
        X_calib = calib_df[features].copy()
        y_calib = calib_df['Passenger_Count'].copy()
        X_test = test_df[features].copy()
        y_test = test_df['Passenger_Count'].copy()


        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                "objective": "regression",
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 40),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
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
                fold_mae.append(mean_absolute_error(y_val, preds))
            
            return np.mean(fold_mae)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        best_params = study.best_params

        print(f"\nBest MAE: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        models = {}
        for alpha in [0.1, 0.5, 0.9]:
            quantile_params = best_params.copy()
            quantile_params.update({"objective": "quantile", "alpha": alpha})
            model = lgb.LGBMRegressor(**quantile_params)
            model.fit(X_train, y_train)
            models[alpha] = model

        cal_lower = models[0.1].predict(X_calib)
        cal_upper = models[0.9].predict(X_calib)
        scores = np.maximum(0, np.maximum(cal_lower - y_calib, y_calib - cal_upper))

        alpha_cqr = 0.1
        q_hat = np.quantile(scores, 1 - alpha_cqr)
        print(f"q_hat: {q_hat:.3f}")

        y_pred_lower = models[0.1].predict(X_test) - q_hat
        y_pred_upper = models[0.9].predict(X_test) + q_hat
        y_pred_median = models[0.5].predict(X_test)


        conformal_preds_df = pd.DataFrame({
            'Ride_start_datetime': test_df['Ride_start_datetime'].values,
            'Actual': y_test.values,
            'Prediction': y_pred_median,
            'Lower_Bound': y_pred_lower,
            'Upper_Bound': y_pred_upper
        })

        conformal_preds_filename = f"cp_{file}"
        cp_save_path = os.path.join(CONFORMAL_PREDS_DIR, conformal_preds_filename)
        conformal_preds_df.to_csv(cp_save_path, index=False)

        metrics = evaluate_conformal_predictions(conformal_preds_df, alpha=0.1)
        res_cp = {
                'route_name': file,
                'model': 'LightGBM',
                'empirical_coverage': metrics['empirical_coverage'],
                'coverage_error': metrics['coverage_error'],
                'avg_interval_width': metrics['avg_interval_width'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2_Score': metrics['R2_Score']
                }
        all_results.append(res_cp)

    except Exception as e:
        print(f"FAILED to process {file}. Error: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = [
                'route_name', 'model', 'emperical_coverage',
                'coverage_error', 'avg_interval_width', 'MAE',
                'MAPE', 'R2_Score'
            ]
        existing_cols = [c for c in cols if c in results_df.columns]
        results_df = results_df[existing_cols]

        results_df.to_csv(OUTPUT_FILENAME, index=False)