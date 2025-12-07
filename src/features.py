import numpy as np
import pandas as pd
import holidays

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

