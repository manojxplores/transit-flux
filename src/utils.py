import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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
    

def evaluate_conformal_predictions(results_df, alpha=0.1):
    df = results_df.copy()
    df['interval_width'] = df['Upper_Bound'] - df['Lower_Bound']
    df['covered'] = ((df['Actual'] >= df['Lower_Bound']) &
                     (df['Actual'] <= df['Upper_Bound'])).astype(int)

    coverage = df['covered'].mean()                    
    avg_width = df['interval_width'].mean()        
    mae = mean_absolute_error(df['Actual'], df['Prediction'])
    mape = mean_absolute_percentage_error(df['Actual'], df['Prediction'])
    r2 = r2_score(df['Actual'], df['Prediction'])

    metrics = {
        'empirical_coverage': round(coverage, 3)*100,
        'coverage_error': round(coverage - (1 - alpha), 3),
        'avg_interval_width': round(avg_width, 3),
        'MAE': round(mae, 3),
        'MAPE': round(mape, 3),
        'R2_Score': round(r2, 3)
    }

    print("\nCP evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")

    return metrics