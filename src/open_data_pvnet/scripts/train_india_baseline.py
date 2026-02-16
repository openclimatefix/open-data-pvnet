"""
India PVNet Training Script - Solar-Only Baseline

This script trains a simple forecast model on India solar data.
Without NWP data (OCF GFS is UK-only), we use a solar-only approach:
- Historical solar generation patterns
- Solar position (time-based features)
- Day-of-week/month seasonality

For full PVNet with NWP, India-specific GFS data needs to be processed from NOAA.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"C:\Users\asus vivoBook\Desktop\New folder (2)\pvnet-india-data")
PROCESSED_DIR = BASE_DIR / "processed"


def load_india_data() -> pd.DataFrame:
    """Load India solar data from Zarr."""
    zarr_path = PROCESSED_DIR / "india_solar_2024-2025.zarr"
    ds = xr.open_zarr(str(zarr_path))
    
    df = ds.to_dataframe().reset_index()
    df = df.dropna(subset=['solar_generation_mw'])
    
    logger.info(f"Loaded {len(df)} rows of India solar data")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for solar prediction."""
    df = df.copy()
    
    # Ensure datetime is proper type
    df['datetime'] = pd.to_datetime(df['datetime_gmt'])
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Solar position approximation (simplified)
    # Peak solar at ~12-13 IST (6:30-7:30 UTC)
    df['hours_from_noon_utc'] = abs(df['hour'] - 6.5)  # Approximate India peak
    
    # Sine/cosine encoding for cyclic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Is daytime (approximate for India)
    df['is_daytime'] = ((df['hour'] >= 1) & (df['hour'] <= 12)).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """Create lagged features for time series."""
    df = df.copy().sort_values('datetime')
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    # Also add rolling averages
    for window in [3, 6, 12, 24]:
        df[f'{target_col}_roll_{window}h'] = df[target_col].rolling(window, min_periods=1).mean()
    
    return df


def train_baseline_model(df: pd.DataFrame):
    """Train a simple gradient boosting model as baseline."""
    logger.info("Training baseline model...")
    
    # Feature columns
    feature_cols = [
        'hour', 'hour_sin', 'hour_cos',
        'month', 'month_sin', 'month_cos',
        'day_of_week', 'day_of_year',
        'hours_from_noon_utc', 'is_daytime',
        'solar_generation_mw_lag_1h',
        'solar_generation_mw_lag_24h',
        'solar_generation_mw_roll_3h',
        'solar_generation_mw_roll_24h',
    ]
    
    # Filter valid rows
    df_train = df.dropna(subset=feature_cols + ['solar_generation_mw'])
    
    X = df_train[feature_cols]
    y = df_train['solar_generation_mw']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time series: no shuffle
    )
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"\n{'='*60}")
        logger.info("BASELINE MODEL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"MAE: {mae:.2f} MW")
        logger.info(f"RMSE: {rmse:.2f} MW")
        logger.info(f"Mean Solar: {y_test.mean():.2f} MW")
        logger.info(f"MAE/Mean: {mae/y_test.mean()*100:.1f}%")
        
        # Feature importance
        logger.info("\nTop Feature Importances:")
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model, mae, rmse
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, None, None


def main():
    logger.info("="*60)
    logger.info("INDIA PVNET - BASELINE TRAINING")
    logger.info("="*60)
    logger.info("Note: Using solar-only approach (no NWP data)")
    logger.info("")
    
    # Load data
    df = load_india_data()
    
    # Add features
    df = add_temporal_features(df)
    df = create_lag_features(df, 'solar_generation_mw', lags=[1, 2, 3, 6, 12, 24])
    
    logger.info(f"Features created. Shape: {df.shape}")
    
    # Train
    model, mae, rmse = train_baseline_model(df)
    
    if model is not None:
        logger.info("\n" + "="*60)
        logger.info("✅ Baseline training complete!")
        logger.info("="*60)
        logger.info("\nNext Steps:")
        logger.info("1. Add NWP data (needs NOAA GFS processing for India)")
        logger.info("2. Integrate with full PVNet model architecture")
        logger.info("3. Compare with persistence baseline")
    else:
        logger.error("❌ Training failed")


if __name__ == "__main__":
    main()
