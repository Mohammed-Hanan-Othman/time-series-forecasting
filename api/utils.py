from schemas import DailyRecord
from typing import List
import pandas as pd

FEATURES_REVENUE = [
    "dayofweek", "month", "Holiday_Promotion", "Discount", "Price", "Competitor_Pricing",
    "promo_weekend", "discount_promo", "price_competition", "discount_x_pricegap",
    "revenue_lag_1", "revenue_lag_7", "rev_rolling_7_mean", "rev_rolling_14_std",
    "rev_momentum", "rev_rolling_3_std", "days_since_discount", "days_since_promo"
]

def create_rev_forecast_seed(data: List[DailyRecord]) -> pd.DataFrame:
    """
    Converts the last N days of structured DailyRecord inputs into a base DataFrame
    for forecasting revenue.

    This function:
    - Converts Pydantic models to a Pandas DataFrame
    - Parses the Date field and sets it as index
    - Normalizes the Discount field to a decimal format
    - Computes Revenue for known days using Units_Sold, Price, and Discount

    This seed DataFrame will later be used to generate lag-based and rolling features
    for recursive multi-step forecasting.

    Parameters:
        data (List[DailyRecord]): A list of DailyRecord inputs (at least 30 days).

    Returns:
        pd.DataFrame: A DataFrame indexed by Date with normalized and base-level features.
    """
    try:
        # Convert list of DailyRecord to DataFrame
        df = pd.DataFrame([record.model_dump() for record in data])

        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Sort by Date to ensure temporal order
        df = df.sort_values("Date")

        # Set Date as index
        df.set_index("Date", inplace=True)

        # Normalize discount
        df["Discount"] = df["Discount"] / 100.0

        # Calculate Revenue (for known days only)
        df["Revenue"] = df["Units_Sold"] * df["Price"] * (1 - df["Discount"])

        # Drop `Product_ID` column
        df.drop(columns=["Product_ID"], inplace=True)
        return df
    except Exception as e:
        print("❌ Error in create_forecast_seed:", e)

def generate_rev_forecast_features(df_seed: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """
    Generate full feature set for forecasting, including lag and rolling features.
    Adds N future rows and simulates features using past and predicted revenue.

    Args:
        df_seed (pd.DataFrame): DataFrame of the last 30+ days, including 'Revenue'.
        forecast_horizon (int): How many future days to forecast (e.g., 7 or 14).

    Returns:
        pd.DataFrame: Feature-complete DataFrame for all future steps.
    """
    try:    
        df = df_seed.copy()
        last_date = df.index[-1]

        # Generate empty rows for future days
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        future_df = pd.DataFrame(index=future_dates)
        
        # Copy over static/planned features to future rows
        static_cols = ["Product_ID", "Price", "Discount", "Competitor_Pricing", "Holiday_Promotion"]
        for col in static_cols:
            if col in df.columns:
                future_df[col] = df[col].iloc[-1]

        # Append to seed
        df_full = pd.concat([df, future_df])

        return df_full
    except Exception as e:
        print(e)

def recompute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes lag and rolling-based features for the given revenue DataFrame.
    This is used during recursive forecasting to update features after each prediction.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'Revenue' and time-based columns.

    Returns:
        pd.DataFrame: Updated DataFrame with all engineered features.
    """
    df = df.copy()
    
    # --- Time-based features ---
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # --- Interaction features ---
    df["promo_weekend"] = ((df["Holiday_Promotion"] == 1) & (df["dayofweek"] >= 5)).astype(int)
    df["discount_promo"] = df["Discount"] * df["Holiday_Promotion"]
    df["price_competition"] = df["Competitor_Pricing"] - df["Price"]
    df["discount_x_pricegap"] = df["Discount"] * df["price_competition"]

    # --- Lag & Rolling features ---
    df["revenue_lag_1"] = df["Revenue"].shift(1)
    df["revenue_lag_7"] = df["Revenue"].shift(7)
    df["rev_rolling_7_mean"] = df["Revenue"].shift(1).rolling(window=7).mean()
    df["rev_rolling_14_std"] = df["Revenue"].shift(1).rolling(window=14).std()
    df["rev_momentum"] = df["revenue_lag_1"] - df["revenue_lag_7"]
    df["rev_rolling_3_std"] = df["Revenue"].shift(1).rolling(window=3).std()

    # --- Days since last discount ---
    discount_active = df["Discount"] > 0
    df["days_since_discount"] = (~discount_active).astype(int)
    df["days_since_discount"] = (
        df["days_since_discount"].cumsum()
        - df["days_since_discount"].cumsum().where(discount_active).ffill().fillna(0).astype(int)
    )

    # --- Days since last promo ---
    promo_active = df["Holiday_Promotion"] == 1
    df["days_since_promo"] = (~promo_active).astype(int)
    df["days_since_promo"] = (
        df["days_since_promo"].cumsum()
        - df["days_since_promo"].cumsum().where(promo_active).ffill().fillna(0).astype(int)
    )

    return df

def run_rev_recursive_forecast(df_full: pd.DataFrame, forecast_horizon: int, model) -> pd.Series:
    """
    Runs a rolling forecast using a trained model and engineered features.

    This function predicts the next N days of revenue one day at a time,
    updating the dataset with each new prediction to recalculate lag/rolling features
    for the next day's prediction.

    Args:
        df_full (pd.DataFrame): DataFrame with historical + empty future rows
                                (output of `generate_rev_forecast_features`).
        forecast_horizon (int): Number of future days to forecast (e.g., 7 or 14).
        model: Trained LightGBM (or similar) model.

    Returns:
        pd.Series: Forecasted revenue values, indexed by future dates.
    """
    predictions = []

    for day in range(1, forecast_horizon + 1):
        # Get the current date to predict
        forecast_date = df_full.index[-forecast_horizon + day - 1]

        # STEP 1: Extract past data up to the day before the forecast
        df_past = df_full.loc[:forecast_date - pd.Timedelta(days=1)].copy()

        # STEP 2: Recompute lag/rolling features using updated past
        df_past = recompute_lag_features(df_past)

        # Copy the last row of engineered features into current forecast date
        engineered_features = df_past.iloc[-1]
        df_full.loc[forecast_date, engineered_features.index] = engineered_features

        # STEP 3: Predict revenue for the current forecast day
        X = df_full.loc[forecast_date, FEATURES_REVENUE].values.reshape(1, -1)
        y_pred = model.predict(X)[0]

        # Save prediction to df_full for use in subsequent steps
        df_full.at[forecast_date, "Revenue"] = y_pred
        predictions.append((forecast_date, y_pred))

    # Return only the predicted future values as a Series
    return pd.Series({date: pred for date, pred in predictions})
