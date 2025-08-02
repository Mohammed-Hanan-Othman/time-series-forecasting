from fastapi import FastAPI, HTTPException
import traceback
import pickle
import datetime
from schemas import PredictionRequest
from utils import create_rev_forecast_seed, generate_rev_forecast_features, run_rev_recursive_forecast

with open("./models/lightgbm_revenue.pkl", "rb") as file:
    model_revenue = pickle.load(file)

app = FastAPI()

@app.post("/predict/revenue", status_code=200)
def predict_revenue(request:PredictionRequest):
    """
    Generates a revenue forecast using the last 30+ days of historical data.

    Args:
        request (PredictionRequest): List of DailyRecord entries with required fields.

    Returns:
        JSON: Dictionary with forecasted revenue for each of the days in the forecast period.
    """
    if (len(request.data) < 30 ):
        raise HTTPException(
            status_code=400,
            detail={"error": "You would need to send at least 30 days of data for accurate predictions"}
        )
    try:
        forecast_period = request.forecast_days
        df = create_rev_forecast_seed(request.data)
        new_df = generate_rev_forecast_features(df, forecast_period)
        print("The columns are :zz",new_df.columns)
        forecast_series = (
            run_rev_recursive_forecast(new_df,forecast_period,model_revenue)
            .round(2)
        )
        return {
            "success":True,
            "data": {
                "forecast_days": forecast_period,
                "units": "GHS",
                "product_id": request.product_id,
                "generated_at": datetime.datetime.now(),
                "predictions": forecast_series.to_dict()
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is up and running!"}
