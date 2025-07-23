# 📈📉 Sales Prediction using Time Series Analysis

This project focuses on performing time series analysis on sales data to identify trends, patterns, and forecast future sales performance. The goal is to support data-driven business decisions around planning, budgeting, and strategic growth. The analysis and modeling are conducted using Python in a Jupyter Notebook, with final models and forecasts saved in the `models/` folder for future use and evaluation.

## 📁 Project Structure

<pre>
.
├── api/ # 💡 NEW: FastAPI app for model deployment
│ ├── app.py # API code that loads the model and handles prediction
│ ├── requirements.txt # Dependencies for the API
│ └── utils.py # Any preprocessing helper functions
├── data/ # Dataset links 
├── models/ # Saved trained models (pickle files)
├── notebooks/ # Jupyter Notebooks for exploration and model building
│ └── sales_prediction_lightgbm.ipynb
│ └── sales_prediction.ipynb
├── README.md # This files
└── .gitignore # Files and folders to ignore in version control
</pre>

##  📊 Dataset

Refer to the `data/` folder for details

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/Mohammed-Hanan-Othman/time-series-forecasting.git
cd time-series-forecasting
```

2. **Installation and Setup:**
This assumes an anaconda environment.

3. **Open notebook**
Open any of the notebooks
``` bash
jupyter notebook notebooks/sales_prediction.ipynb
jupyter notebook notebooks/sales_prediction_lightgbm.ipynb
```

4. **Run:**
    - Run all cells to reproduce the results.
    - Alternatively, create a new `.py` file and load the saved models from the `./models/` folder to make predictions


## 🧠 Model Overview

Different models were  used here:

1. **Base Model:**
    - Model type: ARIMA (order=(5,1,0)).
    - Notebook: [Base Model](./notebooks/sales_prediction.ipynb)
    - Evaluation Metrics:
        - Root Mean Square Error
    - Model Output: Model trained and saved as a `.pkl` file in `./models/` folder.
    - 📈 Results:
        - Walk forward validation RMSE: 74464.67

1. **Second Model:**
    - Model type: LGBMRegressor (LightGBM).
    - Notebook: [LightGBM model](./notebooks/sales_prediction_lightgbm.ipynb)
    - Feature Engineering:
        - Aggregated sales data to daily levels.
        - Added lag features for both units and revenue (such as `revenue_lag_7` and `units_lag_7`).
        - Added rolling features such as `rev_rolling_7_mean` and `units_rolling_7_mean`.
        - Added interactive features such as `promo_weekend`, `days_since_promo` and `price_competition` to make available to the model, the combined effects of some features.
        - Refer to the `Feature Engineering` section of the notebook.
    - Evaluation Metrics:
        - Root Mean Square Error
        - Mean Absolute Error
    - Model Output: Model trained and saved as a `.pkl` file in `./models/` folder.
    - Comments and Issues:
        - Significant improvement in evaluation as compared to the Base Model. This is owed to significant feature engineering, improving the model's "knowledge" about the data set.

            | Model        | Test RMSE         |
            |--------------|------------------|
            | ARIMA        | 74,464.67 (Revenue)    |
            | LightGBM     | 215.02 (Units) / 17,910.91 (Revenue) |

        - Spikes in residuals towards mid months and start of months, indicate that a quarterly or bi-weekly features would have improvement on the model's performance.
        
    - 📈 Results:
        - Units Model:
            - `Training MAE`: 96.76237129129042
            - `Training RMSE`: 156.7168398345387
            - `Test MAE`: 149.19269454223104
            - `Test RMSE`: 215.0246996291744
        - Revenue Model:
            - `Training MAE`: 4202.384341148988
            - `Training RMSE`: 6534.812370685227
            - `Test MAE`: 11333.096425190528
            - `Test RMSE`: 17910.90607771323

### 🛠 Feature Engineering Highlights (From the Second Model)
- Lag features: `units_lag_7`, `revenue_lag_7`
- Rolling stats: `units_rolling_7_mean`, `rev_rolling_7_mean`
- Interaction terms: `promo_weekend`, `price_competition`
- Custom recency features: `days_since_promo`

## 📌 Future Improvements

### 🔬 Modeling
- Try forecasting models like Prophet or XGBoost
- Use model ensembles for more robust results

### 📅 Feature Engineering
- Add bi-weekly, monthly, and quarter-end/start indicators
- Engineer seasonality features

### 📈 Interpretability
- Apply SHAP analysis to understand feature influence

### ☁️ Deployment
- Deploy model using FastAPI + Render or Railway
- Build an interactive dashboard to visualize predictions


## Contributing
Contributions are welcome. To contribute:

- Fork the repository
- Create a new branch (git checkout -b feature-xyz)
- Commit your changes (git commit -am 'Add new feature')
- Push to the branch (git push origin feature-xyz)
- Open a Pull Request

Contributors
- GitHub: @Mohammed-Hanan-Othman – Initial work

Feel free to open issues or suggestions via GitHub Issues.

## 📄 License
This project is for educational purposes only. Feel free to use and share.

## 👨‍💻 Author
- GitHub: @Mohammed-Hanan-Othman