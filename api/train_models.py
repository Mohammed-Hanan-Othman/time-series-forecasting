import pandas as pd
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Load training data (for revenue model)
X_train_revenue = pd.read_csv("../data/X_train_revenue.csv").set_index("Date")
y_train_revenue = pd.read_csv("../data/y_train_revenue.csv").set_index("Date")
X_test_revenue = pd.read_csv("../data/X_test_revenue.csv").set_index("Date")
y_test_revenue = pd.read_csv("../data/y_test_revenue.csv").set_index("Date")

# Load training data (for units model)
X_train_units = pd.read_csv("../data/X_train_units.csv").set_index("Date")
y_train_units = pd.read_csv("../data/y_train_units.csv").set_index("Date")
X_test_units = pd.read_csv("../data/X_test_units.csv").set_index("Date")
y_test_units = pd.read_csv("../data/y_test_units.csv").set_index("Date")

# Train and fit revenue model
revenue_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=20,
    min_child_samples=30,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)
revenue_model.fit(
    X_train_revenue, y_train_revenue,
    eval_set=[(X_test_revenue, y_test_revenue)],
    eval_metric=['rmse', 'mae'],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)
    ]
)

# Train and fit revenue model
units_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=20,
    min_child_samples=30,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)
units_model.fit(
    X_train_units, y_train_units,
    eval_set=[(X_test_units, y_test_units)],
    eval_metric=['rmse', 'mae'],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)
    ]
)

# Save models
joblib.dump(revenue_model, "../models/lightgbm_revenue.pkl")
joblib.dump(units_model, "../models/lightgbm_units.pkl")
print("Models saved to ../models/")