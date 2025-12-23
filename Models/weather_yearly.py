# weather_yearly_model.py
# Train RandomForest regression model on yearly weather dataset
#Model-A

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from math import sqrt
import joblib
import os

DATA_PATH = Path("data/state_weather_data_1997_2020.csv")
MODEL_DIR = Path("models/yearly/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COL_STATE = "state"
COL_YEAR = "year"
COL_TEMP = "avg_temp_c"
COL_RAIN = "total_rainfall_mm"
COL_HUM = "avg_humidity_percent"

def rmse(a, b): return sqrt(((a - b) ** 2).mean())

def train_yearly_weather_model():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values([COL_STATE, COL_YEAR])

    results = {}

    states = df[COL_STATE].unique()

    for state in states:
        df_s = df[df[COL_STATE] == state].copy()

        if len(df_s) < 7:
            print(f"Skipping {state}: Not enough data.")
            continue

        # lag features
        df_s["temp_lag_1"] = df_s[COL_TEMP].shift(1)
        df_s["rain_lag_1"] = df_s[COL_RAIN].shift(1)
        df_s["hum_lag_1"] = df_s[COL_HUM].shift(1)
        df_s = df_s.dropna()

        FEATURES = ["year", "temp_lag_1", "rain_lag_1", "hum_lag_1"]

        X = df_s[FEATURES]
        y_temp = df_s[COL_TEMP]
        y_rain = df_s[COL_RAIN]

        # split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        ytemp_train, ytemp_test = y_temp.iloc[:split], y_temp.iloc[split:]
        yrain_train, yrain_test = y_rain.iloc[:split], y_rain.iloc[:split]

        # train models
        rf_temp = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_rain = RandomForestRegressor(n_estimators=200, random_state=42)

        rf_temp.fit(X_train, ytemp_train)
        rf_rain.fit(X_train, yrain_train)

        # evaluate
        temp_pred = rf_temp.predict(X_test)
        rain_pred = rf_rain.predict(X_test)

        temp_mae = mean_absolute_error(ytemp_test, temp_pred)
        rain_mae = mean_absolute_error(y_rain.iloc[split:], rain_pred)

        # save
        safe_state = state.replace(" ", "_")
        joblib.dump(rf_temp, MODEL_DIR / f"{safe_state}_temp.pkl")
        joblib.dump(rf_rain, MODEL_DIR / f"{safe_state}_rain.pkl")

        results[state] = {
            "temp_mae": float(temp_mae),
            "rain_mae": float(rain_mae)
        }

        print(f"Trained yearly model for {state}: Temp MAE={temp_mae:.2f}, Rain MAE={rain_mae:.2f}")

    return results


def predict_next_year(state, year, temp_prev, rain_prev, hum_prev):
    safe_state = state.replace(" ", "_")

    temp_model = joblib.load(MODEL_DIR / f"{safe_state}_temp.pkl")
    rain_model = joblib.load(MODEL_DIR / f"{safe_state}_rain.pkl")

    X = pd.DataFrame([{
        "year": year,
        "temp_lag_1": temp_prev,
        "rain_lag_1": rain_prev,
        "hum_lag_1": hum_prev
    }])

    return {
        "pred_temp": float(temp_model.predict(X)[0]),
        "pred_rain": float(rain_model.predict(X)[0])
    }


# if __name__ == "__main__":
#     train_yearly_weather_model()

if __name__ == "__main__":
    print("Training yearly model...")            # <-- visible output
    summary = train_yearly_weather_model()
    print("\nTraining Summary:\n", summary)
    print("\nDONE!")
