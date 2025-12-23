# weather_observation_model.py
# Train model on observation dataset (Temp, Condition, Humidity)
#Model-B

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

DATA_PATH = Path("data/weather-1.csv")
MODEL_DIR = Path("models/observation/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COL_STATE = "state"
COL_DIST = "Distict"
COL_TEMP = "Temperature"
COL_COND = "Condition"
COL_HUM = "Humidity"

def train_observation_model():
    df = pd.read_csv(DATA_PATH)

    # Clean missing values
    df = df.dropna(subset=[COL_TEMP, COL_HUM, COL_COND])

    # Temperature Model
    X_temp = df[[COL_HUM]]   # you can add more features
    y_temp = df[COL_TEMP]

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    rf_temp = RandomForestRegressor(n_estimators=200)
    rf_temp.fit(X_train, y_train)

    temp_pred = rf_temp.predict(X_test)
    mae_temp = mean_absolute_error(y_test, temp_pred)

    joblib.dump(rf_temp, MODEL_DIR / "temp_model.pkl")
    print(f"Temperature model trained. MAE={mae_temp:.2f}")

    # Condition Classifier
    X_cond = df[[COL_TEMP, COL_HUM]]
    y_cond = df[COL_COND]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cond, y_cond, test_size=0.2, random_state=42)

    rf_cond = RandomForestClassifier(n_estimators=200)
    rf_cond.fit(Xc_train, yc_train)

    cond_pred = rf_cond.predict(Xc_test)
    acc = accuracy_score(yc_test, cond_pred)

    joblib.dump(rf_cond, MODEL_DIR / "condition_model.pkl")
    print(f"Condition model trained. Accuracy={acc:.2f}")

    return mae_temp, acc


def predict_observation(temp=None, humidity=None):
    rf_temp = joblib.load(MODEL_DIR / "temp_model.pkl")
    rf_cond = joblib.load(MODEL_DIR / "condition_model.pkl")

    df_input = pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity
    }])

    predicted_temp = rf_temp.predict(df_input[["Humidity"]])[0]
    predicted_cond = rf_cond.predict(df_input)[0]

    return {
        "pred_temperature": float(predicted_temp),
        "pred_condition": predicted_cond
    }


# if __name__ == "__main__":
#     train_observation_model()

if __name__ == "__main__":
    print("Training observation model...")       # <-- visible output
    results = train_observation_model()
    print("\nModel training complete.")
    print("Results:", results)
    print("\nDONE!")
