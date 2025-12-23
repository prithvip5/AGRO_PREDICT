# crop_recommendation_pipeline.py
# SINGLE FILE: Training + Prediction for Crop Recommendation

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# FILE PATHS
# ===============================
# CROP_DATA_PATH = "Crop_recommendation.csv"
# YIELD_DATA_PATH = "crop_yield.csv"

CROP_DATA_PATH = r"C:\Users\PRITHVI\Anweshna Model\AGRO_PREDICT\Models\Data\Crop_recommendation.csv"
YIELD_DATA_PATH = r"C:\Users\PRITHVI\Anweshna Model\AGRO_PREDICT\Models\Data\crop_yield.csv"

MODEL_PATH = "crop_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
YIELD_TABLE_PATH = "yield_table.pkl"

# ===============================
# TRAINING FUNCTION
# ===============================
def train_model():
    print("\n--- TRAINING STARTED ---")

    # Load datasets
    df_crop = pd.read_csv(CROP_DATA_PATH)
    df_yield = pd.read_csv(YIELD_DATA_PATH)

    # Clean data
    df_crop = df_crop.dropna()
    df_yield = df_yield.dropna()

    # Normalize crop names
    df_crop["label"] = df_crop["label"].str.lower()
    df_yield["crop"] = df_yield["crop"].str.lower()

    # Create yield lookup table
    yield_table = (
        df_yield
        .groupby("crop")["yield"]
        .mean()
        .reset_index()
        .rename(columns={"yield": "avg_yield"})
    )

    # Prepare features and labels
    X = df_crop.drop(columns=["label"])
    y = df_crop["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save everything
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(yield_table, YIELD_TABLE_PATH)

    print("Model, encoder & yield table saved successfully.")

# ===============================
# PREDICTION FUNCTION
# ===============================
def recommend_crop(input_data, top_n=3):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    yield_table = joblib.load(YIELD_TABLE_PATH)

    df = pd.DataFrame([input_data])

    probabilities = model.predict_proba(df)[0]
    crops = encoder.inverse_transform(range(len(probabilities)))

    results = pd.DataFrame({
        "crop": crops,
        "probability": probabilities
    })

    # Merge yield data
    results = results.merge(
        yield_table, how="left", left_on="crop", right_on="crop"
    )
    results["avg_yield"] = results["avg_yield"].fillna(0)

    # Final score (suitability + yield)
    results["final_score"] = results["probability"] * (1 + results["avg_yield"])

    return results.sort_values("final_score", ascending=False).head(top_n)

# ===============================
# USER INPUT FUNCTION
# ===============================
def get_user_input():
    print("\nEnter Farmer Details:")
    return {
        "N": float(input("Nitrogen (N): ")),
        "P": float(input("Phosphorus (P): ")),
        "K": float(input("Potassium (K): ")),
        "temperature": float(input("Temperature (°C): ")),
        "humidity": float(input("Humidity (%): ")),
        "ph": float(input("Soil pH: ")),
        "rainfall": float(input("Rainfall (mm): "))
    }

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":

    # Train only if model doesn't exist
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print("Model already trained. Skipping training.")

    # Prediction
    user_input = get_user_input()
    recommendations = recommend_crop(user_input)

    print("\n--- CROP RECOMMENDATION ---")
    for idx, row in recommendations.iterrows():
        print(
            f"Crop: {row['crop'].title()} | "
            f"Suitability: {row['probability']:.2f} | "
            f"Avg Yield: {row['avg_yield']:.2f}"
        )
