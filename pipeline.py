# pipeline.py – recreate grievance_pipeline.joblib
# Run this once to train and save the pipeline used by your Flask app.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------------- CONFIG -----------------
DATA_FILE = Path(r"D:\urban grievance folder\311_Service_Requests_from_2010_to_Present_20251124.csv")
MODEL_FILE = Path(r"D:\urban grievance folder\grievance_pipeline.joblib")
N_ROWS = 100_000   # you can increase if your PC can handle it
# ------------------------------------------


def main():
    print("Loading data from:", DATA_FILE)
    df = pd.read_csv(DATA_FILE, nrows=N_ROWS)

    # 1) Keep only the columns we care about (as in your code)
    df = df[[
        "Created Date",
        "Agency Name",
        "Complaint Type",
        "Descriptor",
        "Status",
        "Borough",
        "Latitude",
        "Longitude",
        "Location"
    ]]
    print("\nOriginal shape:", df.shape)

    # 2) Basic cleaning
    df = df.drop_duplicates()
    df["Descriptor"] = df["Descriptor"].fillna("Unknown")
    df = df.dropna(subset=["Latitude", "Longitude", "Location"]).copy()

    print("After dropping NA lat/lon/location:", df.shape)

    # 3) Datetime features
    df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
    df = df.dropna(subset=["Created Date"]).copy()
    df["Hour"] = df["Created Date"].dt.hour
    df["DayOfWeek"] = df["Created Date"].dt.dayofweek
    df["Month"] = df["Created Date"].dt.month

    # 4) Drop unused columns
    df = df.drop(columns=["Created Date", "Status", "Location"])

    # 5) Handle rare labels for Complaint Type
    freq = df["Complaint Type"].value_counts()
    rare_labels = freq[freq < 30].index
    df["Complaint Type"] = df["Complaint Type"].replace(rare_labels, "Other")

    print("\nComplaint Type counts (after grouping rare into 'Other'):")
    print(df["Complaint Type"].value_counts().head())

    # 6) Split features/target
    X = df.drop(columns=["Complaint Type"])
    y = df["Complaint Type"]

    # Columns we will use (this must match what your Flask app sends)
    text_col = "Descriptor"
    cat_cols = ["Agency Name", "Borough"]
    num_cols = ["Latitude", "Longitude", "Hour", "DayOfWeek", "Month"]

    # 7) Build transformers
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )

    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # numeric -> passthrough (already numeric)
    numeric_transformer = "passthrough"

    # 8) ColumnTransformer: combine text + categorical + numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("cat", cat_transformer, cat_cols),
            ("num", numeric_transformer, num_cols),
        ]
    )

    # 9) Classifier (you can tune params if you like)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    # 10) Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    # 11) Train/test split for quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

    print("\nFitting pipeline…")
    pipeline.fit(X_train, y_train)

    # 12) Quick evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy on held-out test:", acc)
    print("\nClassification report (top 10 labels):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 13) Save pipeline
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print("\nSaved pipeline to:", MODEL_FILE.resolve())


if __name__ == "__main__":
    main()
