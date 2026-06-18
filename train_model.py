import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_prepare_data():
    path = os.path.join(BASE_DIR, "water_dataX.csv")
    df = pd.read_csv(path, encoding="unicode_escape")

    rename_map = {
        "Temp": "temp",
        "D.O. (mg/l)": "do",
        "PH": "ph",
        df.columns[6]: "conductivity",
        "B.O.D. (mg/l)": "bod",
        df.columns[8]: "nitrate",
        df.columns[10]: "total_coliform",
    }
    df = df.rename(columns=rename_map)

    feature_cols = ["temp", "do", "ph", "conductivity", "bod", "nitrate", "total_coliform"]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    def classify_water(row):
        if (6.5 <= row["ph"] <= 8.5) and (row["do"] >= 4) and (row["bod"] <= 3):
            return 1
        else:
            return 0

    df["target"] = df.apply(classify_water, axis=1)
    return df, feature_cols


def train_and_save_models():
    df, feature_cols = load_and_prepare_data()

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)

    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    results = {
        "Logistic Regression": {
            "model": lr_model,
            "accuracy": accuracy_score(y_test, lr_preds),
            "report": classification_report(y_test, lr_preds, output_dict=True),
            "cm": confusion_matrix(y_test, lr_preds),
            "scaled": True,
        },
        "Decision Tree": {
            "model": dt_model,
            "accuracy": accuracy_score(y_test, dt_preds),
            "report": classification_report(y_test, dt_preds, output_dict=True),
            "cm": confusion_matrix(y_test, dt_preds),
            "scaled": False,
        },
        "Random Forest": {
            "model": rf_model,
            "accuracy": accuracy_score(y_test, rf_preds),
            "report": classification_report(y_test, rf_preds, output_dict=True),
            "cm": confusion_matrix(y_test, rf_preds),
            "scaled": False,
        },
    }

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(lr_model, os.path.join(models_dir, "lr_model.pkl"))
    joblib.dump(dt_model, os.path.join(models_dir, "dt_model.pkl"))
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))

    return results, X_train, X_test, y_train, y_test, scaler, feature_cols, df


if __name__ == "__main__":
    results, _, _, _, _, _, _, _ = train_and_save_models()
    for name, info in results.items():
        print(f"{name}: Accuracy = {info['accuracy']:.4f}")
