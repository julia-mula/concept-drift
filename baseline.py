import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import csv

def mmre(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

def prepare_data(df):
    df = df.drop(columns="Creation_Date", errors="ignore")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
    embeddings = embedder.encode(sentences)

    type_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    priority_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_ohe.fit(df[["Type"]])
    priority_ohe.fit(df[["Priority"]])

    type_encoded = type_ohe.transform(df[["Type"]])
    priority_encoded = priority_ohe.transform(df[["Priority"]])

    X = np.hstack((embeddings, type_encoded, priority_encoded))
    y = np.log1p(df["Result"].astype(float))

    return X, y

if __name__ == "__main__":
    input_folder = "data/cleaned"
    output_folder = "results/baseline"
    os.makedirs(output_folder, exist_ok=True)

    files_to_process = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".csv") and f.startswith("issues_")
    ])

    for idx, filename in enumerate(files_to_process):
        print(f"\nProcessing {filename} ...")
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath, on_bad_lines="skip")

        if len(df) < 10:
            print("Too few samples")
            continue

        X, y = prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

        results = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "MMRE": mmre(y_true, y_pred)
        }

        print("Evaluation Results:")
        for name, val in results.items():
            print(f"  {name}: {val:.4f}")

        result_path = os.path.join(output_folder, f"baseline_random_{idx}.csv")
        with open(result_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)