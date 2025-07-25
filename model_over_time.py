import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import csv

def fit_global_encoders(filepath):
    df = pd.read_csv(filepath, on_bad_lines="skip")
    type_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    priority_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_ohe.fit(df[["Type"]])
    priority_ohe.fit(df[["Priority"]])
    return type_ohe, priority_ohe, df

def prepare_data_from_df(df, type_ohe, priority_ohe):
    df.drop(columns="Creation_Date", errors='ignore', inplace=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
    embeddings = model.encode(sentences)

    type_encoded = type_ohe.transform(df[["Type"]])
    priority_encoded = priority_ohe.transform(df[["Priority"]])

    X = np.hstack((embeddings, type_encoded, priority_encoded))
    y = np.log1p(df["Result"].astype(float))

    return X, y

def prepare_oldest_data(df, percent, type_ohe, priority_ohe):
    n_rows = int(len(df) * percent / 100.0)
    subset = df.iloc[:n_rows].reset_index(drop=True)
    return prepare_data_from_df(subset, type_ohe, priority_ohe)

def prepare_latest_data(df, percent, type_ohe, priority_ohe):
    n_rows = int(len(df) * percent / 100.0)
    subset = df.iloc[-n_rows:].reset_index(drop=True)
    return prepare_data_from_df(subset, type_ohe, priority_ohe)

def mmre(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

class Validator:
    def __init__(self, model, metrics=None):
        self.model = model
        self.metrics = metrics or {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
            "MMRE": mmre
        }

    def tune_and_validate(self, X_train, y_train, X_test, y_test, param_grid):
        y_train_true = np.expm1(y_train)
        sample_weight = y_train_true / y_train_true.max()

        best_score = float('inf')
        best_model = None

        for params in param_grid:
            model = self.model.set_params(**params)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            y_val_pred = model.predict(X_test)
            val_mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_val_pred))

            if val_mae < best_score:
                best_score = val_mae
                best_model = model

        y_test_pred = best_model.predict(X_test)
        y_test_true = np.expm1(y_test)
        y_test_pred_rounded = np.round(np.expm1(y_test_pred))

        results = {}
        for name, func in self.metrics.items():
            score = func(y_test_true, y_test_pred_rounded)
            results[name] = score

        return best_model, results

if __name__ == "__main__":
    input_folder = "data/cleaned"
    output_folder = "results/results_over_time"
    os.makedirs(output_folder, exist_ok=True)

    files_to_process = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".csv") and f.startswith("issues_")
    ])

    percentages = list(range(10, 85, 5)) 
    param_grid = [
        {"n_estimators": 100, "max_depth": None}
    ]

    for idx, filename in enumerate(sorted(files_to_process)):
        if filename.endswith(".csv"):
            print(f"\n🔎 Processing {filename} ...")

            filepath = os.path.join(input_folder, filename)
            type_ohe, priority_ohe, full_df = fit_global_encoders(filepath)

            test_X, test_y = prepare_latest_data(full_df, percent=20, type_ohe=type_ohe, priority_ohe=priority_ohe)


            all_results = []

            for percent in percentages:
                print(f"\n--- Training on {percent}% oldest data ---")
                train_X, train_y = prepare_oldest_data(full_df, percent, type_ohe=type_ohe, priority_ohe=priority_ohe)

                base_model = RandomForestRegressor(random_state=42)
                validator = Validator(base_model)
                best_model, metrics = validator.tune_and_validate(train_X, train_y, test_X, test_y, param_grid)

                print("--- Metrics ---")
                for name, val in metrics.items():
                    print(f"{name}: {val:.4f}")

                all_results.append({
                    "percent": percent,
                    **metrics
                })

            result_path = os.path.join(output_folder, f"issues_{idx}.csv")
            with open(result_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["percent", "MAE", "MSE", "R2", "MMRE"])
                writer.writeheader()
                writer.writerows(all_results)

            print(f"Results saved to {result_path}")