import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import csv
import os

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
            y_pred = model.predict(X_test)
            val_mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

            if val_mae < best_score:
                best_score = val_mae
                best_model = model

        y_test_pred = best_model.predict(X_test)
        y_test_true = np.expm1(y_test)
        y_test_pred_rounded = np.round(np.expm1(y_test_pred))

        results = {}
        for name, func in self.metrics.items():
            results[name] = func(y_test_true, y_test_pred_rounded)

        return best_model, results

def prepare_data_from_df(df, model, type_ohe, priority_ohe):
    df = df.drop(columns="Creation_Date", errors="ignore")

    sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
    embeddings = model.encode(sentences)

    type_encoded = type_ohe.transform(df[["Type"]])
    priority_encoded = priority_ohe.transform(df[["Priority"]])

    X = np.hstack((embeddings, type_encoded, priority_encoded))
    y = np.log1p(df["Result"].astype(float))

    return X, y

if __name__ == "__main__":
    cleaned_dir = "data/cleaned"
    res_dir = "results/sliding_window"
    os.makedirs(res_dir, exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    param_grid = [
        {"n_estimators": 100, "max_depth": None}
    ]

    for filename in os.listdir(cleaned_dir):
        if filename.startswith("issues_") and filename.endswith(".csv"):
            file_path = os.path.join(cleaned_dir, filename)
            print(f"\nProcessing {filename}...")

            
            df = pd.read_csv(file_path, on_bad_lines="skip")
            print(f"  Number of rows: {len(df)}")
'''
                if len(df) < 24:
                    print(f"  Skipping {filename} (too few rows for sliding window)")
                    continue

                type_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                priority_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                type_ohe.fit(df[["Type"]])
                priority_ohe.fit(df[["Priority"]])

                num_chunks = 12
                chunk_size = len(df) // num_chunks
                chunks = [df.iloc[i * chunk_size: (i + 1) * chunk_size].reset_index(drop=True) for i in range(num_chunks)]

                all_results = []

                for i in range(num_chunks - 1):
                    print(f"  Window {i+1}: Train on chunk {i}, Test on chunk {i+1}")

                    df_train = chunks[i]
                    df_test = chunks[i + 1]

                    X_train, y_train = prepare_data_from_df(df_train, model, type_ohe, priority_ohe)
                    X_test, y_test = prepare_data_from_df(df_test, model, type_ohe, priority_ohe)

                    base_model = RandomForestRegressor(random_state=42)
                    validator = Validator(base_model)
                    best_model, metrics = validator.tune_and_validate(X_train, y_train, X_test, y_test, param_grid)

                    all_results.append({
                        "window": i + 1,
                        **metrics
                    })

                output_number = filename.split("_p")[-1].replace(".csv", "")
                output_filename = f"p{output_number}.csv"
                output_path = os.path.join(res_dir, output_filename)

                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["window", "MAE", "MSE", "R2", "MMRE"])
                    writer.writeheader()
                    writer.writerows(all_results)

                print(f"Results saved to {output_path}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
'''
