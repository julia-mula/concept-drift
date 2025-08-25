import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from river.drift import ADWIN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

project_numbers = [1, 3, 4, 5, 8, 11, 12, 13, 24, 25, 28, 29, 34, 36, 43]
INPUT_FOLDER = "data/cleaned"
RESULT_FOLDER = "results/concept_drift_retrain"
INDEX_FILE = os.path.join(RESULT_FOLDER, "results_indexes_retrained.csv")
COMPARE_FILE = os.path.join(RESULT_FOLDER, "drift_metrics_retrained.csv")
INITIAL_TRAIN_SIZE = 100
DELTA = 0.01
OLD_MODEL_WEIGHT = 0.3
BEFORE_WINDOW = 20
AFTER_WINDOW = 20

os.makedirs(RESULT_FOLDER, exist_ok=True)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

with open(INDEX_FILE, "w") as f:
    f.write("Project,DriftIndexes\n")

with open(COMPARE_FILE, "w") as f:
    f.write("Project,DriftIndex,MeanErrorBefore,MeanErrorAfterRetrain\n")

for project in project_numbers:
    filename = os.path.join(INPUT_FOLDER, f"issues_{project}.csv")
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    print(f"Processing Project {project}")
    df = pd.read_csv(filename, on_bad_lines="skip")

    if df.empty or len(df) <= INITIAL_TRAIN_SIZE:
        print(f"Skipping Project {project} - not enough data.")
        continue

    try:
        sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
        embeddings = sentence_model.encode(sentences, show_progress_bar=False)

        type_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        priority_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        type_encoded = type_ohe.fit_transform(df[["Type"]])
        priority_encoded = priority_ohe.fit_transform(df[["Priority"]])

        X_all = np.hstack((embeddings, type_encoded, priority_encoded))
        y_all = np.log1p(df["Result"].astype(float))

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        adwin = ADWIN(delta=DELTA)
        errors = []
        drift_points = []
        old_models = []

        X_train, y_train = X_all[:INITIAL_TRAIN_SIZE], y_all[:INITIAL_TRAIN_SIZE]
        model.fit(X_train, y_train)

        for i in range(INITIAL_TRAIN_SIZE, len(X_all)):
            x_i = X_all[i].reshape(1, -1)
            y_true = y_all[i]

            y_pred_new = model.predict(x_i)[0]

            if old_models:
                y_pred_old_avg = np.mean([m.predict(x_i)[0] for m in old_models])
                y_pred = OLD_MODEL_WEIGHT * y_pred_old_avg + (1 - OLD_MODEL_WEIGHT) * y_pred_new
            else:
                y_pred = y_pred_new

            error = abs(np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true)
            errors.append(error)
            adwin.update(error)

            if adwin.drift_detected:
                drift_idx = i - INITIAL_TRAIN_SIZE
                drift_points.append(drift_idx)

                before_errors = errors[-BEFORE_WINDOW:] if len(errors) >= BEFORE_WINDOW else errors
                mean_before = np.mean(before_errors) if before_errors else np.nan

                snapshot = clone(model)
                snapshot.fit(X_train, y_train)
                old_models.append(snapshot)

                train_start = max(0, i - INITIAL_TRAIN_SIZE)
                X_train, y_train = X_all[train_start:i], y_all[train_start:i]
                model.fit(X_train, y_train)

                after_errors = []
                for j in range(i + 1, min(i + 1 + AFTER_WINDOW, len(X_all))):
                    x_j = X_all[j].reshape(1, -1)
                    y_j_true = y_all[j]
                    y_j_pred_new = model.predict(x_j)[0]

                    if old_models:
                        y_j_pred_old_avg = np.mean([m.predict(x_j)[0] for m in old_models])
                        y_j_pred = OLD_MODEL_WEIGHT * y_j_pred_old_avg + (1 - OLD_MODEL_WEIGHT) * y_j_pred_new
                    else:
                        y_j_pred = y_j_pred_new

                    err_j = abs(np.expm1(y_j_true) - np.expm1(y_j_pred)) / np.expm1(y_j_true)
                    after_errors.append(err_j)

                mean_after = np.mean(after_errors) if after_errors else np.nan

                with open(COMPARE_FILE, "a") as f:
                    f.write(f"{project},{drift_idx},{mean_before:.4f},{mean_after:.4f}\n")

        plt.figure(figsize=(14, 6))
        plt.plot(errors, label="Relative Error (MMRE)", linewidth=1)
        for idx, dp in enumerate(drift_points):
            plt.axvline(x=dp, color='red', linestyle='--', label='Drift Detected' if idx == 0 else "")
        plt.xlabel("Sample Index (relative to test start)")
        plt.ylabel("Relative Error")
        plt.title(f"Project {project} - ADWIN Concept Drift (Retrained)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(RESULT_FOLDER, f"issues_{project}_retrained.png")
        plt.savefig(plot_path)
        plt.close()

        with open(INDEX_FILE, "a") as f:
            drift_str = ";".join(map(str, drift_points))
            f.write(f"{project},{drift_str}\n")

    except Exception as e:
        print(f"Error processing project {project}: {e}")
