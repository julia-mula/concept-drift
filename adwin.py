import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from river.drift import ADWIN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

project_numbers = [1, 3, 4, 5, 8, 11, 12, 13, 24, 25, 28, 29, 34, 36, 43]

INPUT_FOLDER = "data/cleaned"
RESULT_FOLDER = "results/concept_drift"
INDEX_FILE = os.path.join(RESULT_FOLDER, "results_indexes.csv")
INITIAL_TRAIN_SIZE = 100
DELTA = 0.01

os.makedirs(RESULT_FOLDER, exist_ok=True)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

with open(INDEX_FILE, "w") as f:
    f.write("Project,DriftIndexes\n")

for project in project_numbers:
    filename = os.path.join(INPUT_FOLDER, f"issues_{project}.csv")
    if not os.path.exists(filename):
        print(f"[!] Missing file: {filename}")
        continue

    print(f"[+] Processing Project {project}")
    df = pd.read_csv(filename, on_bad_lines="skip")

    if df.empty or len(df) <= INITIAL_TRAIN_SIZE:
        print(f"[!] Skipping Project {project} - not enough data.")
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

        X_train, y_train = X_all[:INITIAL_TRAIN_SIZE], y_all[:INITIAL_TRAIN_SIZE]
        model.fit(X_train, y_train)

        for i in range(INITIAL_TRAIN_SIZE, len(X_all)):
            x_i = X_all[i].reshape(1, -1)
            y_true = y_all[i]
            y_pred = model.predict(x_i)[0]

            error = abs(np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true)
            errors.append(error)
            adwin.update(error)

            if adwin.drift_detected:
                drift_points.append(i - INITIAL_TRAIN_SIZE)
                train_start = max(0, i - INITIAL_TRAIN_SIZE)
                model.fit(X_all[train_start:i], y_all[train_start:i])

        plt.figure(figsize=(14, 6))
        plt.plot(errors, label="Relative Error (MMRE)", linewidth=1)
        for idx, dp in enumerate(drift_points):
            plt.axvline(x=dp, color='red', linestyle='--', label='Drift Detected' if idx == 0 else "")
        plt.xlabel("Sample Index (relative to test start)")
        plt.ylabel("Relative Error")
        plt.title(f"Project {project} - ADWIN Concept Drift")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(RESULT_FOLDER, f"issues_{project}.png")
        plt.savefig(plot_path)
        plt.close()

        with open(INDEX_FILE, "a") as f:
            drift_str = ";".join(map(str, drift_points))
            f.write(f"{project},{drift_str}\n")

    except Exception as e:
        print(f"Error processing project {project}: {e}")
