import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from river.drift import ADWIN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------- DATA PREPARATION ----------
def prepare_data_from_df(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
    embeddings = model.encode(sentences)

    type_ohe = OneHotEncoder(sparse_output=False)
    priority_ohe = OneHotEncoder(sparse_output=False)

    type_encoded = type_ohe.fit_transform(df[["Type"]])
    priority_encoded = priority_ohe.fit_transform(df[["Priority"]])

    X = np.hstack((embeddings, type_encoded, priority_encoded))
    y = np.log1p(df["Result"].astype(float))  

    return X, y, df.index


# ---------- MAIN ----------
if __name__ == "__main__":
    filepath = "final_data_project_3.csv"
    df = pd.read_csv(filepath, on_bad_lines="skip")

    df = df.iloc[::-1].reset_index(drop=True)

    X_all, y_all, idx_all = prepare_data_from_df(df)

    window_sizes = [60, 70, 100]
    deltas = [0.001, 0.05]

    results = []

    for w_size in window_sizes:
        for delta in deltas:
            adwin = ADWIN(delta=delta)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42,
            )

            start_point = w_size
            drift_points = []
            errors = []

            X_train = X_all[:w_size]
            y_train = y_all[:w_size]

            model.fit(X_train, y_train)

            for i in range(start_point, len(X_all)):
                X_test = X_all[i].reshape(1, -1)
                if i > 600:
                      y_true = np.log1p(300) 
                else:
                    y_true = y_all[i]

                y_pred = model.predict(X_test)[0]
                y_true_orig = np.expm1(y_true)
                y_pred_orig = np.expm1(y_pred)
                error = abs(y_true_orig - y_pred_orig) / y_true_orig
                errors.append(error)

                drift_detected = adwin.update(error)

        

                if drift_detected:
                    drift_points.append(i)
                    train_start = max(0, i - w_size + 1)
                    X_train = X_all[train_start:i + 1]
                    y_train = y_all[train_start:i + 1]
                    model.fit(X_train, y_train)


            results.append((w_size, delta, len(drift_points)))

    print("\nDrift detection results for parameter combinations:")
    for w_size, delta, num_drifts in results:
        print(f"Window size: {w_size}, delta: {delta:.4f} -> Detected drifts: {num_drifts}")

    plt.figure(figsize=(16, 8))
    plt.plot(errors, label="Relative Error (MMRE)", linewidth=1)
    for dp in drift_points:
        plt.axvline(x=dp - start_point, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label='Concept Drift' if dp == drift_points[0] else "")
    plt.title(f"Errors and Drift Detection (window={w_size}, delta={delta})")
    plt.xlabel("Project Index (relative)")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
