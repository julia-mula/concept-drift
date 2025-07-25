import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from river.drift import PageHinkley
import matplotlib.pyplot as plt

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
    y = np.log1p(df["Result"].astype(float))  # log1p for stability

    return X, y, df.index

# ---------- MAIN ----------
if __name__ == "__main__":
    filepath = "final_data_project_3.csv"
    df = pd.read_csv(filepath, on_bad_lines="skip")


    df = df.iloc[::-1].reset_index(drop=True)

    X_all, y_all, idx_all = prepare_data_from_df(df)

    window_size = 30
    model = LGBMRegressor(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42)
    ph = PageHinkley(min_instances=30, delta=0.005, threshold=0.005, alpha=0.999)  # Tune as needed

    X_train = X_all[:window_size]
    y_train = y_all[:window_size]
    model.fit(X_train, y_train)

    start_point = window_size
    drift_points = []
    errors = []

    for i in range(start_point, len(X_all)):
        X_test = X_all[i].reshape(1, -1)
        y_true = y_all[i]

        y_pred = model.predict(X_test)[0]
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)

        error = abs(y_true_orig - y_pred_orig) / y_true_orig
        errors.append(error)

        ph.update(error)
        if ph.update(error):  
            print(f"Drift detected at index {i} (CSV row: {idx_all[i]})")
            drift_points.append(i)

            train_start = max(0, i - window_size + 1)
            X_train = X_all[train_start:i+1]
            y_train = y_all[train_start:i+1]
            model.fit(X_train, y_train)

            ph.reset()


    # ---------- PLOT ----------
    plt.figure(figsize=(16, 8))
    plt.plot(errors, label="Relative Error (MMRE)", linewidth=1)
    for i in drift_points:
        plt.axvline(x=i - start_point, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Concept Drift' if i == drift_points[0] else "")
    plt.title("Prediction Error Over Time with Page-Hinkley Drift Detection")
    plt.xlabel("Project Index (relative)")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
