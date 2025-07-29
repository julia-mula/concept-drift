import os
import pandas as pd
import matplotlib.pyplot as plt

input_folder = "results/sliding_window"
output_folder = "results/sliding_window/plots"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.startswith("p_") and filename.endswith(".csv"):
        number = filename.split("p_")[1].split(".csv")[0]
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        required_columns = {"window", "MAE", "MSE", "R2", "MMRE"}
        if not required_columns.issubset(df.columns):
            print(f"Skipping {filename} — missing columns")
            continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(df["window"], df["MAE"], marker='o', color='blue')
        axs[0, 0].set_title("MAE vs Sliding Window")
        axs[0, 0].set_xlabel("Sliding Window")
        axs[0, 0].set_ylabel("MAE")
        axs[0, 0].grid(True)

        axs[0, 1].plot(df["window"], df["MSE"], marker='o', color='green')
        axs[0, 1].set_title("MSE vs Sliding Window")
        axs[0, 1].set_xlabel("Sliding Window")
        axs[0, 1].set_ylabel("MSE")
        axs[0, 1].grid(True)

        axs[1, 0].plot(df["window"], df["R2"], marker='o', color='purple')
        axs[1, 0].set_title("R² vs Sliding Window")
        axs[1, 0].set_xlabel("Sliding Window")
        axs[1, 0].set_ylabel("R² Score")
        axs[1, 0].grid(True)

        axs[1, 1].plot(df["window"], df["MMRE"], marker='o', color='red')
        axs[1, 1].set_title("MMRE vs Sliding Window")
        axs[1, 1].set_xlabel("Sliding Window")
        axs[1, 1].set_ylabel("MMRE")
        axs[1, 1].grid(True)

        plt.tight_layout()

        output_path = os.path.join(output_folder, f"plot{number}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved plot to {output_path}")
