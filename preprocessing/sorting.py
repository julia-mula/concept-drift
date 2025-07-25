import pandas as pd
import os
import glob

cleaned_folder = 'cleaned'
all_cleaned_files = glob.glob(os.path.join(cleaned_folder, 'issues_*.csv'))

for file_path in all_cleaned_files:
    print(f"Processing: {file_path}")
    
    df = pd.read_csv(file_path)

    if 'Creation_Date' not in df.columns:
        print(f"Skipped (no Creation_Date column): {file_path}")
        continue

    df['Creation_Date'] = pd.to_datetime(df['Creation_Date'], errors='coerce')
    df = df.dropna(subset=['Creation_Date'])

    df = df.sort_values(by='Creation_Date', ascending=True)

    df.to_csv(file_path, index=False)

print(" All files cleaned and sorted.")
