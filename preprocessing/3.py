import pandas as pd
import os
import glob

result = "ps"  
input_folder = 'temp'
output_folder = 'cleaned'

os.makedirs(output_folder, exist_ok=True)

MAX_PROJECT_ID = 44

def process_and_route_rows(input_csv):
    try:
        df = pd.read_csv(input_csv, on_bad_lines='skip')
    except Exception as e:
        print(f"Failed to read {input_csv}: {e}")
        return

    if df.empty or 'Project_ID' not in df.columns:
        print(f"Skipping {input_csv} due to missing or empty data.")
        return

    columns_to_remove = [
        'ID', 'Jira_ID', 'Issue_Key', 'URL', 'Title_Changed_After_Estimation',
        'Status', 'Resolution', 'Last_Updated', 'Estimation_Date',
        'Timespent', 'In_Progress_Minutes', 'Resolution_Time_Minutes',
        'Description_Changed_After_Estimation', 'Story_Point_Changed_After_Estimation',
        'Pull_Request_URL', 'Creator_ID', 'Reporter_ID', 'Assignee_ID', 'Sprint_ID'
    ]
    df.drop(columns=columns_to_remove, errors='ignore', inplace=True)

    df['Creation_Date'] = pd.to_datetime(df.get('Creation_Date'), errors='coerce')
    df['Resolution_Date'] = pd.to_datetime(df.get('Resolution_Date'), errors='coerce')
    df['Total_Effort_Minutes'] = pd.to_numeric(df.get('Total_Effort_Minutes'), errors='coerce')

    for prj_id in df['Project_ID'].dropna().unique():
        prj_id = int(prj_id)
        if prj_id > MAX_PROJECT_ID:
            continue  

        df_prj = df[df['Project_ID'] == prj_id].copy()
        df_prj.drop(columns=['Project_ID'], errors='ignore', inplace=True)

        if result == "time":
            df_prj = df_prj[~((df_prj['Total_Effort_Minutes'] == 0) &
                              (df_prj['Creation_Date'].isna() | df_prj['Resolution_Date'].isna()))]
            df_prj['Result'] = df_prj.apply(
                lambda row: (
                    (row['Resolution_Date'] - row['Creation_Date']).total_seconds() / 60
                    if pd.notna(row['Creation_Date']) and pd.notna(row['Resolution_Date']) and row['Total_Effort_Minutes'] == 0
                    else row['Total_Effort_Minutes']
                ),
                axis=1
            )
        else:  # result == "ps"
            df_prj['Story_Point'] = pd.to_numeric(df_prj['Story_Point'], errors='coerce')
            df_prj = df_prj[df_prj['Story_Point'].notna()]
            df_prj = df_prj[df_prj['Story_Point'] > 0]
            df_prj = df_prj[df_prj['Story_Point'] < 20]
            df_prj['Result'] = df_prj['Story_Point']


        df_prj.drop(columns=['Resolution_Date', 'Total_Effort_Minutes'], errors='ignore', inplace=True)

        if df_prj.empty:
            continue 

        output_csv = os.path.join(output_folder, f'issues_{prj_id}.csv')

        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, df_prj], ignore_index=True).drop_duplicates()
            combined_df.to_csv(output_csv, index=False)
        else:
            df_prj.to_csv(output_csv, index=False)

        print(f"Processed project {prj_id} into {output_csv}")

all_input_files = glob.glob(os.path.join(input_folder, '*.csv'))

for file_path in all_input_files:
    print(f"Processing file: {file_path}")
    process_and_route_rows(file_path)

print("All files processed.")
