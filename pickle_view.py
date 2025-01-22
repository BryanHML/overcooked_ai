# import pandas as pd

# file_path = "2019_hh_trials_all.pickle"

# # df = pd.read_pickle(file_path)

# print(df.columns)

import pandas as pd

# Path to the pickle file
file_path = "2019_hh_trials_all.pickle"

# Load the pickle file into a DataFrame
try:
    df = pd.read_pickle(file_path)
    print("Columns in the DataFrame:")
    print(df.columns)
    
    # Save the DataFrame to a CSV file
    csv_file_path = "2019_hh_trials_all.csv"  # Specify the output CSV file path
    df.to_csv(csv_file_path, index=False)  # Save without the index column
    print(f"Pickle file successfully converted to CSV and saved at: {csv_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
