import os
import pandas as pd
import re

# Directory where the files are stored
base_dir = "Data"

# List to hold DataFrames
dataframes = []

# Loop through file numbers 1 to 22
for i in range(24, 30):
    file_name = f"MyDisplay ({i}).csv"
    file_path = os.path.join(base_dir, file_name)

    if os.path.exists(file_path):
        # Read each CSV file and ensure "Tax ID " column is treated as string
        df = pd.read_csv(file_path, dtype={"Tax ID ": str})
        dataframes.append(df)
    else:
        print(f"Warning: {file_name} not found.")

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Add a new column "cleaning" based on the "Tax ID " column
pattern = re.compile(r"^\d{11}$|^\d{5}-\d{3}-\d{3}$")
combined_df["cleaning"] = combined_df["Tax ID"].apply(lambda x: bool(pattern.match(x)) if pd.notnull(x) else False)

# Save the combined DataFrame to a new CSV file
output_path = os.path.join(base_dir, "Combined_Data2024.csv")
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path}")
