import pandas as pd

def fix_filenames(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Replace '/' with '_' in the filename column
    if "filename" in df.columns:
        df["filename"] = df["filename"].str.replace("/", "_")
    else:
        print("Error: 'filename' column not found in the CSV.")
        return
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Filenames updated and saved to {output_csv}")

# File paths
input_csv = "data/processed/feelingblue_annotations.csv"
output_csv = "data/processed/fixed_feelingblue_annotations.csv"

# Run the function
fix_filenames(input_csv, output_csv)
