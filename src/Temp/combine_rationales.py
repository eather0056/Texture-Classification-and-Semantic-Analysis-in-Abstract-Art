import pandas as pd

def combine_rationales(input_csv, output_csv):
    # Load the dataset
    df = pd.read_csv(input_csv)
    
    # Combine rationales for the same image, emotion, and category
    combined_df = (
        df.groupby(["filename", "emotion", "category"])["rationale"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))  # Concatenate rationales
        .reset_index()
    )
    
    # Merge back with other columns to keep the dataset complete
    other_columns = df[["filename", "emotion", "category"]].drop_duplicates()
    result_df = pd.merge(other_columns, combined_df, on=["filename", "emotion", "category"])
    
    # Save the result to a new CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Rationales combined and saved to {output_csv}")

# File paths
input_csv = "data/processed/feelingblue_annotations.csv"
output_csv = "data/processed/combined_feelingblue_annotations.csv"

# Run the function
combine_rationales(input_csv, output_csv)
