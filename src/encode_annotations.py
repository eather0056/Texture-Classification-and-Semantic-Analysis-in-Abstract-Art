import pandas as pd

def encode_annotations(input_csv, output_csv):
    # Load annotations
    df = pd.read_csv(input_csv)

    # Encode emotions as integers
    df['emotion_encoded'] = df['emotion'].astype('category').cat.codes

    # Save encoded annotations
    df.to_csv(output_csv, index=False)
    print(f"Encoded annotations saved to {output_csv}")

# Paths
input_csv = "data/processed/feelingblue_annotations.csv"
output_csv = "data/processed/encoded_annotations.csv"

# Encode annotations
encode_annotations(input_csv, output_csv)
