import pandas as pd
import re  # For parsing RGB strings

def preprocess_annotations(input_csv, output_csv):
    # Load annotations
    df = pd.read_csv(input_csv)

    # Function to parse RGB strings
    def parse_rgb(rgb_string):
        try:
            # Extract numbers using regex
            numbers = re.findall(r'\d+', rgb_string)
            return [int(c) / 255.0 for c in numbers]
        except Exception as e:
            print(f"Error parsing RGB: {rgb_string}, {e}")
            return [0, 0, 0]  # Fallback value

    # Apply parsing to all RGB columns
    for column in ['RGB1', 'RGB2', 'RGB3', 'RGB4', 'RGB5', 'RGB6']:
        df[column] = df[column].apply(parse_rgb)
    
    # Save processed annotations
    df.to_csv(output_csv, index=False)
    print("Color annotations preprocessed and saved.")

# Set paths
input_csv = "data/annotations/wikiartabstractcolors.csv"
output_csv = "data/processed/cleaned_color_annotations.csv"
preprocess_annotations(input_csv, output_csv)
