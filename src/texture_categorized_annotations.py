import pandas as pd
import re

# Load the CSV file
input_csv = "data/processed/combined_encoded_annotations.csv"
output_csv = "data/processed/texture_categorized_annotations.csv"

df = pd.read_csv(input_csv)

# Group by filename and combine all rationales into one string
df_combined = df.groupby("filename", as_index=False).agg({"rationale": " ".join})

# Define texture categories and keywords
texture_categories = {
    "circular": ["circle", "circular", "disk", "whirling", "spinning"],
    "lines": ["line", "lines", "vertical", "straight", "bar", "linear"],
    "smooth": ["smooth", "gradient", "calm", "soft"],
    "rough": ["rough", "jagged", "aggressive", "violent", "chaotic"],
    "dots": ["dot", "dotted", "point", "spots"],
    "complex": ["complex", "messy", "chaotic", "cluster", "random"],
}

# Function to classify textures based on rationale
def classify_texture(rationale):
    rationale = rationale.lower()  # Convert rationale to lowercase for matching
    matched_categories = []

    for category, keywords in texture_categories.items():
        for keyword in keywords:
            if re.search(rf"\b{keyword}\b", rationale):  # Match whole word
                matched_categories.append(category)
                break  # Avoid duplicate matches for the same category

    return ", ".join(matched_categories) if matched_categories else "uncategorized"

# Apply the classification function to the combined rationale column
df_combined["texture_category"] = df_combined["rationale"].apply(classify_texture)

# Save the results to a new CSV file
df_combined.to_csv(output_csv, index=False)

print(f"Texture categorization completed. Results saved to {output_csv}.")
