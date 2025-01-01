import os
import pandas as pd

# Directory containing the image folders
image_dir = "data/raw/wikiart/images"

# List of known texture categories (folder names)
texture_categories = ["chaotic", "circular", "dots", "lines", "rough", "smooth"]

# Initialize list for records
records = []

# Traverse the directory
for folder_name in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder_name)
    if os.path.isdir(folder_path):
        # Get texture category from folder name
        texture = folder_name if folder_name in texture_categories else "uncategorized"

        # Add all images in this folder to the records
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                records.append({
                    "filename": filename,
                    "texture": texture,
                    "category": folder_name
                })

# Convert records to DataFrame
texture_df = pd.DataFrame(records)

# Save to CSV
output_file = "data/processed/texture_predictions.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
texture_df.to_csv(output_file, index=False)

print(f"Texture predictions saved to {output_file}")

# Display sample data
print("Sample Data:")
print(texture_df.head())
