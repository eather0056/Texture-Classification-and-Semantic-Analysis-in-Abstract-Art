import pandas as pd
import os
import shutil

# Paths
annotations_csv = "data/processed/texture_categorized_annotations.csv"
images_folder = "data/processed/normalized_images"
output_folder = "data/processed/texture_categories"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(annotations_csv)

# Iterate over each row in the CSV
for _, row in df.iterrows():
    filename = row["filename"]
    texture_categories = row["texture_category"].split(", ")  # Split categories into a list

    # Check if the image exists in the source folder
    src_path = os.path.join(images_folder, filename)
    if not os.path.exists(src_path):
        print(f"Image not found: {src_path}")
        continue

    # Copy the image to each category folder
    for category in texture_categories:
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)  # Create the category folder if it doesn't exist

        dest_path = os.path.join(category_folder, filename)
        shutil.copy(src_path, dest_path)  # Copy the image
        print(f"Copied {filename} to {category_folder}")

print("Images organized by texture categories.")
