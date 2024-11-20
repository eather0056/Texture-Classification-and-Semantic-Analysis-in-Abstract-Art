import json
import os
import pandas as pd

def verify_files(json_path, csv_path, image_folder):
    # Verify JSON
    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return
    
    # Handle different JSON structures
    if isinstance(data, list):
        json_images = set(item["image"] for item in data)
    elif isinstance(data, dict):
        # Replace 'images' with the correct key in your JSON if it’s a dictionary
        json_images = set(data.get("images", []))
    else:
        print("Unexpected JSON structure.")
        return

    # Verify CSV
    csv_data = pd.read_csv(csv_path)
    if "filename" in csv_data.columns:
        csv_images = set(csv_data["filename"].tolist())
    else:
        raise ValueError("The CSV file does not contain a 'filename' column.")

    # Verify Files
    missing = []
    found = []
    for img in json_images.union(csv_images):
        img_path = os.path.join(image_folder, img)
        if not os.path.exists(img_path):
            missing.append(img)
        else:
            found.append(img)

    # Print summary
    print(f"Total files in JSON and CSV: {len(json_images.union(csv_images))}")
    print(f"Found files: {len(found)}")
    print(f"Missing files: {len(missing)}")

    # Skip missing files and continue processing
    if missing:
        print(f"Skipping {len(missing)} missing files.")
        for img in missing:
            print(f" - {img}")

# Paths to the files and directories
json_file_path = "data/annotations/feelingbluelemas.json"
csv_file_path = "data/annotations/wikiartabstractcolors.csv"
image_folder_path = "data/raw/wikiart/"

# Run the verification
verify_files(json_file_path, csv_file_path, image_folder_path)
