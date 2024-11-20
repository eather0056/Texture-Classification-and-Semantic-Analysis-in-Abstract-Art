from PIL import Image
import os
import numpy as np

def normalize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            try:
                # Open image
                img = Image.open(input_path).convert("RGB")
                # Resize image
                img = img.resize(size)
                # Normalize pixel values to [0, 1]
                img_array = np.array(img) / 255.0
                # Save normalized image
                output_path = os.path.join(output_folder, file)
                Image.fromarray((img_array * 255).astype('uint8')).save(output_path)
                print(f"Normalized: {file}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")

# Paths
input_folder = "data/raw/wikiart"
output_folder = "data/processed/normalized_images"

# Normalize images
normalize_images(input_folder, output_folder)