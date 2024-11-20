import os
from PIL import Image

def preprocess_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size)
                output_path = os.path.join(output_folder, file)
                img.save(output_path)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")

preprocess_images("data/raw/wikiart/", "data/processed/")
