import os
import random

# Paths
images_dir = "data/raw/wikiart/images"
output_dir = "data/raw/wikiart/labels"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Lists to store file paths for each split
train_files = []
val_files = []
test_files = []

# Iterate through each category folder
for category in os.listdir(images_dir):
    category_path = os.path.join(images_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    # Get all image files in the category folder
    images = [f"{category}/{img}" for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    # Shuffle the images for randomness
    random.shuffle(images)
    
    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Assign images to splits
    train_files.extend(images[:train_end])
    val_files.extend(images[train_end:val_end])
    test_files.extend(images[val_end:])

# Write to text files
with open(os.path.join(output_dir, "train.txt"), "w") as train_file:
    train_file.write("\n".join(train_files))
with open(os.path.join(output_dir, "val.txt"), "w") as val_file:
    val_file.write("\n".join(val_files))
with open(os.path.join(output_dir, "test.txt"), "w") as test_file:
    test_file.write("\n".join(test_files))

print("Train, validation, and test text files created successfully.")
