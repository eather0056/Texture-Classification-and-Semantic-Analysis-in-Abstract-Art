import os

# Define the path
checkpoint_dir = "./models/checkpoints"

# Check if the directory exists
if not os.path.exists(checkpoint_dir):
    print(f"Path does not exist: {checkpoint_dir}. Creating the directory.")
    os.makedirs(checkpoint_dir, exist_ok=True)
else:
    print(f"Path already exists: {checkpoint_dir}.")
