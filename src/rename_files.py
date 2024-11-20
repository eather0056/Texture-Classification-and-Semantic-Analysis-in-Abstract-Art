import os

def standardize_filenames(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "/" in file:
                new_name = file.replace("/", "_")
                os.rename(os.path.join(root, file), os.path.join(root, new_name))
                print(f"Renamed {file} to {new_name}")
            else:
                print(f"No renaming needed for {file}")

standardize_filenames("data/raw/wikiart/")
