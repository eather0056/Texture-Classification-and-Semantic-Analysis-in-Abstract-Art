import json
import pandas as pd

def preprocess_feelingblue_annotations(input_json, output_csv):
    # Load JSON file
    with open(input_json, "r") as f:
        data = json.load(f)

    # Initialize a list to store processed records
    records = []

    # Iterate through each file in the JSON
    for filename, file_data in data.items():
        if "emotion" in file_data:
            # Extract emotions
            for emotion, emotion_data in file_data["emotion"].items():
                # Process MAX and MIN rationales
                for category, annotations in emotion_data.items():  # category: MAX or MIN
                    for anno_id, annotation in annotations.items():
                        # Extract relevant fields
                        rationale = annotation.get("rationale", "No rationale provided")
                        polarity = annotation.get("polarity", [])
                        subjectivity = annotation.get("subjectivity", [])
                        annotator = annotation.get("annotator", "Unknown")

                        # Convert polarity and subjectivity to strings for CSV compatibility
                        polarity_str = ", ".join(map(str, polarity)) if polarity else "None"
                        subjectivity_str = ", ".join(map(str, subjectivity)) if subjectivity else "None"

                        # Store the record
                        records.append({
                            "filename": filename,
                            "emotion": emotion,
                            "category": category,  # MAX or MIN
                            "rationale": rationale,
                            "polarity": polarity_str,
                            "subjectivity": subjectivity_str,
                            "annotator": annotator
                        })

    # Convert records to a DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed annotations saved to {output_csv}")

# Set paths
input_json = "data/annotations/feelingbluelemas.json"
output_csv = "data/processed/feelingblue_annotations.csv"

# Run the preprocessing function
preprocess_feelingblue_annotations(input_json, output_csv)
