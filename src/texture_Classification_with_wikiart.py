import os
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
from torchvision import transforms

# Define the path to the trained model and processor
model_path = "./models/fine_tuned_clip_model_complete.pth"
processor_path = "./models/fine_tuned_processor"

# Define the directory with test images
test_image_dir = "data/raw/dtd/test/wikiart"

# Load the fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

# Load the processor
processor = CLIPProcessor.from_pretrained(processor_path)

# Define a function to classify textures
def classify_texture(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True, do_rescale=False).to(device)
        
        # Get predictions
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            logits = model.classification_head(image_features)  # Classification head predicts texture classes
            predicted_class = logits.argmax(dim=1).item()
        
        return predicted_class
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Load the class labels (e.g., "striped," "bumpy," etc.)
class_labels = os.listdir("data/raw/dtd/images")  # Assuming class folder names represent labels

# Classify all test images
results = []
for root, _, files in os.walk(test_image_dir):
    for file in files:
        image_path = os.path.join(root, file)
        predicted_class_index = classify_texture(image_path)
        if predicted_class_index is not None:
            predicted_label = class_labels[predicted_class_index]
            results.append((image_path, predicted_label))
            print(f"Image: {image_path}, Predicted Texture: {predicted_label}")

# Save the results to a file
output_file = "texture_classification_results.txt"
with open(output_file, "w") as f:
    for image_path, label in results:
        f.write(f"{image_path}\t{label}\n")

print(f"Texture classification completed. Results saved to {output_file}.")
