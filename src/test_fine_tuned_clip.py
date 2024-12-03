import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load the fine-tuned model
def load_model_and_processor():
    model_path = "./models/fine_tuned_clip_model_complete.pth"
    processor_path = "./models/fine_tuned_processor"

    # Load the fine-tuned CLIP model
    clip_model = torch.load(model_path)
    clip_model.eval()  # Set to evaluation mode

    # Load the processor
    processor = CLIPProcessor.from_pretrained(processor_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    return clip_model, processor, device

# Prepare the test dataset and DataLoader
def prepare_test_loader(test_data_dir, batch_size=32):
    # Define transforms for test data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_dataset

# Perform inference on the test dataset
def evaluate_model(clip_model, processor, device, test_loader, test_dataset):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Preprocess images for CLIP
            inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)

            # Forward pass
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            logits = clip_model.classification_head(image_features)  # Use the classification head

            # Compute predictions
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy

# Test a single image
def predict_single_image(clip_model, processor, device, test_dataset, image_path):
    # Load a single image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)

    # Perform inference
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        logits = clip_model.classification_head(image_features)
        prediction = torch.argmax(logits, dim=1)

    # Map prediction to class label
    class_label = test_dataset.classes[prediction.item()]
    print(f"Predicted class: {class_label}")

    return class_label

if __name__ == "__main__":
    # Load the model and processor
    clip_model, processor, device = load_model_and_processor()

    # Prepare the test dataset
    test_data_dir = "data/raw/dtd/test"  # Update with your test dataset path
    test_loader, test_dataset = prepare_test_loader(test_data_dir)

    # Evaluate the model on the test dataset
    print("Evaluating model on the test dataset...")
    evaluate_model(clip_model, processor, device, test_loader, test_dataset)

    # Test a single image
    # test_image_path = "/home/eth/Texture-Classification-and-Semantic-Analysis-in-Abstract-Art/data/raw/wikiart/arthur-pinajian_untitled-1960.jpg"  # Update with your test image path
    # print("Testing a single image...")
    # predict_single_image(clip_model, processor, device, test_dataset, test_image_path)
