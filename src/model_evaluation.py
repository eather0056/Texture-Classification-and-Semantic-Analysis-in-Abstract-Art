import os
import torch
from torchvision import transforms
from transformers import CLIPProcessor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import numpy as np

# Helper function to load dataset from labels file
def load_dataset_from_labels(data_dir, label_file, transform):
    image_paths, labels = [], []
    class_names = sorted(os.listdir(os.path.join(data_dir, "images")))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_name, image_name = line.split("/")
            image_path = os.path.join(data_dir, "images", class_name, image_name)
            image_paths.append(image_path)
            labels.append(class_to_idx[class_name])

    return image_paths, labels

# Custom Dataset Class
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Evaluation function with visualization
def evaluate_and_visualize_model(clip_model, processor, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            logits = clip_model.classification_head(image_features)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=1))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Accuracy per Class
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accuracies)
    plt.title("Per-Class Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.savefig("per_class_accuracy.png")
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Per-Class Accuracy
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accuracies, color='skyblue')
    plt.xlabel("Class Name")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)
    plt.savefig("per_class_accuracy.png")
    plt.show()

    # Save the Classification Report as Text
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Classification Report saved as classification_report.txt")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
data_dir = "data/raw/wikiart"
test_image_paths, test_labels = load_dataset_from_labels(data_dir, os.path.join(data_dir, "labels/test.txt"), transform)
test_dataset = CustomImageDataset(test_image_paths, test_labels, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model and processor
clip_model = torch.load("./models/wikiart/fine_tuned_clip_model_complete.pth")
processor = CLIPProcessor.from_pretrained("./models/wikiart/fine_tuned_processor")
class_names = sorted(os.listdir(os.path.join(data_dir, "images")))

# Evaluate and visualize
evaluate_and_visualize_model(clip_model, processor, test_loader, class_names)
