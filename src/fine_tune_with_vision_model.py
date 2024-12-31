import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from PIL import Image
import numpy as np

# Function to load fine-tuned weights into the vision encoder
def load_fine_tuned_weights(clip_model, fine_tuned_vision_model):
    fine_tuned_state_dict = fine_tuned_vision_model.state_dict()
    clip_state_dict = clip_model.vision_model.state_dict()

    # Create a new state dictionary for the vision model
    new_state_dict = OrderedDict()

    for key, value in fine_tuned_state_dict.items():
        if key in clip_state_dict and clip_state_dict[key].shape == value.shape:
            new_state_dict[key] = value
        else:
            print(f"Skipping {key} due to mismatch in shape or absence in original model.")

    # Load the updated state dictionary
    clip_model.vision_model.load_state_dict(new_state_dict, strict=False)

# Fine-tuning function
def fine_tune_clip_with_vision_model(train_loader, val_loader, epochs=5, learning_rate=1e-4):
    # Load the fine-tuned vision encoder
    vision_model = CLIPModel.from_pretrained("tanganke/clip-vit-base-patch32_dtd")

    # Load the original CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load the fine-tuned vision encoder weights into the original CLIP model
    load_fine_tuned_weights(clip_model, vision_model)

    # Add a custom classification head
    num_classes = len(set(train_loader.dataset.labels))
    clip_model.classification_head = nn.Linear(512, num_classes)  # Adjust to your needs

    # Define the processor for input handling
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Calculate class weights for weighted loss
    class_counts = torch.bincount(torch.tensor(train_loader.dataset.labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    # Define optimizer and learning rate scheduler
    optimizer = Adam(clip_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    # Ensure the models directory exists
    os.makedirs("./models/wikiart/checkpoints", exist_ok=True)

    # Initialize lists for tracking loss and accuracy
    train_losses, val_losses, val_accuracies = [], [], []

    # Training loop
    for epoch in range(epochs):
        clip_model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Preprocess images for CLIP
            inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)

            # Forward pass
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            logits = clip_model.classification_head(image_features)  # Use custom classification head
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        print(f"Epoch {epoch + 1}: Training Loss = {train_loss / len(train_loader)}")

        # Validation step
        clip_model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)
                image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                logits = clip_model.classification_head(image_features)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / len(val_loader.dataset))
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader)}, Accuracy = {correct / len(val_loader.dataset):.4f}")

        # Save a checkpoint after each epoch
        checkpoint_path = f"./models/wikiart/checkpoints/epoch_{epoch + 1}_clip_model.pth"
        torch.save(clip_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Step the learning rate scheduler
        scheduler.step()

    # Save the final fine-tuned model and processor
    torch.save(clip_model.state_dict(), "./models/wikiart/fine_tuned_clip_model.pth")
    print("Final model weights saved as fine_tuned_clip_model.pth")

    torch.save(clip_model, "./models/wikiart/fine_tuned_clip_model_complete.pth")
    print("Final complete model saved as fine_tuned_clip_model_complete.pth")

    processor.save_pretrained("./models/wikiart/fine_tuned_processor")
    print("Processor configuration saved in fine_tuned_processor/")

    # Plot training analysis
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training and Validation Analysis")
    plt.legend()
    plt.grid()
    plt.savefig("training_analysis.png")
    print("Training analysis plots saved as training_analysis.png")

# Evaluation function
def evaluate_model(clip_model, processor, test_loader, class_names):
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

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion Matrix saved as confusion_matrix.png")

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

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_dir = "data/raw/wikiart"
train_image_paths, train_labels = load_dataset_from_labels(data_dir, os.path.join(data_dir, "labels/train.txt"), transform)
val_image_paths, val_labels = load_dataset_from_labels(data_dir, os.path.join(data_dir, "labels/val.txt"), transform)
test_image_paths, test_labels = load_dataset_from_labels(data_dir, os.path.join(data_dir, "labels/test.txt"), transform)

train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform)
val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform)
test_dataset = CustomImageDataset(test_image_paths, test_labels, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model
fine_tune_clip_with_vision_model(train_loader, val_loader, epochs=5, learning_rate=1e-4)

# Load the model for evaluation
clip_model = torch.load("./models/wikiart/fine_tuned_clip_model_complete.pth")
processor = CLIPProcessor.from_pretrained("./models/wikiart/fine_tuned_processor")
class_names = sorted(os.listdir(os.path.join(data_dir, "images")))

# Evaluate the model
evaluate_model(clip_model, processor, test_loader, class_names)
