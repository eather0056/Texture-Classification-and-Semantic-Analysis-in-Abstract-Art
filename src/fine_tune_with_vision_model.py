from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from collections import OrderedDict
from preprocess_dtd import preprocess_dtd_dataset  # Import your preprocessing function

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
    vision_model = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_dtd")

    # Load the original CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load the fine-tuned vision encoder weights into the original CLIP model
    load_fine_tuned_weights(clip_model, vision_model)

    # Add a custom classification head
    num_classes = len(train_loader.dataset.dataset.classes)
    clip_model.classification_head = nn.Linear(512, num_classes)  # Correct input size for get_image_features()

    # Define the processor for input handling
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(clip_model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

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

        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader)}, Accuracy = {correct / len(val_loader.dataset):.4f}")

# Load the DTD dataset
data_dir = "data/raw/dtd"
train_loader, val_loader = preprocess_dtd_dataset(data_dir)

# Train the model
fine_tune_clip_with_vision_model(train_loader, val_loader, epochs=5, learning_rate=1e-4)
