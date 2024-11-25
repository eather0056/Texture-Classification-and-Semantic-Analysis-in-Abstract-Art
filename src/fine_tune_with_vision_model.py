from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from preprocess_dtd import preprocess_dtd_dataset  # Import your preprocessing function

def fine_tune_clip_with_vision_model(train_loader, val_loader, epochs=5, learning_rate=1e-4):
    # Load the fine-tuned vision encoder
    vision_model = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_dtd")

    # Load the original CLIP model and substitute the vision encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.vision_model.load_state_dict(vision_model.state_dict())

    # Replace the classifier layer with a new one
    num_classes = len(train_loader.dataset.dataset.classes)
    clip_model.visual_projection = nn.Linear(clip_model.visual_projection.in_features, num_classes)
    
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
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            # Forward pass
            outputs = clip_model.get_image_features(**inputs)
            logits = outputs.logits_per_image
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
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                outputs = clip_model.get_image_features(**inputs)
                logits = outputs.logits_per_image
                loss = criterion(logits, labels)

                val_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader)}, Accuracy = {correct / len(val_loader.dataset):.4f}")

# Load data
data_dir = "data/raw/dtd"
train_loader, val_loader = preprocess_dtd_dataset(data_dir)

# Train the model
fine_tune_clip_with_vision_model(train_loader, val_loader, epochs=5, learning_rate=1e-4)