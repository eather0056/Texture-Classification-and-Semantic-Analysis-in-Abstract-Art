from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from preprocess_dtd import preprocess_dtd_dataset  # Import the preprocessing function

def fine_tune_clip_dtd(train_loader, val_loader, epochs=5, learning_rate=1e-4):
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("tanganke/clip-vit-base-patch32_dtd")
    processor = CLIPProcessor.from_pretrained("tanganke/clip-vit-base-patch32_dtd")
    
    # Replace the classifier layer with a new one
    num_classes = len(train_loader.dataset.dataset.classes)
    model.visual_projection = nn.Linear(model.visual_projection.in_features, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Preprocess images for CLIP
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch {epoch + 1}: Training Loss = {train_loss / len(train_loader)}")
        
        # Validation step
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
        
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader)}, Accuracy = {correct / len(val_loader.dataset):.4f}")

# Load data
data_dir = "data/raw/dtd"
train_loader, val_loader = preprocess_dtd_dataset(data_dir)

# Train the model
fine_tune_clip_dtd(train_loader, val_loader, epochs=5, learning_rate=1e-4)
