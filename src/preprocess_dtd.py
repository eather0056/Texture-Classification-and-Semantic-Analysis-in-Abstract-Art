import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def preprocess_dtd_dataset(data_dir, batch_size=32, train_split=0.8):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the dataset
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, "images"), transform=transform)
    
    # Split into train and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Paths
data_dir = "data/raw/dtd"
train_loader, val_loader = preprocess_dtd_dataset(data_dir)

print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
