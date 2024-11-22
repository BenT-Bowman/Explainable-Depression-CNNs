import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from timm import create_model
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the DeiT model
model_name = "deit_base_distilled_patch16_224"  # You can change to other variants like deit_small, deit_tiny
model = create_model(model_name, pretrained=True, num_classes=2)
model = model.to(device)

# exit()

# 2. Prepare your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

dataset = ImageFolder(root='cwt_image\Main', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
dataset_val = ImageFolder(root='cwt_image\Validation', transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=True)
# 3. Define the training loop
criterion = torch.nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Optimizer

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm with the running loss
            tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in tqdm(validation_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Compute average loss and accuracy
    validation_loss = running_loss / len(validation_loader)
    validation_accuracy = correct_predictions / total_predictions

    print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
# Save the model
torch.save(model.state_dict(), "deit_model.pth")
