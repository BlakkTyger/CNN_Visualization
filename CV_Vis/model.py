import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Check for CUDA availability (optional)
if torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cpu'
    print("No GPU available, training on CPU")

# Define the model architecture (Convolutional Neural Network)
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 4, 3)  # Input channels, output channels, kernel size
    self.pool = nn.MaxPool2d(2, 2)  # Kernel size, stride
    self.conv2 = nn.Conv2d(4, 4, 3)
    self.fc1 = nn.Linear(4 * 5 * 5, 120)  # Input features, output features
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)  # Output layer for 10 digits

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x))) 
    # print(x.shape) # Apply ReLU activation
    x = self.pool(F.relu(self.conv2(x)))
    # print(x.shape)
    x = x.view(-1, 4 * 5 * 5)  # Flatten for fully-connected layers
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Define data transforms (normalization)
transformtr = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomCrop(size=28, padding=4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),  # Convert to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize based on MNIST statistics

])
transform = transforms.Compose([
    
    transforms.ToTensor(),  # Convert to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize based on MNIST statistics

])

# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transformtr)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create data loaders and move them to GPU
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model, loss function, and optimizer
model = CNN()
model.to(device)  # Already sets the model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):  # You can adjust the number of epochs for better accuracy
  correct = 0
  total = 0
  for i, (images, labels) in enumerate(train_loader):
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy within the loop
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  # Print epoch information including accuracy
  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format(
      epoch + 1, 10, i + 1, len(train_loader), loss.item(), 100 * correct / total))

# Test the model on test data (unchanged)
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print('Accuracy of the network on the 10000 test images: {} %'.format(
      100 * correct / total
  ))
# Save the trained model (optional)
torch.save(model.state_dict(), 'mnist.pth')
