import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) #flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 3. Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss() # Loss function for classifciation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")

# 5. Testing the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

import matplotlib.pyplot as plt

# Visualize some test images and predictions
def visualize_predictions():
    dataiter = iter(testloader)
    images, labels = next(iter(dataiter))
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Plot images with their predicted labels
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].numpy().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predicted[i].item()}")
        ax.axis('off')
    #plt.show()
    plt.savefig('visualized_predictions.png', bbox_inches='tight')

    plt.show()

visualize_predictions()