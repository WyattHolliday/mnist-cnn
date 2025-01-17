from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import os

train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=False, num_workers=1)
}

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # conv output width = square image width - kernel size + 1 (/ 2 for pooling)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1x28x28 -> 10x12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10x12x12 -> 20x4x4 = 320
        self.conv2_drop = nn.Dropout2d() # randomly zeroes some elements during training
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # flatten data for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

device = torch.device('cpu') # use CPU for now
model = ConvolutionalNeuralNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train() # set the model to training mode
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item()}')

def test():
    model.eval() # set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)')

class CustomImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_files = [file for file in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, file))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # MNIST images are grayscale
        image = self.transform(image)
        return image, self.image_files[idx]

def classify_custom_images():
    dataset = CustomImageDataset(str(Path(__file__).parent.absolute()) + '\\custom_images')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for images, filenames in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for filename, prediction in zip(filenames, predicted):
                print(f'Image: {filename}, Predicted Label: {prediction.item()}')


if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()
    classify_custom_images()

    # model_path = 'models/mnist_cnn.pth'
    # torch.save(model.state_dict(), str(Path(__file__).parent.absolute()) + model_path)