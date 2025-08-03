import os
import torch

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(device)

# Load the MNIST dataset
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# 判断训练集是否存在如果存在就不再重复下载
# Load the MNIST dataset
data_path = './data'

# Always define trainset and testset
trainset = datasets.MNIST(data_path, download=not os.path.exists(data_path), train=True, transform=transform)
testset = datasets.MNIST(data_path, download=not os.path.exists(data_path), train=False, transform=transform)

# Create DataLoaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print('MNIST dataset loaded')
print(f'Train dataset shape: {trainset.data.shape}')
print(f"Test dataset shape: {testset.data.shape}")
