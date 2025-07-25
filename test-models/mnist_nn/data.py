from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

train_dataset = MNIST(root='data/', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='data/', train=False, download=True, transform=ToTensor())

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
