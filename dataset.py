from torchvision import datasets, transforms
from torch.utils.data import Dataset

## I told ChatGPT to make this lol, edited some parts tho

# mnist_path = 'MFD/MNIST'
# fsdd_path = 'MFD/FSDD'

# transform = transforms.Compose([
#     transforms.Grayscale(),  # Convert to grayscale
#     transforms.ToTensor(),   # Convert to tensor
#     transforms.Resize(size=(28,28), antialias=True)
# ])

# mnist_dataset = datasets.ImageFolder(root=mnist_path, transform=transform)
# fsdd_dataset = datasets.ImageFolder(root=fsdd_path, transform=transform)

# assert set(mnist_dataset.classes) == set(fsdd_dataset.classes), "Classes in datasets do not match!"

# Create a custom dataset for pairing
class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        item1, label1 = self.dataset1[index]
        item2, label2 = self.dataset2[index]
        return {'image1': item1, 'label1': label1, 'image2': item2, 'label2': label2}

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

# # Create the paired dataset
# paired_dataset = PairedDataset(mnist_dataset, fsdd_dataset)
   



