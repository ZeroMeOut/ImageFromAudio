from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

## I told ChatGPT to make this lol, edited some parts tho

mnist_path = 'MFD/MNIST'
fsdd_path = 'MFD/FSDD'

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to tensor
    transforms.Resize(size=(28,28), antialias=True)
])

mnist_dataset = datasets.ImageFolder(root=mnist_path, transform=transform)
fsdd_dataset = datasets.ImageFolder(root=fsdd_path, transform=transform)

# assert set(mnist_dataset.classes) == set(fsdd_dataset.classes), "Classes in datasets do not match!"

# Create a custom dataset for pairing
class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return  len(self.dataset2)
    
    def __getitem__(self, index):
        image1, label1 = self.dataset1[index]
        image2, label2 = self.dataset2[index]
        return image1, image2

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

# Create the paired dataset
# paired_dataset = PairedDataset(mnist_dataset, fsdd_dataset)
# loader = DataLoader(paired_dataset, batch_size=64, shuffle=True)

# v = 0
# epochs = 1
# batch_size = 64

# for epoch in range(epochs):
#     for batch_idx,(data, target) in enumerate(loader):
#         l1 = data
#         l2 = target[0]
#         print(l1)
#         break
#     break
        



