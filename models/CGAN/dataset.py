from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


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



