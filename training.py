import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from VAEGAN.VAEGAN import Discriminator, Generator, initialize_weights
from dataset import PairedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES = 64

mnist_path = 'MFD/MNIST'
fsdd_path = 'MFD/FSDD'

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to tensor
    transforms.Resize(size=(28,28), antialias=True)
])

mnist_dataset = datasets.ImageFolder(root=mnist_path, transform=transform)
fsdd_dataset = datasets.ImageFolder(root=fsdd_path, transform=transform)

paired_dataset = PairedDataset(mnist_dataset, fsdd_dataset)
loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(CHANNELS_IMG, FEATURES).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

# gen.train()
# critic.train()

