import math
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from CVAE.CVAE import Generator, initialize_weights, loss_fn
from dataset import PairedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4  # Karpathy constant
BATCH_SIZE = 64
CHANNELS_IMG = 1
NUM_EPOCHS = 100
FEATURES = 64

mnist_path = 'MFD/MNIST'
fsdd_path = 'MFD/FSDD'

transform_mnist = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to tensor
    transforms.Resize(size=(28,28), antialias=True)
])

transform_fsdd = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to tensor
    transforms.Resize(size=(28,28), antialias=True)
])

mnist_dataset = datasets.ImageFolder(root=mnist_path, transform=transform_mnist)
fsdd_dataset = datasets.ImageFolder(root=fsdd_path, transform=transform_fsdd)

paired_dataset = PairedDataset(mnist_dataset, fsdd_dataset)
loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(gen)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (target, data) in enumerate(loader):
        target = target.to(device)
        data = data.to(device)

        fake, mu, logvar = gen(data)
        loss_gen, bce, kld = loss_fn(fake, target, mu, logvar)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if  batch_idx % BATCH_SIZE == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                loss G: {loss_gen:.4f}, BCE: {bce:.4f}, KLD: {kld:.4f}"
            )

            with torch.no_grad():

                fake, mu, logvar = gen(data)

                # take out (up to) 32 examples
                img_grid_real = make_grid(target[:32], normalize=True)
                img_grid_fake = make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1