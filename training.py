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
NUM_EPOCHS = 10
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
disc = Discriminator(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
criterion = nn.BCELoss()

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, batch in enumerate(loader):
        mnist = batch['image1'].to(device)
        audio = batch['image2'].to(device)
        fake = gen(audio)

        disc_mnist = disc(mnist).reshape(-1).to(device)
        loss_disc_mnist = criterion(disc_mnist, torch.ones_like(disc_mnist).to(device))

        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_mnist + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(audio)
                # take out (up to) 32 examples
                img_grid_real = make_grid(mnist[:32], normalize=True)
                img_grid_fake = make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


