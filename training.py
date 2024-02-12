import math
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from VAEGAN.VAEGAN import Discriminator, Generator, initialize_weights, loss_fn
from dataset import PairedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4  # Karpathy constant
BATCH_SIZE = 32
CHANNELS_IMG = 1

NUM_EPOCHS = 100

FEATURES = 64
NOISE_DIM = 100
LATENT_SPACE = 10

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

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES, LATENT_SPACE).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
criterion = nn.BCELoss()
criterion2 = nn.BCELoss(reduction='sum')

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):

    for batch_idx, (target, data) in enumerate(loader):
        mnist = target.to(device)
        audio = data.to(device)
        noise = torch.randn(audio.shape[0], NOISE_DIM, 28, 28).to(device)
        fake, mu, logvar = gen(noise, audio)


        disc_mnist = disc(mnist, audio).reshape(-1).to(device)
        loss_disc_mnist = criterion(disc_mnist, torch.ones_like(disc_mnist).to(device))

        disc_fake = disc(fake.detach(), audio).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake).to(device))
        loss_disc = (loss_disc_mnist + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake, audio).reshape(-1)
        loss_gen, discbce, bce, kld = loss_fn(fake, mnist, output, mu, logvar)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % BATCH_SIZE == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, D_BCE: {discbce:.4f}, BCE: {bce:.4f}, KLD: {kld:.4f}"
            )

            # print(
            #     f"Epoch [{epoch}/{NUM_EPOCHS}] \
            #     Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, BCE: {bce:.4f}, KLD: {kld:.4f}"
            # )

            with torch.no_grad():
                fake, mu, logvar = gen(noise, audio)
                # take out (up to) 32 examples
                img_grid_real = make_grid(mnist[:32], normalize=True)
                img_grid_fake = make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1