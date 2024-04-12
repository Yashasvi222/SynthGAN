import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_real_samples(n):
    data = np.random.randn(n)  # generates random data observations
    return data


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Dropout(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# function to train discriminator
def train_discriminator(discriminator, optimizer, real_data, fake_data):
    optimizer_D.zero_grad()

    # train on real data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, torch.ones_like(prediction_real))
    error_real.backward()  # backpropagation

    # train on fake data
    prediction_fake = discriminator(fake_data.detach())
    error_fake = loss(prediction_fake, torch.zeros_like(prediction_fake))
    error_fake.backward()

    optimizer_D.step()

    return error_real + error_fake


# function to train generator
def train_generator(generator, optimizer, fake_data):
    optimizer_G.zero_grad()

    prediction = discriminator(fake_data)
    error = loss(prediction, torch.ones_like(prediction))
    error.backward

    optimizer_G.step()

    return error


# Hyperparameters
batch_size = 128
lr = 3e-4
epochs = 5000

# models and optimizers
generator = Generator()
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
loss = nn.BCELoss()
latent_dim = 128

# Training Loop
for epoch in range(epochs):
    # Generate Real and Fake data
    real_data = torch.Tensor(generate_real_samples(batch_size)).view(-1, 1)
    fake_data = generator(Variable(torch.randn(batch_size, latent_dim)))

    # train discriminator
    d_loss = train_discriminator(discriminator, optimizer_D, real_data, fake_data)

    # train generator
    g_loss = train_generator(discriminator, optimizer_G, fake_data)

    if epoch % 100 == 0:
        print(f"Epoch:{epoch}, D Loss:{d_loss.item()}, G Loss: {g_loss.item()}")

# Generate synthetic time-series data using the trained Generator
generated_data = generator(torch.randn(100, latent_dim)).detach().numpy()

# Plot the generated data
plt.plot(generated_data)
plt.title("Generated time-series data")
plt.show()
