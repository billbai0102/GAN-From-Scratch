import numpy as np
import matplotlib.pyplot as plt

import params as p
from functions import Loss
from sample_data import sample
from Generator import Generator
from Discriminator import Discriminator


if __name__ == '__main__':
    """
    Trains model using SGD
    """
    criterion = Loss()
    Generator = Generator()
    Discriminator = Discriminator()

    label = {
        'real': 1,
        'fake': 0
    }

    for epoch in range(p.epochs):
        # Train discriminator to recognize **real** data
        target_real = sample()
        pred_real = Discriminator.forward(target_real)
        d_loss_real = criterion.forward(pred_real, label['real'])
        d_loss_derivative = criterion.backward()
        Discriminator.backward(d_loss_derivative)

        # Train discriminator to recognize **fake** data
        noise = np.random.randn(p.gen_in)
        target_fake = Generator.forward(noise)
        pred_fake = Discriminator.forward(target_fake)
        d_loss_fake = criterion.forward(pred_fake, label['fake'])
        d_loss_derivative = criterion.backward()
        Discriminator.backward(d_loss_derivative)

        # Train generator create real-looking data
        gd_loss_fake = Discriminator.forward(target_fake)
        g_loss = criterion.forward(gd_loss_fake, label['real'])
        g_loss_derivative = Discriminator.backward(g_loss, False)
        Generator.backward(g_loss_derivative)
        loss_D = d_loss_real + d_loss_fake

        # print status updates every 100
        if epoch % 100 == 0:
            print(f'Discriminator Loss (Real): {d_loss_real.item((0, 0)):.4f}\t'
                  f'Discriminator Loss (Fake): {d_loss_fake.item((0, 0)):.4f}\t'
                  f'Generator Loss: {g_loss.item((0, 0)):.4f}')

    # show generator data
    x = [i for i in range(p.gen_out)]
    for i in range(50):
        noise = np.random.randn(p.gen_in)
        y = Generator.forward(noise).reshape(p.gen_out)
        plt.plot(x, y)
    plt.show()
