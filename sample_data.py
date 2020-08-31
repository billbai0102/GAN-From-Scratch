import matplotlib.pyplot as plt
import numpy as np
import params


def sample():
    """
    samples data for the GAN to replicate
    """
    shift = np.random.uniform(0, 1)
    frequency = np.random.uniform(1, 1.15)
    multiplier = np.random.uniform(.5, .7)
    data = [multiplier * np.sin(shift + frequency * i) for i in range(params.gen_out)]
    return np.array(data)


if __name__ == '__main__':
    x = [i for i in range(params.gen_out)]
    for i in range(10):
        y = sample()
        plt.plot(x, y)
    plt.show()
