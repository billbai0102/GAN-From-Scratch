import numpy as np
import matplotlib.pyplot as plt

from functions import Activation, Loss
from generate_data import sample


def init_weights(in_c, out_c):
    """
    Initializes weights
    :param in_c: input channels
    :param out_c: output channels
    :return: initialized weights
    """
    scale = np.sqrt(2. / (in_c + out_c))
    weights = np.random.uniform(-scale, scale, (in_c, out_c))
    print(f'Weights initialized: {weights}')
    return weights


if __name__ == '__main__':
    a = Activation()
