import numpy as np
import matplotlib.pyplot as plt

from functions import Activation as A, Loss
from sample_data import sample
import Generator
import Discriminator


if __name__ == '__main__':
    criterion = Loss()

