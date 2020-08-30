import numpy as np

class Loss:
    def __init__(self):
        self.logit = None
        self.label = None

    def forward(self, logit, label):

class Activation:
    def __init__(self):
        pass

    def ReLU(self, x):
        """
        ReLU activator
        :param x: input
        :return: max(x, 0)
        """
        return max(x, 0.)

    def LeakyReLU(self, x, k):
        """
        Leaky ReLU activator
        :param x: input
        :param k: multiplier
        :return: max(x*k, x)
        """
        return np.where(x >= 0, x, x * k)

    def Tanh(self, x):
        """
        Tanh activator
        :param x: input
        :return: tanh(x)
        """
        return np.tanh(x)

    def Sigmoid(self, x):
        """
        Sigmoid activator
        :param x: input
        :return: sigmoid(x)
        """
        return 1. / (1. + np.exp(-x))

    def dReLU(self, x):
        """
        Derivative of ReLU for calculating gradients
        :param x: input
        :return: ReLU(x)
        """
        return self.ReLU(x)

    def dLeakyReLU(self, x, k):
        """
        Derivative of LeakyReLU for calculating gradients
        :param x: input
        :return: Derivative of LeakyReLU
        """
        return np.where(x >= 0, 1., k)

    def dTanh(self, x):
        """
        Derivative of Tanh for calculating gradients
        :param x: input
        :return: Derivative of Tanh
        """
        return 1. - self.Tanh(x) ** 2

    def dSigmoid(self, x):
        """
        Derivative of Sigmoid for calculating gradients
        :param x: input
        :return: Derivative of Sigmoid
        """
        return self.Sigmoid(x) * (1. - self.Sigmoid(x))


