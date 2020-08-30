import numpy as np


class Loss:
    def __init__(self):
        self.logit = None
        self.label = None

    def forward(self, logit, label):
        """
        Binary Cross Entropy Loss implementation
        :param logit: Prediction
        :param label: Target
        :return: Calculated loss
        """
        if logit[0, 0] < 1e-7:
            logit[0, 0] = 1e-7
        if 1. - logit[0, 0] < 1e-7:
            logit[0, 0] = 1. - 1e-7
        self.logit = logit
        self.label = label
        return -(label * np.log(logit) + (1 - label) * np.log(1-logit))

    def backward(self):
        """
        Backward propogation for the neural network
        :return: Calculated gradient
        """
        return (1 - self.label) / (1 - self.logit) - self.label / self.logit


class Weights:
    @classmethod
    def init_weights(cls, in_c, out_c):
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


class Activation:
    @classmethod
    def ReLU(cls, x):
        """
        ReLU activator
        :param x: input
        :return: max(x, 0)
        """
        return max(x, 0.)

    @classmethod
    def LeakyReLU(cls, x, k):
        """
        Leaky ReLU activator
        :param x: input
        :param k: multiplier
        :return: max(x*k, x)
        """
        return np.where(x >= 0, x, x * k)

    @classmethod
    def Tanh(cls, x):
        """
        Tanh activator
        :param x: input
        :return: tanh(x)
        """
        return np.tanh(x)

    @classmethod
    def Sigmoid(cls, x):
        """
        Sigmoid activator
        :param x: input
        :return: sigmoid(x)
        """
        return 1. / (1. + np.exp(-x))

    @classmethod
    def dReLU(cls, x):
        """
        Derivative of ReLU for calculating gradients
        :param x: input
        :return: ReLU(x)
        """
        return Activation.ReLU(x)

    @classmethod
    def dLeakyReLU(cls, x, k):
        """
        Derivative of LeakyReLU for calculating gradients
        :param x: input
        :return: Derivative of LeakyReLU
        """
        return np.where(x >= 0, 1., k)

    @classmethod
    def dTanh(cls, x):
        """
        Derivative of Tanh for calculating gradients
        :param x: input
        :return: Derivative of Tanh
        """
        return 1. - Activation.Tanh(x) ** 2

    @classmethod
    def dSigmoid(cls, x):
        """
        Derivative of Sigmoid for calculating gradients
        :param x: input
        :return: Derivative of Sigmoid
        """
        return Activation.Sigmoid(x) * (1. - Activation.Sigmoid(x))


