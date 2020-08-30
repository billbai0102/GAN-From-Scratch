import numpy as np
from functions import Weights as w, Activation as a
import params as p


class Generator:
    def __init__(self):
        self.input = None

        self.weight1 = w.init_weights(p.in_dim, p.gen_hidden)
        self.bias1 = w.init_weights(1, p.gen_hidden)
        self.out1 = None

        self.weight2 = w.init_weights(p.gen_hidden, p.gen_hidden)
        self.bias2 = w.init_weights(1, p.gen_hidden)
        self.out2 = None

        self.weight3 = w.init_weights(p.gen_hidden, p.out_dim)
        self.bias3 = w.init_weights(1, p.out_dim)
        self.out3 = None

        self.output = None

    def forward(self, input):
        """
        x_l = f(out_(l-1) * w_l + b_l), where f(x) can be ReLU or Tanh depending on layer
        :param input: initial input
        :return: forward propagated output
        """
        self.input = input.reshape(1, p.in_dim)

        self.out1 = np.matmul(self.input, self.weight1) + self.bias1
        self.out1 = a.ReLU(self.out1)

        self.out2 = np.matmul(self.out1, self.weight2) + self.bias2
        self.out2 = a.ReLU(self.out2)

        self.out3 = np.matmul(self.out2, self.weight3) + self.bias3
        self.output = a.Tanh(self.out3)

    def backward(self, output):
        pass
