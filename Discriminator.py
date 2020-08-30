from functions import Weights as w
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
        pass

    def backward(self, output):
        pass