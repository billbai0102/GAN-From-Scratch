import numpy as np
from functions import Weights as w, Activation as a
import params as p


class Generator(object):
    def __init__(self):
        # input layer
        self.input = None

        # hidden layer 1
        self.weight1 = w.init_weights(p.gen_in, p.gen_hidden)
        self.bias1 = w.init_weights(1, p.gen_hidden)
        self.out1 = None

        # hidden layer 2
        self.weight2 = w.init_weights(p.gen_hidden, p.gen_hidden)
        self.bias2 = w.init_weights(1, p.gen_hidden)
        self.out2 = None

        # hidden layer 3
        self.weight3 = w.init_weights(p.gen_hidden, p.gen_out)
        self.bias3 = w.init_weights(1, p.gen_out)
        self.out3 = None

        # output layer
        self.output = None

    def forward(self, input):
        """
        x_l = f(out_(l-1) * w_l + b_l), where f(x) can be ReLU or Tanh depending on layer
        :param input: initial input
        :return: forward propagated output
        """
        self.input = input.reshape(1, p.gen_in)

        # calculate output layer 1 + ReLU activation
        self.out1 = np.matmul(self.input, self.weight1) + self.bias1
        self.out1 = a.ReLU(self.out1)

        # calculate output layer 2 + ReLU activation
        self.out2 = np.matmul(self.out1, self.weight2) + self.bias2
        self.out2 = a.ReLU(self.out2)

        # calculate output layer 3
        self.out3 = np.matmul(self.out2, self.weight3) + self.bias3

        # calculate output layer w/ Tanh activation
        self.output = a.Tanh(self.out3)
        
        return self.output

    def backward(self, outputs):
        """
        Updates weights and biases through back propagation / gradient **descent**
        :param output: output of model
        """
        d = outputs
        d *= a.dTanh(self.output)

        '''
        Third Layer
        '''
        # calculate derivatives with respects to weight and bias
        d_weight3 = np.matmul(np.transpose(self.out2), d)
        d_bias3 = d.copy()

        # calculate output derivative
        d = np.matmul(d, np.transpose(self.weight3))

        # update weight
        if np.linalg.norm(d_weight3) > p.grad_clip:
            d_weight3 = p.grad_clip / np.linalg.norm(d_weight3) * d_weight3
        self.weight3 -= p.gen_step * d_weight3
        self.weight3 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight3))

        # update bias
        self.bias3 -= p.gen_step * d_bias3
        self.bias3 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias3))

        # recalculate output derivative
        d *= a.dReLU(self.out2)

        '''
        Second Layer
        '''
        # calculate derivatives with respects to weight and bias
        d_weight2 = np.matmul(np.transpose(self.out1), d)
        d_bias2 = d.copy()

        # calculate output derivative
        d = np.matmul(d, np.transpose(self.weight2))

        # update weight
        if np.linalg.norm(d_weight2) > p.grad_clip:
            d_weight2 = p.grad_clip / np.linalg.norm(d_weight2) * d_weight2
        self.weight2 -= p.gen_step * d_weight2
        self.weight2 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight2))

        # update bias
        self.bias2 -= p.gen_step * d_bias2
        self.bias2 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias2))

        # recalculate out derivative
        d *= a.dReLU(self.out1)

        '''
        First Layer
        '''
        # calculate derivatives with respects to weight and bias
        d_weight1 = np.matmul(np.transpose(self.input), d)
        d_bias1 = d.copy()

        # update weight
        if np.linalg.norm(d_weight1) > p.grad_clip:
            d_weight1 = p.grad_clip / np.linalg.norm(d_weight1) * d_weight1
        self.weight1 -= p.gen_step * d_weight1
        self.weight1 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight1))

        # update bias
        self.bias1 -= p.gen_step * d_bias1
        self.bias1 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias1))




