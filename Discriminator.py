import numpy as np
from functions import Activation as a, Weights as w
import params as p


class Discriminator(object):
    def __init__(self):
        # input layer
        self.input = None
        
        # hidden layer 1
        self.weight1 = w.init_weights(p.dis_in, p.dis_hidden)
        self.bias1 = w.init_weights(1, p.dis_hidden)
        self.out1 = None

        # hidden layer 2
        self.weight2 = w.init_weights(p.dis_hidden, p.dis_hidden)
        self.bias2 = w.init_weights(1, p.dis_hidden)
        self.out2 = None

        # hidden layer 3
        self.weight3 = w.init_weights(p.dis_hidden, p.dis_out)
        self.bias3 = w.init_weights(1, p.dis_out)
        self.out3 = None
        
        # output layer
        self.output = None

    def forward(self, inputs):
        """
        x_l = f(out_(l-1) * w_l + b_l), where f(x) can be LeakyReLU or Sigmoid depending on layer
        :param input: initial input
        :return: forward propagated output
        """
        # calculate input layer
        self.input = inputs.reshape(1, p.dis_in)

        # calculate hidden layer 1 + LeakyReLU activation
        self.out1 = np.matmul(self.input, self.weight1) + self.bias1
        self.out1 = a.LeakyReLU(self.out1)

        # calculate hidden layer 2 + LeakyReLU activation
        self.out2 = np.matmul(self.out1, self.weight2) + self.bias2
        self.out2 = a.LeakyReLU(self.out2)

        # calculate hidden layer 3
        self.out3 = np.matmul(self.out2, self.weight3) + self.bias3

        # calculate output layer w/ Sigmoid activation
        self.output = a.Sigmoid(self.out3)

        # return model output
        return self.output

    def backward(self, output, grad=True):
        """
        Updates weights and biases through back propagation / gradient **ascent**
        :param output: output of model
        :param grad: bool
        :return: ascent
        """
        d = output
        d *= a.dSigmoid(self.output)
        
        '''
        Third Layer
        '''
        # calculates derivatives with respect to weight and bias
        d_w3 = np.matmul(np.transpose(self.out2), d)
        d_b3 = d.copy()

        # calculates derivative of output
        d = np.matmul(d, np.transpose(self.weight3))

        if grad:
            # update weight
            if np.linalg.norm(d_w3) > p.grad_clip:
                d_w3 = p.grad_clip / np.linalg.norm(d_w3) * d_w3
            self.weight3 += p.dis_step * d_w3
            self.weight3 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight3))

            # update bias
            self.bias3 += p.dis_step * d_b3
            self.bias3 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias3))

        # update derivative with respect to activation function
        d *= a.dLeakyReLU(self.out2)
        
        '''
        Second Layer
        '''
        # calculates derivatives with respect to weight and bias
        d_w2 = np.matmul(np.transpose(self.out1), d)
        d_b2 = d.copy()

        # calculate output derivative
        d = np.matmul(d, np.transpose(self.weight2))

        if grad:
            # update weight
            if np.linalg.norm(d_w2) > p.grad_clip:
                d_w2 = p.grad_clip / np.linalg.norm(d_w2) * d_w2
            self.weight2 += p.dis_step * d_w2
            self.weight2 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight2))

            # update bias
            self.bias2 += p.dis_step * d_b2
            self.bias2 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias2))

        # update derivative with respect to activation function
        d *= a.dLeakyReLU(self.out1)
        
        '''
        First Layer
        '''
        # calculates derivatives with respect to weight and bias
        d_w1 = np.matmul(np.transpose(self.input), d)
        d_b1 = d.copy()

        # calculate output derivative
        d = np.matmul(d, np.transpose(self.weight1))

        if grad:
            # update weight
            if np.linalg.norm(d_w1) > p.grad_clip:
                d_w1 = p.grad_clip / np.linalg.norm(d_w1) * d_w1
            self.weight1 += p.dis_step * d_w1
            self.weight1 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.weight1))

            # update bias
            self.bias1 += p.dis_step * d_b1
            self.bias1 = np.maximum(-p.weight_clip, np.minimum(p.weight_clip, self.bias1))
        return d
