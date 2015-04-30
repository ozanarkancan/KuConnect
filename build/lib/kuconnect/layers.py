import numpy as np
import theano
import theano.tensor as T
from dropout import *
from inits import *

def initialize_weights(n_in, n_out, bias=True, init="normal", scale=0.01):
    if init == "normal":
        init_func = normal
    elif init == "uniform":
        init_func = uniform
    elif init == "orthogonal":
        init_func = orthogonal
    else:
        raise ValueError("Unknown weight initialization function")

    W = init_func(n_in, n_out, name='W', scale=scale)

    if bias:
        b = zeros(n_out, name='b')
        return W, b
    else:
        return W

class InputLayer(object):
	def __init__(self, input, srng, dropout_rate=0.5):
		self.output = input
		self.d_output = dropout(input, dropout_rate)
		self.params = None

class OutputLayer(object):
	def __init__(self, input, d_input, n_in, n_out, bias=True, losstype="softmax"):
		self.n_in = n_in
		self.n_out = n_out

		self.W, self.b = initialize_weights(n_in, n_out)

		self.output = T.dot(input, self.W) + self.b
		self.d_output = T.dot(d_input, self.W) + self.b

        if losstype == "softmax":
		    self.p_y_given_x = T.nnet.softmax(self.output)
		    self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		    self.output = self.p_y_given_x
		    self.loss = lambda y: -T.mean(T.log(self.p_y_given_x + 1e-8)[T.arange(y.shape[0]), y])
		    self.error = lambda y: T.mean(T.neq(self.y_pred, y))

		    self.d_p_y_given_x = T.nnet.softmax(self.d_output)
		    self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
		    self.d_output = self.d_p_y_given_x
		    self.d_loss = lambda y: -T.mean(T.log(self.d_p_y_given_x + 1e-8)[T.arange(y.shape[0]), y])
		    self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
        elif losstype == "mse":
            self.loss = lambda y: T.mean((self.out - y) ** 2)
            self.d_loss = lambda y: T.mean((self.d_out - y) ** 2)
        else:
            raise ValueError('Unknown loss')

		self.params = [self.W, self.b]

class Layer(object):
	def __init__(self, input, d_input, n_in, n_out, srng, dropout_rate=0.5, activation="relu"):
		self.n_in = n_in
		self.n_out = n_out

		self.W, self.b = initialize_weights(n_in, n_out)

		self.output = T.dot(input, self.W) + self.b
		self.d_output = T.dot(d_input, self.W) + self.b

		if activation == "relu":
			act = lambda x : x * (x > 1e-6)
		elif activation == "tanh":
			act = T.tanh
		elif activation == "sigmoid":
			act = T.nnet.sigmoid
		else:
			act = None
		
		self.output = self.output if activation is None else act(self.output)
		self.d_output = self.d_output if activation is None else act(self.d_output)

		self.d_output = dropout(self.d_output, srng, dropout_rate)
		self.output = self.output

		self.params = [self.W, self.b]
