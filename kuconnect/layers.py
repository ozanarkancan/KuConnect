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
    elif init == "identity":
        init_func = identity
    else:
        raise ValueError("Unknown weight initialization function")

    W = init_func(n_in, n_out, name='W', scale=scale)

    if bias:
        b = zeros(n_out, name='b')
        return W, b
    else:
        return W

class InputLayer(object):
	def __init__(self, input, dropout_rate=0.5):
		self.output = input
		self.d_output = dropout(input, dropout_rate)
		self.params = None

class BidirectionalInputLayer(object):
	def __init__(self, input, dropout_rate=0.5):
		self.f_output = input
		self.f_d_output = dropout(input, dropout_rate)
		self.b_output = input
		self.b_d_output = dropout(input, dropout_rate)
		self.params = None

class MeanPoolingLayer(object):
    def __init__(self, input, d_input, indices, n_in, n_out):
        self.input = input
        self.d_input = d_input
        self.params = None
        self.n_in = n_in
        self.n_out = n_out
        self.indices = indices
        
        def step(i, H):
            return T.mean(H[i[0]:i[1], :], 0)

        self.output, _ = theano.scan(fn=step,
            outputs_info=None,
            sequences=indices,
            non_sequences=self.input)
        
        self.d_output, _ = theano.scan(fn=step,
            outputs_info=None,
            sequences=indices,
            non_sequences=self.d_input)

        self.memo = None

class BidirectionalMeanPoolingLayer(object):
    def __init__(self, f_input, f_d_input, b_input, b_d_input, indices, n_in, n_out):
        self.f_input = f_input
        self.f_d_input = f_d_input
        self.b_input = b_input[::-1, :]
        self.b_d_input = b_d_input[::-1, :]
        self.params = None
        self.n_in = n_in
        self.n_out = n_out
        self.indices = indices

        self.forw = MeanPoolingLayer(self.f_input, self.f_d_input, indices, n_in, n_out)
        self.back = MeanPoolingLayer(self.b_input, self.b_d_input, indices, n_in, n_out)

        self.f_output = self.forw.output
        self.f_d_output = self.forw.d_output
        self.b_output = self.back.output[::-1, :]
        self.b_d_output = self.back.d_output[::-1, :]
        self.memo = None
 
class PoolingLayer(object):
    def __init__(self, input, d_input, indices, n_in, n_out):
        self.input = input
        self.d_input = d_input
        self.params = None
        self.n_in = n_in
        self.n_out = n_out
        self.indices = indices
        self.output = self.input[indices, :]
        self.d_output = self.d_input[indices, :]
        
        self.memo = None

class BidirectionalPoolingLayer(object):
    def __init__(self, f_input, f_d_input, b_input, b_d_input, indices, n_in, n_out):
        self.f_input = f_input
        self.f_d_input = f_d_input
        self.b_input = b_input[::-1, :]
        self.b_d_input = b_d_input[::-1, :]
        self.params = None
        self.n_in = n_in
        self.n_out = n_out
        self.indices = indices

        self.f_output = self.f_input[indices, :]
        self.f_d_output = self.f_d_input[indices, :]
        self.b_output = self.b_input[indices, :][::-1, :]
        self.b_d_output = self.b_d_input[indices, :][::-1, :]
        self.memo = None

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
            self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
            self.error = lambda y: T.mean(T.neq(self.y_pred, y))
            
            self.d_p_y_given_x = T.nnet.softmax(self.d_output)
            self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
            self.d_output = self.d_p_y_given_x
            self.d_loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.d_p_y_given_x, y))
            self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
        elif losstype == "mse":
            self.loss = lambda y: T.mean((self.output - y) ** 2)
            self.d_loss = lambda y: T.mean((self.d_output - y) ** 2)
        else:
            raise ValueError('Unknown loss')
        self.params = [self.W, self.b]

class BidirectionalOutputLayer(object):
    def __init__(self, f_input, f_d_input, b_input, b_d_input, n_in, n_out, bias=True, losstype="softmax"):
        self.n_in = n_in
        self.n_out = n_out
        
        self.Wf, self.b = initialize_weights(n_in, n_out)
        self.Wb = initialize_weights(n_in, n_out, bias=False)
        
        self.output = T.dot(f_input, self.Wf) + T.dot(b_input, self.Wb) + self.b
        self.d_output = T.dot(f_d_input, self.Wf) + T.dot(b_d_input, self.Wb) + self.b
        
        if losstype == "softmax":
            self.p_y_given_x = T.nnet.softmax(self.output)
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            self.output = self.p_y_given_x
            self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
            self.error = lambda y: T.mean(T.neq(self.y_pred, y))
            
            self.d_p_y_given_x = T.nnet.softmax(self.d_output)
            self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
            self.d_output = self.d_p_y_given_x
            self.d_loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.d_p_y_given_x, y))
            self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
        elif losstype == "mse":
            self.loss = lambda y: T.mean((self.output - y) ** 2)
            self.d_loss = lambda y: T.mean((self.d_output - y) ** 2)
        else:
            raise ValueError('Unknown loss')
        self.params = [self.Wf, self.Wb, self.b]

class RecurrentOutputLayer(object):
    def __init__(self, input, d_input, n_in, n_out, bias=True, losstype="softmax", recout=0): 
        self.n_in = n_in
        self.n_out = n_out
        
        self.W, self.b = initialize_weights(n_in, n_out)
        self.Wrs = []

        for i in xrange(recout):
            self.Wrs.append(initialize_weights(n_out, n_out, bias=False, init="identity"))

        if recout < 2:
            self.y0 = zeros(n_out, 'y0')
            self.d_y0 = zeros(n_out, 'd_y0')
        else:
            self.y0 = theano.shared(np.zeros((recout, n_out), dtype='float32'),name='y0')
            self.d_y0 = theano.shared(np.zeros((recout, n_out), dtype='float32'), name='d_y0')
        
        preact = T.dot(input, self.W)
        preact_d = T.dot(d_input, self.W)

        def step(x_t, *args):
            y_t = x_t + self.b
            for i in xrange(recout):
                y_t += T.dot(args[i], self.Wrs[i])

            if losstype == "softmax":
                y_t = T.nnet.softmax(y_t).dimshuffle((1,))
            
            return y_t

        o_info = self.y0 if recout < 2 else dict(initial=self.y0, taps=range(-1*recout,0))
        d_o_info = self.d_y0 if recout < 2 else dict(initial=self.d_y0, taps=range(-1*recout,0))

        self.output, _ = theano.scan(step,
            sequences=preact,
            outputs_info=[o_info],
            n_steps=input.shape[0])
        
        self.d_output, _ = theano.scan(step,
            sequences=preact_d,
            outputs_info=[d_o_info],
            n_steps=d_input.shape[0])

        if losstype == "softmax":
            self.p_y_given_x = self.output
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
            self.error = lambda y: T.mean(T.neq(self.y_pred, y))
            
            self.d_p_y_given_x = self.d_output
            self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
            self.d_loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.d_p_y_given_x, y))
            self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
        elif losstype == "mse":
            self.loss = lambda y: T.mean((self.output - y) ** 2)
            self.d_loss = lambda y: T.mean((self.d_output - y) ** 2)
        else:
            raise ValueError('Unknown loss')
        
        self.params = [self.W, self.b] + self.Wrs

class BidirectionalRecurrentOutputLayer(object):
    def __init__(self, f_input, f_d_input, b_input, b_d_input, n_in, n_out, bias=True, losstype="softmax", recout=0): 
        self.n_in = n_in
        self.n_out = n_out
        
        self.W_f, self.b = initialize_weights(n_in, n_out)
        self.W_b = initialize_weights(n_in, n_out, bias=False)
        self.Wrs = []

        for i in xrange(recout):
            self.Wrs.append(initialize_weights(n_out, n_out, bias=False, init="identity"))

        if recout < 2:
            self.y0 = zeros(n_out, 'y0')
            self.d_y0 = zeros(n_out, 'd_y0')
        else:
            self.y0 = theano.shared(np.zeros((recout, n_out), dtype='float32'),name='y0')
            self.d_y0 = theano.shared(np.zeros((recout, n_out), dtype='float32'), name='d_y0')

        def step(f_x_t, b_x_t, *args):
            y_t = T.dot(f_x_t, self.W_f) + T.dot(b_x_t, self.W_b) + self.b
            for i in xrange(recout):
                y_t += T.dot(args[i], self.Wrs[i])

            if losstype == "softmax":
                y_t = T.nnet.softmax(y_t).dimshuffle((1,))
            
            return y_t

        o_info = self.y0 if recout < 2 else dict(initial=self.y0, taps=range(-1*recout,0))
        d_o_info = self.d_y0 if recout < 2 else dict(initial=self.d_y0, taps=range(-1*recout,0))

        self.output, _ = theano.scan(step,
            sequences=[f_input, b_input],
            outputs_info=[o_info],
            n_steps=f_input.shape[0])
        
        self.d_output, _ = theano.scan(step,
            sequences=[f_d_input, b_d_input],
            outputs_info=[d_o_info],
            n_steps=b_d_input.shape[0])

        if losstype == "softmax":
            self.p_y_given_x = self.output
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
            self.error = lambda y: T.mean(T.neq(self.y_pred, y))
            
            self.d_p_y_given_x = self.d_output
            self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
            self.d_loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.d_p_y_given_x, y))
            self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))
        elif losstype == "mse":
            self.loss = lambda y: T.mean((self.output - y) ** 2)
            self.d_loss = lambda y: T.mean((self.d_output - y) ** 2)
        else:
            raise ValueError('Unknown loss')
        
        self.params = [self.W_f, self.W_b, self.b] + self.Wrs

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

