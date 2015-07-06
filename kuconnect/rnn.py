import theano
import theano.tensor as T
from theano import shared
import numpy as np
import random
import cPickle
from inits import *
from dropout import *

floatX = theano.config.floatX

def initialize_weights(n_in, n_hidden, bias=False, init="normal", scale=0.01, n_out = None):
    """
    Initialize weights, use normal or uniform distribution
    """
    if init == "normal":
        init_func = normal
        init_func_rec = normal
    elif init == "uniform":
        init_func = uniform
        init_func_rec = uniform
    elif init == "identity":
        init_func = normal
        init_func_rec = identity
    else:
        raise ValueError('Unknown rnn weight initialization function')

    W_ih = init_func(n_in, n_hidden, name='W_ih', scale = scale)
    W_hh = init_func_rec(n_hidden, n_hidden, name='W_hh', scale = scale)
    params = [W_ih, W_hh]
    
    if n_out != None:
        W_hy = init_func(n_hidden, n_out, name='W_hy', scale = scale)
        W_yh = init_func(n_out, n_hidden, name='W_yh', scale = scale)
        params += [W_hy, W_yh]

    if bias:
        b_hh = zeros(n_hidden, name='b_hh')
        params += [b_hh]
        if n_out != None:
            b_hy = zeros(n_out, name='b_hh')
            params += [b_hy]

    return params
    

class Elman(object):
    def __init__(self, input, d_input, n_in, n_hidden, h0=None, d_h0=None, activation="tanh", bias=True,
        init="identity", scale=0.01, dropout_rate=0, truncate=-1):
        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden

        if bias:
            self.W_ih, self.W_hh, self.b_hh = initialize_weights(n_in,
                n_hidden, bias, init, scale)
            self.params = [self.W_ih, self.W_hh, self.b_hh]
        else:
            self.W_ih, self.W_hh = initialize_weights(n_in, n_hidden, bias,
                init, scale)
            self.params = [self.W_ih, self.W_hh]

        self.act = get_activation_function(activation)
        self.h0 = zeros(n_hidden, 'h0') if h0 == None else h0
        self.d_h0 = zeros(n_hidden, 'd_h0') if d_h0 == None else d_h0

        def step(x_t, h_tm1):
            tot = T.dot(x_t, self.W_ih) + T.dot(h_tm1, self.W_hh)
            if bias:
                tot += self.b_hh
            h_t = self.act(tot)
            return h_t

        self.h, _ = theano.scan(step,
            sequences=self.input,
            outputs_info=[self.h0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        self.d_h, _ = theano.scan(step,
            sequences=self.d_input,
            outputs_info=[self.d_h0],
            n_steps=self.d_input.shape[0],
            truncate_gradient=truncate)

        self.output = self.h
        self.d_output = dropout(self.d_h, dropout_rate)
        self.memo = [(self.h0, self.h[-1])]

class ElmanFeedback(object):
    def __init__(self, input, d_input, n_in, n_hidden, n_out, h0=None, d_h0=None, activation="tanh", bias=True,
        init="normal", scale=0.01, dropout_rate=0, truncate=-1):
        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_out

        if bias:
            self.W_ih, self.W_hh, self.W_hy, self.W_yh, self.b_hh, self.b_hy = initialize_weights(n_in,
                n_hidden, bias, init, scale, n_out)
            self.params = [self.W_ih, self.W_hh, self.W_hy, self.W_yh,
                self.b_hh, self.b_hy]
        else:
            self.W_ih, self.W_hh, self.W_hy, self.W_yh = initialize_weights(n_in, n_hidden, bias,
                init, scale, n_out)

            self.params = [self.W_ih, self.W_hh, self.W_hy, self.W_yh]

        self.act = get_activation_function(activation)
        
        self.h0 = zeros(n_hidden, 'h0') if h0 == None else h0
        self.d_h0 = zeros(n_hidden, 'd_h0') if d_h0 == None else d_h0
        
        self.y0 = zeros(n_out, 'y0')
        self.d_y0 = zeros(n_out, 'd_y0')

        def step(x_t, h_tm1, y_tm1):
            tot = T.dot(x_t, self.W_ih) + T.dot(h_tm1, self.W_hh) + T.dot(y_tm1, self.W_yh)
            h_t = self.act(tot + self.b_hh) if bias else self.act(tot)
            
            tot = T.dot(h_t, self.W_hy)
            y_t = tot + self.b_hy if bias else tot

            return h_t, y_t
        
        def d_step(x_t, h_tm1, y_tm1):
            tot = T.dot(x_t, self.W_ih) + T.dot(h_tm1, self.W_hh) + T.dot(y_tm1, self.W_yh)
            h_t = self.act(tot + self.b_hh) if bias else self.act(tot)
            d_h_t = dropout(h_t, dropout_rate)
            tot = T.dot(d_h_t, self.W_hy)
            y_t = tot + self.b_hy if bias else tot

            return h_t, y_t

        [self.h, self.y], _ = theano.scan(step,
            sequences=self.input,
            outputs_info=[self.h0, self.y0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        [self.d_h, self.d_y], _ = theano.scan(d_step,
            sequences=self.d_input,
            outputs_info=[self.d_h0, self.d_y0],
            n_steps=self.d_input.shape[0],
            truncate_gradient=truncate)

        self.p_y_given_x = T.nnet.softmax(self.y)
        self.d_p_y_given_x = T.nnet.softmax(self.d_y)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.d_y_pred = T.argmax(self.d_p_y_given_x, axis=1)
        
        self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
        self.error = lambda y: T.mean(T.neq(self.y_pred, y))

        self.d_loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.d_p_y_given_x, y))
        self.d_error = lambda y: T.mean(T.neq(self.d_y_pred, y))

        self.output = self.y
        self.d_output = self.d_y
        self.memo = [(self.h0, self.h[-1]), (self.y0, self.y[-1])]

class BidirectionalElman(object):
    def __init__(self, input, d_input, n_in, n_hidden, hf0=None, hb0=None,
        d_hf0=None, d_hb0=None, activation="tanh", bias=True,
        init="identity", scale=0.01, dropout_rate=0, truncate=-1):

        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden

        self.forw = Elman(input, d_input, n_in, n_hidden,
            activation=activation, bias=bias, init=init, scale=scale,
            dropout_rate=dropout_rate, truncate=truncate)
        
        self.back = Elman(input[:, ::-1], d_input[:, ::-1], n_in, n_hidden,
            activation=activation, bias=bias, init=init, scale=scale,
            dropout_rate=dropout_rate, truncate=truncate)

        self.params = self.forw.params + self.back.params

        self.f_output = self.forw.output
        self.f_d_output = self.forw.d_output
        self.b_output = self.back.output[::-1]
        self.b_d_output = self.back.d_output[::-1]
        self.memo = self.forw.memo + self.back.memo
