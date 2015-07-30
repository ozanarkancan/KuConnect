import theano
import theano.tensor as T
import numpy as np
import random
from inits import *
from dropout import *

floatX = theano.config.floatX
rng = np.random.RandomState()

def initialize_weights(n_in, n_hidden, bias=False, scale=0.01):
    W_xi = normal(n_in, n_hidden, name='W_xi', scale=scale)
    W_hi = orthogonal(n_hidden, n_hidden, name='W_hi')
    W_ci = theano.shared(np.asarray(rng.normal(size=(n_hidden,)) * scale, dtype=floatX), name='W_ci')
    W_xf = normal(n_in, n_hidden, name='W_xf', scale=scale)
    W_hf = orthogonal(n_hidden, n_hidden, name='W_hf')
    W_cf = theano.shared(np.asarray(rng.normal(size=(n_hidden,)) * scale, dtype=floatX), name='W_cf')
    W_xc = normal(n_in, n_hidden, scale=scale, name='W_xc')
    W_hc = orthogonal(n_hidden, n_hidden, name='W_hc')
    W_xo = normal(n_in, n_hidden, scale=scale, name='W_xo')
    W_ho = orthogonal(n_hidden, n_hidden, name='W_ho')
    W_co = theano.shared(np.asarray(rng.normal(size=(n_hidden,)) * scale, dtype=floatX), name='W_co')
    
    if bias:
        b_i = zeros(n_hidden, name='b_i')
        b_f = theano.shared(np.ones((n_hidden,),dtype=floatX) * 20, name='b_f')
        b_c = zeros(n_hidden, name='b_c')
        b_o = zeros(n_hidden, name='b_o')
        return W_xi, W_hi, W_ci, W_xf, W_hf, W_cf, W_xc, W_hc, W_xo, W_ho, W_co, b_i, b_f, b_c, b_o
    else:
        return W_xi, W_hi, W_ci, W_xf, W_hf, W_cf, W_xc, W_hc, W_xo, W_ho, W_co

class LSTM(object):
    def __init__(self, input, d_input, n_in, n_hidden, h0=None, d_h0=None, bias=True,
        dropout_rate=0, truncate=-1, scale=0.01):
        
        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden
        
        if bias:
            self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, self.W_xo, \
            self.W_ho, self.W_co, self.b_i, self.b_f, self.b_c, self.b_o \
            = initialize_weights(n_in, n_hidden, bias, scale)
            
            self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, \
            self.W_xo, self.W_ho, self.W_co, self.b_i, self.b_f, self.b_c, self.b_o]
        else:
            self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, self.W_xo, \
            self.W_ho, self.W_co = initialize_weights(n_in, n_hidden, bias, scale)
            
            self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, \
            self.W_xo, self.W_ho, self.W_co]
        
        self.h0 = zeros(n_hidden, name='h0') if h0 == None else h0
        self.c0 = zeros(n_hidden, name='c0')
        
        self.d_h0 = zeros(n_hidden, name='d_h0') if d_h0 == None else d_h0
        self.d_c0 = zeros(n_hidden, name='d_c0')
 
        def step(x_t, h_tm1, c_tm1):
            tot_i = T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci
            i_t = T.nnet.sigmoid(tot_i + self.b_i) if bias else T.nnet.sigmoid(tot_i)

            tot_f = T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf
            f_t = T.nnet.sigmoid(tot_f + self.b_f) if bias else T.nnet.sigmoid(tot_f)

            tot_c = T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc)
            c_t = f_t * c_tm1 + i_t * T.tanh(tot_c + self.b_c) if bias else f_t * c_tm1 + i_t * T.tanh(tot_c)

            tot_o = T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + c_t * self.W_co
            o_t = T.nnet.sigmoid(tot_o + self.b_o) if bias else T.nnet.sigmoid(tot_o)
            h_t = o_t * T.tanh(c_t)
            return h_t, c_t
            
        [self.h, self.c], _ = theano.scan(step, sequences=self.input,
            outputs_info=[self.h0, self.c0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        [self.d_h, self.d_c], _ = theano.scan(step, sequences=self.d_input,
            outputs_info=[self.d_h0, self.d_c0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        self.output = self.h
        self.d_output = dropout(self.d_h, dropout_rate)
        self.memo = [(self.h0, self.h[-1]), (self.c0, self.c[-1])]

class LSTMPeephole(object):
    def __init__(self, input, d_input, n_in, n_hidden, h0=None, d_h0=None, bias=True,
        dropout_rate=0, truncate=-1, scale=0.01):
        
        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden
        
        if bias:
            self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, self.W_xo, \
            self.W_ho, self.W_co, self.b_i, self.b_f, self.b_c, self.b_o \
            = initialize_weights(n_in, n_hidden, bias, scale)
            
            self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc, self.W_hc, \
            self.W_xo, self.W_ho, self.b_i, self.b_f, self.b_c, self.b_o]
        else:
            self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf, self.W_xc, self.W_hc, self.W_xo, \
            self.W_ho, self.W_co = initialize_weights(n_in, n_hidden, bias, scale)
            
            self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc, self.W_hc, \
            self.W_xo, self.W_ho]
        
        self.h0 = zeros(n_hidden, name='h0') if h0 == None else h0
        self.c0 = zeros(n_hidden, name='c0')
        
        self.d_h0 = zeros(n_hidden, name='d_h0') if d_h0 == None else d_h0
        self.d_c0 = zeros(n_hidden, name='d_c0')
 
        def step(x_t, h_tm1, c_tm1):
            tot_i = T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi)
            i_t = T.tanh(tot_i + self.b_i) if bias else T.tanh(tot_i)

            tot_f = T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf)
            f_t = T.nnet.sigmoid(tot_f + self.b_f) if bias else T.nnet.sigmoid(tot_f)

            tot_c = T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc)
            c_t = f_t * c_tm1 + i_t * T.nnet.sigmoid(tot_c + self.b_c) if bias else f_t * c_tm1 + i_t * T.nnet.sigmoid(tot_c)

            tot_o = T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho)
            o_t = T.tanh(tot_o + self.b_o) if bias else T.tanh(tot_o)
            h_t = o_t * T.tanh(c_t)
            return h_t, c_t
            
        [self.h, self.c], _ = theano.scan(step, sequences=self.input,
            outputs_info=[self.h0, self.c0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        [self.d_h, self.d_c], _ = theano.scan(step, sequences=self.d_input,
            outputs_info=[self.d_h0, self.d_c0],
            n_steps=self.input.shape[0],
            truncate_gradient=truncate)

        self.output = self.h
        self.d_output = dropout(self.d_h, dropout_rate)
        self.memo = [(self.h0, self.h[-1]), (self.c0, self.c[-1])]

class BidirectionalLSTM(object):
    def __init__(self, input, d_input, n_in, n_hidden,
        bias=True, dropout_rate=0, truncate=-1, scale=0.01):
        
        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden

        self.forw = LSTM(input, d_input, n_in, n_hidden,
            dropout_rate=dropout_rate, bias=bias, truncate=truncate)
        
        self.back = LSTM(input[::-1, :], d_input[::-1, :], n_in, n_hidden,
            dropout_rate=dropout_rate, bias=bias, truncate=truncate)

        self.params = self.forw.params + self.back.params

        self.f_output = self.forw.output
        self.f_d_output = self.forw.d_output
        self.b_output = self.back.output[::-1, :]
        self.b_d_output = self.back.d_output[::-1, :]
        
        self.memo = self.forw.memo + self.back.memo
