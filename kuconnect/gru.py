import theano
import theano.tensor as T
import numpy as np
import random
from inits import *
from dropout import *

floatX = theano.config.floatX

def initialize_weights(n_in, n_hidden, bias=False, scale=0.01):
    W_xr = normal(n_in, n_hidden, name='W_xr', scale=0.01)
    W_hr = orthogonal(n_hidden, n_hidden, name='W_hr')
    W_xz = normal(n_in, n_hidden, name='W_xz')
    W_hz = orthogonal(n_hidden, n_hidden, name='W_hz')
    W_xh = normal(n_in, n_hidden, name='W_xh')
    W_hh = orthogonal(n_hidden, n_hidden, name='W_hh')
    
    if bias:
        b_r = theano.shared(np.zeros((n_hidden,),dtype=floatX), name='b_r')
        b_z = theano.shared(np.ones((n_hidden,),dtype=floatX) * 10, name='b_z')
        b_h = theano.shared(np.zeros((n_hidden,),dtype=floatX), name='b_h')
        return W_xr, W_hr, W_xz, W_hz, W_xh, W_hh, b_r, b_z, b_h
    else:
        return W_xr, W_hr, W_xz, W_hz, W_xh, W_hh

class GRU(object):
    def __init__(self, input, d_input, n_in, n_hidden, bias=True, truncate=-1,
        dropout_rate=0, scale=0.01):

        self.input = input
        self.d_input = d_input
        self.n_in = n_in
        self.n_out = n_hidden

        if bias:
            self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh, \
            self.b_r, self.b_z, self.b_h \
            = initialize_weights(n_in, n_hidden, bias, scale)
            self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh, \
            self.b_r, self.b_z, self.b_h]
        else:
            self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh, \
            = initialize_weights(n_in, n_hidden, bias, scale)
            self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz,
            self.W_xh, self.W_hh]

        self.h0 = zeros(n_hidden, name='h0')
        self.d_h0 = zeros(n_hidden, name='d_h0')
        
        def step(x_t, h_tm1):
            tot_r = T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr)
            r_t = T.nnet.sigmoid(tot_r + self.b_r) if bias else T.nnet.sigmoid(tot_r)

            tot_z = T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz)
            z_t = T.nnet.sigmoid(tot_z + self.b_z) if bias else T.nnet.sigmoid(tot_z)
            
            tot_h = T.dot(x_t, self.W_xh) + T.dot(h_tm1 * r_t, self.W_hh)
            h_c = T.tanh(tot_h + self.b_h) if bias else T.tanh(tot_h)
            h_t = (1 - z_t) * h_tm1 + z_t * h_c
            return h_t
        
        self.h, _ = theano.scan(step, sequences=self.input,outputs_info=[self.h0], n_steps=self.input.shape[0], truncate_gradient=truncate)
        self.d_h, _ = theano.scan(step, sequences=self.input,outputs_info=[self.d_h0], n_steps=self.d_input.shape[0], truncate_gradient=truncate)
        
        self.output = self.h
        self.d_output = dropout(self.d_h, dropout_rate)
        self.memo = [(self.h0, self.h[-1])]

class BidirectionalGRU(object):
    def __init__(self, f_input, f_d_input, b_input, b_d_input, n_in, n_hidden, bias=True, truncate=-1,
        dropout_rate=0, scale=0.01):
        
        self.f_input = f_input
        self.f_d_input = f_d_input
        self.b_input = b_input[::-1, :]
        self.b_d_input = b_d_input[::-1, :]
        
        self.n_in = n_in
        self.n_out = n_hidden

        self.forw = l = GRU(self.f_input, self.f_d_input, n_in, n_hidden,
            dropout_rate=dropout_rate, bias=bias, truncate=truncate)
        
        self.back = l = GRU(self.b_input, self.b_d_input, n_in, n_hidden,
            dropout_rate=dropout_rate, bias=bias, truncate=truncate)

        self.params = self.forw.params + self.back.params

        self.f_output = self.forw.output
        self.f_d_output = self.forw.d_output
        self.b_output = self.back.output[::-1, :]
        self.b_d_output = self.back.d_output[::-1, :]
        self.memo = self.forw.memo + self.back.memo
