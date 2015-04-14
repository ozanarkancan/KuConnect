import numpy as np
from theano import shared
import theano
from learning.utils import *

def uniform(n_in, n_out, name, scale=1, mode="None"):
	if mode == "xavier":
		vals = np.asarray(
			rng.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
			), dtype=floatX
		)
	else:
		vals = np.asarray(
			rng.uniform(
			low=-1*scale,
			high=1*scale,
			size=(n_in, n_out)
			), dtype=floatX
		)
	
	return shared(vals, name=name, borrow=True)

def normal(n_in, n_out, name, scale=1):
	vals = np.asarray(
		rng.randn(n_in, n_out) * scale, dtype=floatX
	)
	
	return shared(vals, name=name, borrow=True)

def orthogonal(n_in, n_out, name, scale=1):
	assert n_in == n_out
	vals = rng.uniform(size=(n_in, n_out))
	u,_,_ = np.linalg.svd(vals)
	vals = np.asarray(u * scale, dtype=floatX)
	return shared(vals, name=name, borrow=True)

def identity(n_in, n_out, name, scale=1):
    assert n_in == n_out
    vals = np.eye(n_in, dtype='float32')
    return shared(vals, name=name, borrow=True)

def zeros(n, name):
    return shared(np.zeros((n,), dtype=floatX), name=name)

def defaults(n, name, scale):
    return shared(np.ones((n, )) * scale, dtype=floatX, name=name)
    
