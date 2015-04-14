import numpy as np
import theano
import theano.tensor as T

#rng = np.random.RandomState()
rng = np.random.RandomState(123456)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

floatX = theano.config.floatX

def get_activation_function(name):
    if name == "sigmoid":
        return T.nnet.sigmoid
    elif name == "tanh":
        return T.tanh
    elif name == "relu":
        return lambda x: x * (x > 0)
    else:
        raise ValueError("Unknown activation function")
