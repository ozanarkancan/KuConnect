import theano
from utils import srng

def dropout(input, dropout_rate=0):
    if dropout_rate > 0:
        retain = 1 - dropout_rate
        d_output = (input / retain) * srng.binomial(input.shape, p=retain,
            dtype='int32').astype('float32')
    else:
        d_output = input

    return d_output
