import argparse
from kuconnect.ldnn import *
import numpy as np
from theano import function
import time

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="adding_problem")
    parser.add_argument("--seq", default=150, type=int, help="length of the sequences")
    return parser

def get_data(seq):
    def get_integers():
        n1 = 0
        n2 = 0
        while n1 == n2:
            n1, n2 = np.random.randint(0, high=seq, size= (2, ))
        return n1, n2

    def get_one_instance():
        x1 = np.random.uniform(0, 1, size=(seq, ))
        n1, n2 = get_integers()
        x2 = np.zeros((seq, ))
        x2[[n1, n2]] = 1
        x = np.transpose(np.array([x1, x2]))
        y = np.dot(x1, x2)
        return x, y

    genX = lambda size: map(list, zip(*[get_one_instance() for i in xrange(size)])) 

    trainX, trainY = genX(int(1e5))
    devX, devY = genX(int(1e4))
    
    return trainX, trainY, devX, devY
    

if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    
    print "Building data"
    trainX, trainY, devX, devY = get_data(args["seq"])

    print "Building the model"
    u = T.matrix(dtype=theano.config.floatX)
    y = T.ivector()
    ldnn = LDNN()
    ldnn.add_input_layer(u, dropout_rate=0)
    ldnn.add_layer(2, 100, dropout_rate=0,
        activation="relu", bias=True, n_out=None)
    ldnn.connect_output(n_out=1, losstype="mse")
    
    cost = ldnn.get_cost(y, l2=self.configuration["lambda"])
    params = self.ldnn.get_params()
    gparams = T.grad(cost, params)
    gparams = clip_norms(gparams, 100)
    updates = sgd(params, gparams, 0.01)

    train_model = function(inputs=[u, y],
        outputs=[cost],
        updates=updates,
        allow_input_downcast=True)
    
    print "Training"
    for i in xrange(10):
        start = time.time()
        losses = [train_model(trainX[j], trainY[j]) for j in xrange(len(trainX))]
        errs = [train_model(trainX[j], trainY[j]) for j in xrange(len(devX))]
        end = time.time()
        print "Epoch: %d Loss: %.6f Err: %.6f" % (i, np.sum(losses) / len(trainX), np.sum(errs) / len(devX))
