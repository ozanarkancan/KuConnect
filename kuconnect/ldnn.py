from rnn import *
from lstm import *
from gru import *
from layers import *
import cPickle
import theano.tensor as T
from inits import zeros

class LDNN(object):
    """
    Large Deep Neural Network
    """
    
    def __init__(self):
        self.layers = []
        self.net_config = [] #(layer type, size, dropout rate)
        self.output_layer = None
        self.memo = []

    def add_input_layer(self, input, dropout_rate):
        self.input = input
        l = InputLayer(input, dropout_rate)
        self.layers.append(l)
    
    def add_bidirectional_input_layer(self, input, dropout_rate):
        self.input = input
        l = BidirectionalInputLayer(input, dropout_rate)
        self.layers.append(l)

    def add_layer(self, n_in, n_hidden, dropout_rate, activation, bias,
        internal=False, n_out=None, truncate=-1, indices=None):
        self.net_config.append((activation, n_hidden, dropout_rate))
        prev = self.layers[-1]
        if activation == "lstm":
            l = LSTM(prev.output, prev.d_output, n_in, n_hidden,
                dropout_rate=dropout_rate, bias=bias, truncate=truncate)
            self.layers.append(l)
            self.memo += l.memo
        elif activation == "lstm-peephole":
            l = LSTMPeephole(prev.output, prev.d_output, n_in, n_hidden,
                dropout_rate=dropout_rate, bias=bias, truncate=truncate)
            self.layers.append(l)
            self.memo += l.memo
        elif activation == "gru":
            l = GRU(prev.output, prev.d_output, n_in, n_hidden,
            dropout_rate=dropout_rate, bias=bias, truncate=truncate)
            self.layers.append(l)
            self.memo += l.memo
        elif activation == "pool":
            l = PoolingLayer(prev.output, prev.d_output, indices,
                prev.n_out, prev.n_out)
        elif activation == "meanpool":
            l = MeanPoolingLayer(prev.output, prev.d_output, indices,
                prev.n_out, prev.n_out)
	    self.layers.append(l)
        else:
            if "feedback" in activation:
                prms = activation.split("-")
                act = activation.split("-")[0]
                if len(prms) == 2:
                    l = ElmanFeedback(input=prev.output, d_input=prev.d_output, n_in=n_in,
                        n_hidden=n_hidden, n_out=n_out, h0=None, d_h0=None,
                        activation=act, bias=bias, dropout_rate=dropout_rate, truncate=truncate)
                elif len(prms) == 3:
                    l = ElmanFeedback2(input=prev.output, d_input=prev.d_output, n_in=n_in,
                        n_hidden=n_hidden, n_out=n_out, h0=None, d_h0=None,
                        activation=act, bias=bias, dropout_rate=dropout_rate, truncate=truncate)
            elif activation.startswith("bi"):
                act = activation.split("-")[1]
                if act == "lstm":
                    l = BidirectionalLSTM(prev.f_output, prev.f_d_output,
                        prev.b_output, prev.b_d_output, n_in, n_hidden,
                        dropout_rate=dropout_rate, bias=bias, truncate=truncate)
                elif act == "gru":
                    l = BidirectionalGRU(prev.f_output, prev.f_d_output,
                        prev.b_output, prev.b_d_output, n_in, n_hidden,
                        dropout_rate=dropout_rate, bias=bias, truncate=truncate)
                elif act == "meanpool":
                    l = BidirectionalMeanPoolingLayer(prev.f_output,
                        prev.f_d_output, prev.b_output, prev.b_d_output,
                        indices, prev.n_out, prev.n_out)
                elif act == "pool":
                    l = BidirectionalPoolingLayer(prev.f_output,
                        prev.f_d_output, prev.b_output, prev.b_d_output,
                        indices, prev.n_out, prev.n_out)
                else:
                    l = BidirectionalElman(prev.f_output, prev.f_d_output,
                        prev.b_output, prev.b_d_output, n_in=n_in,
                        n_hidden=n_hidden, hf0=None, hb0=None, d_hf0=None,
                        d_hb0=None, activation=act, bias=bias, dropout_rate=dropout_rate, truncate=truncate)
            else:
                l = Elman(prev.output, prev.d_output, n_in, n_hidden,
                    activation=activation, dropout_rate=dropout_rate, bias=bias, truncate=truncate)
            self.layers.append(l)
            if not (l.memo is None):
                self.memo += l.memo

    def connect_output(self, n_out, losstype="softmax", lastone=False,
        compile_predict=True, recurrent=0):
        if "feedback" in self.net_config[-1][0]:
            self.output_layer = self.layers[-1]
        elif "bi" in self.net_config[-1][0]:
            p_l = self.layers[-1]
            
            if recurrent == 1:
                self.output_layer = BidirectionalRecurrentOutputLayer(p_l.f_output,
                    p_l.f_d_output, p_l.b_output, p_l.b_d_output, p_l.n_out, n_out, losstype=losstype)
            elif recurrent == 2:
                self.output_layer = BidirectionalRecurrent2OutputLayer(p_l.f_output,
                    p_l.f_d_output, p_l.b_output, p_l.b_d_output, p_l.n_out, n_out, losstype=losstype)
            else:
            	self.output_layer = BidirectionalOutputLayer(p_l.f_output, p_l.f_d_output, 
                    p_l.b_output, p_l.b_d_output, p_l.n_out, n_out, losstype=losstype)
        else:
            input = self.layers[-1].output[-1] if lastone else self.layers[-1].output
            d_input = self.layers[-1].d_output[-1] if lastone else self.layers[-1].d_output
            if recurrent == 1:
                self.output_layer = RecurrentOutputLayer(input, d_input,
                    self.layers[-1].n_out, n_out, losstype=losstype)
            if recurrent == 2:
                self.output_layer = Recurrent2OutputLayer(input, d_input,
                    self.layers[-1].n_out, n_out, losstype=losstype)
            else:
                self.output_layer = OutputLayer(input, d_input,
                    self.layers[-1].n_out, n_out, losstype=losstype)
        outputs = [self.output_layer.y_pred, self.output_layer.p_y_given_x] if losstype == "softmax" else [self.output_layer.output]
	if compile_predict:
        	self.predict = theano.function(inputs=[self.input],
            		outputs=outputs,
            		updates=self.get_one_prediction_updates(),
            		allow_input_downcast=True)

    def get_one_prediction_updates(self):
        updates = []
        for p in self.memo:
            updates.append((p[0], p[1]))
        return updates

    def reset_memory(self):
        for p in self.memo:
            p[0].set_value(np.zeros_like(p[0].get_value(), dtype=floatX))

    def get_params(self):
        params = []
        for l in self.layers:
            if l.params != None:
                params += l.params
        if not ("feedback" in self.net_config[-1][0]):
            params += self.output_layer.params
        return params

    def get_cost(self, y, l2=0):
        cost = self.output_layer.d_loss(y)
        
        if l2 != 0:
            params = self.get_params()
            for p in params:
                if not p.name.startswith('b_'):
                    cost += l2 * (p ** 2).sum()
        return cost

    def save(self, filename="net.save"):
        f = file(filename, 'wb')
        params = self.get_params()
        for p in params:
            cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, filename="net.save"):
        f = file(filename, "rb")
        params = self.get_params()
        for p in params:
            p.set_value(cPickle.load())
        f.close()
        
