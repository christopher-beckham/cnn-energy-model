import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
import numpy as np
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt

# --------------

def nonzero_vs_zero(args={}):
    with gzip.open("mnist.pkl.gz") as f:
        dat = pickle.load(f)
    train_data, valid_data, test_data = dat
    X_train, y_train = train_data
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28) )
    X_valid, y_valid = valid_data
    X_valid = X_valid.reshape((X_valid.shape[0], 1, 28, 28) )

    X_train_zeros = X_train[ y_train == 0 ]
    X_train_notzeros = X_train[ y_train != 0 ]
    X_valid_zeros = X_valid[ y_valid == 0 ]
    X_valid_notzeros = X_valid[ y_valid != 0]

    return X_train_notzeros, X_train_zeros, X_valid_notzeros, X_valid_zeros

def modest_net(args={}):
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv = l_in
    for i in range(0, 3):
        l_conv = Conv2DLayer(l_conv, num_filters=(i+1)*32, filter_size=3, nonlinearity=tanh)
        if i != 2:
            l_conv = MaxPool2DLayer(l_conv, pool_size=2)
    l_conv = Conv2DLayer(l_conv, num_filters=3*32, filter_size=3, nonlinearity=tanh)
    #l_conv = DenseLayer(l_conv, num_units=128, nonlinearity=tanh)
    for layer in get_all_layers(l_conv):
        print layer, layer.output_shape
    print count_params(layer)
    return l_conv

def get_net(net_cfg, args={}):
    l_out = net_cfg(args)
    X = T.tensor4('X')
    b_prime = theano.shared( np.ones( (1, 28, 28) ) )
    net_out = get_output(l_out, X)
    energy = 0.5*((b_prime - X)**2).sum(axis=[1,2,3]).mean() - net_out.sum(axis=[1,2,3]).mean()
    loss = ((-T.grad(energy, X))**2).sum(axis=[1,2,3]).mean()
    params = get_all_params(l_out, trainable=True)
    params += [b_prime]
    lr = theano.shared(floatX(args["learning_rate"]))
    updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    #updates = rmsprop(loss, params, learning_rate=0.01)
    train_fn = theano.function([X], loss, updates=updates)
    energy_fn = theano.function([X], energy)
    return {"train_fn": train_fn, "energy_fn": energy_fn, "lr": lr, "b_prime": b_prime}

def iterate(X_train, bs):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1

def train(cfg, data, num_epochs, out_file, sched={}):
    train_fn, energy_fn = cfg["train_fn"], cfg["energy_fn"]
    X_train_nothing, X_train_anomaly, X_valid_nothing, X_valid_anomaly = data
    idxs = [x for x in range(0, X_train_nothing.shape[0])]
    #train_losses = []
    with open(out_file, "wb") as f:
        for epoch in range(0, num_epochs):
            if epoch+1 in sched:
                lr.set_value( floatX(sched[epoch+1]) )
                sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            np.random.shuffle(idxs)
            X_train_nothing = X_train_nothing[idxs]
            losses = []
            for X_batch in iterate(X_train_nothing, bs=32):
                losses.append( train_fn(X_batch) )
            print epoch+1, np.mean(losses)
            #train_losses.append(np.mean(losses))
            f.write("%i,%f\n" % (epoch+1, np.mean(losses)))

if __name__ == "__main__":
    data = nonzero_vs_zero()
    #print X_train_notzeros.shape, X_train_zeros.shape
    cfg = get_net(modest_net, {"learning_rate": 0.01})
    train(cfg, data, num_epochs=100, out_file="output/modest-net_lr0.01_157k.txt")
