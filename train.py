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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys

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

def three_vs_seven(args={}):
    with gzip.open("mnist.pkl.gz") as f:
        dat = pickle.load(f)
    train_data, valid_data, test_data = dat
    X_train, y_train = train_data
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28) )
    X_valid, y_valid = valid_data
    X_valid = X_valid.reshape((X_valid.shape[0], 1, 28, 28) )
    X_train_three = X_train[ y_train == 3 ]
    X_train_seven = X_train[ y_train == 7 ]
    X_valid_three = X_valid[ y_valid == 3 ]
    X_valid_seven = X_valid[ y_valid == 7 ]
    return X_train_three, X_train_seven, X_valid_three, X_valid_seven


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


def modest_net_twoconv(args={"nonlinearity":tanh}):
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv = l_in
    for i in range(0, 13):
        l_conv = Conv2DLayer(l_conv, num_filters=(i+1)*8, filter_size=3, nonlinearity=args["nonlinearity"])
    #l_conv = ReshapeLayer( DenseLayer(l_conv, num_units=1), (-1, 1, 1, 1))
    for layer in get_all_layers(l_conv):
        print layer, layer.output_shape
    print count_params(layer)
    return l_conv

def too_simple_net(args={}):
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv = Conv2DLayer(l_in, num_filters=8, filter_size=3)
    l_conv = MaxPool2DLayer(l_conv, pool_size=2)
    l_conv = Conv2DLayer(l_conv, num_filters=16, filter_size=3)
    l_conv = MaxPool2DLayer(l_conv, pool_size=2)
    return l_conv

# ------------------------------------


def too_simple_net_ae(args={}):
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv = Conv2DLayer(l_in, num_filters=32, filter_size=3, nonlinearity=softplus)
    l_conv = Pool2DLayer(l_conv, pool_size=2, mode='average_inc_pad')
    l_conv = Conv2DLayer(l_conv, num_filters=48, filter_size=3, nonlinearity=softplus)
    l_conv = Pool2DLayer(l_conv, pool_size=2, mode='average_inc_pad')
    l_conv = Conv2DLayer(l_conv, num_filters=64, filter_size=3, nonlinearity=softplus)
    l_conv = DenseLayer(l_conv, num_units=256, nonlinearity=softplus)
    for layer in get_all_layers(l_conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_conv = InverseLayer(l_conv, layer)

    for layer in get_all_layers(l_conv):
        print layer, layer.output_shape
    print count_params(layer)

    l_out = l_conv
    
    return l_out


def author_net(args={}):
    l_in = InputLayer((None, 1, 28, 28))
    l_conv = Conv2DLayer(l_in, num_filters=128, filter_size=7, nonlinearity=softplus)
    l_conv = Pool2DLayer(l_conv, 2, mode='average_inc_pad')
    l_conv = DenseLayer(l_conv, args["h"], nonlinearity=softplus)
    for layer in get_all_layers(l_conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_conv = InverseLayer(l_conv, layer)

    for layer in get_all_layers(l_conv):
        print layer, layer.output_shape
    print count_params(layer)

    l_out = l_conv
    
    return l_out








# -----------------------------------


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(123)

def get_net(net_cfg, args={"lambda":0.5}):
    l_out = net_cfg(args)

    X = T.tensor4('X')
    X_noise = X + srng.normal(X.shape, std=1.)
    b_prime = theano.shared( np.zeros( (1, 28, 28) ) )
    net_out = get_output(l_out, X)
    net_out_noise = get_output(l_out, X_noise)
    energy = args["lambda"]*((X-b_prime)**2).sum() - net_out.sum()
    energy_noise = args["lambda"]*((X_noise-b_prime)**2).sum() - net_out_noise.sum()
    # reconstruction
    fx = X - T.grad(energy, X)
    fx_noise = X_noise - T.grad(energy_noise, X_noise)
    loss = ((X-fx_noise)**2).sum(axis=[1,2,3]).mean()

    
    
    params = get_all_params(l_out, trainable=True)
    params += [b_prime]
    lr = theano.shared(floatX(args["learning_rate"]))
    #updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X], [loss,energy], updates=updates)
    energy_fn = theano.function([X], energy)
    out_fn = theano.function([X], fx)
    
    return {
        "train_fn": train_fn,
        "energy_fn": energy_fn,
        "out_fn": out_fn,
        "lr": lr,
        "b_prime": b_prime,
        "l_out": l_out
    }

def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1

def train(cfg, data, num_epochs, out_file, sched={}, batch_size=32):
    train_fn = cfg["train_fn"]
    energy_fn = cfg["energy_fn"]
    b_prime = cfg["b_prime"]
    out_fn = cfg["out_fn"]
    X_train_nothing, X_train_anomaly, X_valid_nothing, X_valid_anomaly = data
    idxs = [x for x in range(0, X_train_nothing.shape[0])]
    #train_losses = []
    lr = cfg["lr"]
    with open(out_file, "wb") as f:
        f.write("epoch,loss,avg_base_energy,avg_anom_energy\n")
        for epoch in range(0, num_epochs):
            if epoch+1 in sched:
                lr.set_value( floatX(sched[epoch+1]) )
                sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            np.random.shuffle(idxs)
            X_train_nothing = X_train_nothing[idxs]

            losses = []
            energies = []

            for X_batch in iterate(X_train_nothing, bs=batch_size):
                this_loss, this_energy = train_fn(X_batch)
                losses.append(this_loss)
                energies.append(this_energy)

            anomaly_energies = []
            for X_batch in iterate(X_train_anomaly, bs=batch_size):
                this_energy = energy_fn(X_batch)
                anomaly_energies.append(this_energy)

            #valid_losses = []
            #for X_batch in iterate(X_valid_nothing, bs=batch_size):
            #    this_loss, _ = 

            # plot one image from the validation set
            img_orig = X_valid_nothing[0:1]
            img_reconstruct = out_fn(img_orig)[0][0]
            plt.subplot(121)
            plt.imshow(img_orig[0][0], cmap="gray" )
            plt.subplot(122)
            plt.imshow(img_reconstruct, cmap="gray" )
            plt.savefig('%s/%i.png' % (os.path.dirname(out_file), epoch))
                
            print epoch+1, np.mean(losses), np.mean(energies), np.mean(anomaly_energies)
            f.write("%i,%f,%f,%f\n" % (epoch+1, np.mean(losses), np.mean(energies), np.mean(anomaly_energies)))

            


            
    with open("%s.model" % out_file, "wb") as f:
        pickle.dump([ get_all_param_values(cfg["l_out"]), b_prime.get_value()], f, pickle.HIGHEST_PROTOCOL)
            
def get_energies(cfg, data, batch_size=1):
    energy_fn = cfg["energy_fn"]
    tot = []
    for dataset in data:
        energies = []
        for X_single in iterate(dataset, bs=batch_size):
            energies.append( float(energy_fn(X_single)) )
        tot.append( energies )
    return tot

def save_array(arr, filename, header):
    with open(filename,"wb") as f:
        f.write("%s\n" % header)
        for elem in arr:
            f.write("%f\n" % elem)
        
if __name__ == "__main__":
    
    # ------------
    # EXPERIMENT 2
    # ------------
    data = three_vs_seven()
    X_train_three, X_train_seven, X_valid_three, X_valid_seven = data
    prefix = "three_vs_seven/author_net_512h.txt"
    lamb=0.5
    cfg = get_net(author_net, {"learning_rate": 0.1, "lambda":lamb, "h":512})
    train(cfg, data, num_epochs=1000, out_file=prefix, batch_size=128)
    #collect the energies
    cfg = get_net(author_net, {"learning_rate": 0.1, "lambda":lamb, "h":512})
    with open("%s.model" % prefix) as g:
        model = pickle.load(g)
        set_all_param_values(cfg["l_out"], model[0])
        cfg["b_prime"].set_value(model[1])
    three_train, seven_train, three_valid, seven_valid = get_energies(cfg, data, batch_size=1)
    save_array(three_train, "%s.a.csv" % prefix, "three_train")
    save_array(seven_train, "%s.b.csv" % prefix, "seven_train")
    save_array(three_valid, "%s.c.csv" % prefix, "three_valid")
    save_array(seven_valid, "%s.d.csv" % prefix, "seven_valid")
