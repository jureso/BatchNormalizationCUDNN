import theano
import theano.tensor as T
import lasagne
import numpy as np

def simple_network(input_var, use_cudnn = True):
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from batchnormalization import BatchNormLayer

    l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    l_conv1 = ConvLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                                       pad='same',
                                       W=lasagne.init.HeNormal(gain='relu'), flip_filters=False, b=None)

    l_bn1 = BatchNormLayer(l_conv1, cudnn=use_cudnn)
    l_relu = lasagne.layers.NonlinearityLayer(l_bn1, nonlinearity=lasagne.nonlinearities.rectify)
    return l_relu

def test_batchnormalization_forward():
    input_var = T.tensor4('inputs')

    network = simple_network(input_var)

    output = lasagne.layers.get_output(network)

    fn = theano.function([input_var], output)

    Xin = np.random.randn(100,3,32,32).astype(np.float32)
    Xout = fn(Xin)
    assert Xout.shape == (100,16,32,32)


def test_batchnormalization_backward():
    input_var = T.tensor4('inputs')

    network = simple_network(input_var)

    output = lasagne.layers.get_output(network)

    loss = T.sum(output**2)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, 1)

    fn = theano.function([input_var], output, updates=updates)

    Xin = np.random.randn(100,3,32,32).astype(np.float32)
    Xout = fn(Xin)
    assert Xout.shape == (100,16,32,32)
