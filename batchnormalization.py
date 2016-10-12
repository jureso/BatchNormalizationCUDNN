import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
from lasagne import init
from theano.sandbox.cuda.dnn import dnn_batch_normalization_train, dnn_batch_normalization_test

class BatchNormLayer(Layer):
    """
    batchnormalization.BatchNormLayer(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
    beta=init.Constant(0), gamma=init.Constant(1), mean=init.Constant(0), inv_std=init.Constant(1),
    var=init.Constant(1), cudnn = True, beta_trainable=True, gamma_trainable=True, **kwargs)

    Batch Normalization implementation using CUDNN

    This layer implements batch normalization using CUDNNv5 implementation. of its inputs, following [1]_:

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
       The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
       The axis or axes to normalize over. If ``'auto'`` (the default),
       normalize over all axes except for the second: this will normalize over
       the minibatch dimension for dense layers, and additionally over all
       spatial dimensions for convolutional layers.
    epsilon : scalar
       Small constant :math:`\\epsilon` added to the variance before taking
       the square root and dividing by it, to avoid numerical problems
    alpha : scalar
       Coefficient for the exponential moving average of batch-wise means and
       standard deviations computed during training; the closer to one, the
       more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
       Initial value, expression or initializer for :math:`\\beta`. Must match
       the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
       it to 0.0 instead of learning it.
       See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
       Initial value, expression or initializer for :math:`\\gamma`. Must
       match the incoming shape, skipping all axes in `axes`. Set to ``None``
       to fix it to 1.0 instead of learning it.
       See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
       Initial value, expression or initializer for :math:`\\mu`. Must match
       the incoming shape, skipping all axes in `axes`.
       See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
       Initial value, expression or initializer for :math:`1 / \\sqrt{
       \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
       axes in `axes`.
       See :func:`lasagne.utils.create_param` for more information.
    var : Theano shared variable, expression, numpy array, or callable
       Initial value, expression or initializer for :math:`\\sigma^2 `.
       Must match the incoming shape, skipping all axes in `axes`.
    cudnn : boolean
       Determines if we us CUDNN implementation or internal theano implementation
    beta_trainable : boolean
       Determines if parameter beta is fixed during training or if it is updated.
    gamma_trainable :  boolean
       Determines if parameter gamma is fixed during training or if it is updated.
    **kwargs
       Any additional keyword arguments are passed to the :class:`Layer`
       superclass.
    Notes
    -----
    In large graphs (e.g. ResNets) the compilation using cudnn=True is significantly faster than using cudnn=False.

    See also
    --------
    lasagne.layers.BatchNormLayer
    """

    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), var=init.Constant(1),
                 cudnn = True, beta_trainable=True, gamma_trainable=True, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        assert len(self.input_shape) == 4, "Current implementation only supports 4D tensors! Input shape: " + str(self.input_shape)
        assert axes == 'auto', "Current implementation only support axes='auto'!"

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        self.cudnn = cudnn

        # we manually determine the shape of all parameters (cudnn requires such shapes)
        shape = (1,self.input_shape[1],1,1)

        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=beta_trainable, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=gamma_trainable, regularizable=gamma_trainable)

        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)
        self.var = self.add_param(var, shape, 'var',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):


        if deterministic is False:
            if self.cudnn:
                out, input_mean, input_inv_std = dnn_batch_normalization_train(input, self.gamma, self.beta, mode='spatial',epsilon=self.epsilon)
            else: # we simulate cudnn BN
                axes = self.axes

                input_mean = input.mean(axes, keepdims=True)
                input_var = input.var(axes, keepdims=True)
                input_inv_std = T.inv(T.sqrt(input_var + self.epsilon))
                out = (input - input_mean) * self.gamma * input_inv_std + self.beta

            var = input_inv_std ** (-2) - self.epsilon

            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)

            running_mean.default_update = ((1 - self.alpha) * running_mean + self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) * running_inv_std + self.alpha * input_inv_std)
            running_var.default_update = ((1 - self.alpha) * running_var + self.alpha * var)

            out += (0 * running_mean + 0 * running_inv_std + 0 * running_var)

        else:
            if self.cudnn:
                out = dnn_batch_normalization_test(input, self.gamma, self.beta, self.mean, self.var, mode='spatial', epsilon=self.epsilon)
            else:
                out = (input - self.mean) * self.gamma * T.sqrt((self.var + self.epsilon))**(-1) + self.beta
        return out
