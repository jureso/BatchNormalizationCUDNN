# Batch Normalization using CUDNN in Lasagne

This module reimplements Lasagne version of 
[Batch normalization](http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html#lasagne.layers.BatchNormLayer) 
so that it uses the CUDNNv5 implementation.
 
See **batchnormalization.py** for details.

## Usage

```python
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from batchnormalization import BatchNormLayer

l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
l_conv1 = ConvLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                                   pad='same',
                                   W=lasagne.init.HeNormal(gain='relu'), flip_filters=False, b=None)

l_bn1 = BatchNormLayer(l_conv1)
l_relu = lasagne.layers.NonlinearityLayer(l_bn1, nonlinearity=lasagne.nonlinearities.rectify)
```
