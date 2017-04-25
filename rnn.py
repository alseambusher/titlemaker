import h5py
from keras.layers import Embedding
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.regularizers import l2
from config import *
import numpy as np
import keras.backend as K
import random

regularizer = l2(weight_decay) if weight_decay else None

def rnn(vocab_size, embedding_size, maxlen, embedding, maxlend, maxlenh, load_weights=False):
    random.seed(seed)
    np.random.seed(seed)

    rnn_model = Sequential()
    rnn_model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        embeddings_regularizer=regularizer, weights=[embedding], mask_zero=True,
                        name='embedding_1'))

    if load_weights:
        rnn_model.load_weights('model.hdf5', by_name=True)

        with h5py.File('model.hdf5', mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            weights = [np.copy(v) for v in f['time_distributed_1'].values()[0].itervalues()]
            weights.reverse()

    def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
        desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
        head_activations, head_words = head[:, :, :n], head[:, :, n:]
        desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

        # http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2], [2]))
        activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :maxlend], 'float32'), 1)

        activation_energies = K.reshape(activation_energies, (-1, maxlend))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=([2], [1]))
        return K.concatenate((desc_avg_word, head_words))

    if activation_rnn_size:
        rnn_model.add(Lambda(simple_context,
                     mask = lambda inputs, mask: mask[:,maxlend:],
                     output_shape = lambda input_shape: (input_shape[0], maxlenh, 2*(rnn_size - activation_rnn_size)),
                     name='simplecontext_1'))

    if load_weights:
        return rnn_model, weights

    return rnn_model
