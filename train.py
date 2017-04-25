import os
import keras
import cPickle as pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
import random
import numpy as np
from keras.preprocessing import sequence
import sys
import Levenshtein
from keras.utils import np_utils
from keras.layers.core import Lambda
import keras.backend as K
from sklearn.model_selection import train_test_split
from config import *
from util import prt
from beam import *
from rnn import rnn, regularizer

maxlend = 25
maxlenh = 25
maxlen = maxlend + maxlenh

with open('embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)

vocab_size, embedding_size = embedding.shape

with open('data.pkl', 'rb') as fp:
    X, Y = pickle.load(fp)

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

oov0 = vocab_size-nb_unknown_words

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)

idx2word[empty] = '_'
idx2word[eos] = '~'

model = rnn(vocab_size, embedding_size, maxlen, embedding, maxlend, maxlenh)
model.add(TimeDistributed(Dense(vocab_size,
                                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                name = 'timedistributed_1')))
model.add(Activation('softmax', name='activation_1'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
K.set_value(model.optimizer.lr,np.float32(LR))
model.summary()

if False:
    model.load_weights('model.hdf5')

def lpadd(x, maxlend=maxlend, eos=eos):
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

samples = [lpadd([3]*26)]
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
probs = model.predict(data, verbose=0, batch_size=1)


def gensamples(skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    i = random.randint(0,len(X_test)-1)
    print 'HEAD:',' '.join(idx2word[w] for w in Y_test[i][:maxlenh])
    print 'DESC:',' '.join(idx2word[w] for w in X_test[i][:maxlend])
    sys.stdout.flush()

    print 'HEADS:'
    x = X_test[i]
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start, vocab_size, glove_idx2idx)
        sample, score = beamsearch_t(start=fold_start, k=k, temperature=temperature, use_unk=use_unk, maxsample=maxlen, vocab_size=vocab_size,
                                     model=model, maxlen=maxlen, maxlend=maxlend, sequence=sequence)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample, oov0)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                print score, ' '.join(words)
        #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
        else:
                print score, ' '.join(words)
        codes.append(code)

gensamples(skips=2, batch_size=batch_size, k=10, temperature=1.)


def flip_headline(x, nflips=None, model=None, debug=False):
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    assert np.all(x[:, maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        flips = sorted(random.sample(xrange(maxlend + 1, maxlen), nflips))
        for input_idx in flips:
            if x[b, input_idx] == empty or x[b, input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend + 1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            x_out[b, input_idx] = w
    return x_out


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd) + xh) for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty] * maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i, :, :] = np_utils.to_categorical(xh, vocab_size)

    return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxint)
        random.seed(c + 123456789 + seed)
        for b in range(batch_size):
            t = random.randint(0, len(Xd) - 1)

            xd = Xd[t]
            s = random.randint(min(maxlend, len(xd)), max(maxlend, len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh, len(xh)), max(maxlenh, len(xh)))
            xhs.append(xh[:s])

        c += 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)

r = next(gen(X_train, Y_train, batch_size=batch_size))
print r[0].shape, r[1].shape, len(r)

def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',idx2word, yy)
        prt('H',idx2word, y)
        if maxlend:
            prt('D',idx2word, x)

test_gen(gen(X_train, Y_train, batch_size=batch_size))
test_gen(gen(X_train, Y_train, nflips=6, model=model, debug=False, batch_size=batch_size))
valgen = gen(X_test, Y_test,nb_batches=3, batch_size=batch_size)

for i in range(4):
    test_gen(valgen, n=1)

history = {}
traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

r = next(traingen)

for iteration in range(num_iterations):
    print 'Iteration', iteration
    h = model.fit_generator(traingen, steps_per_epoch=nb_train_samples//batch_size,
                        epochs=num_epochs, validation_data=valgen, validation_steps=nb_val_samples
                           )
    for k,v in h.history.iteritems():
        history[k] = history.get(k,[]) + v
    with open('history.pkl','wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('model.hdf5', overwrite=True)
    gensamples(batch_size=batch_size)
