import keras
import cPickle as pickle
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
import h5py
import sys
import Levenshtein
from config import *
from beam import beamsearch_p, vocab_fold, vocab_unfold
from rnn import rnn

maxlend = 50
maxlenh = 25
maxlen = maxlend + maxlenh

with open('embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)

vocab_size, embedding_size = embedding.shape

with open('data.pkl', 'rb') as fp:
    X, Y = pickle.load(fp)

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] += '^'

idx2word[empty] = '_'
idx2word[eos] = '~'


model, weights = rnn(vocab_size, embedding_size, maxlen, embedding, maxlend, maxlenh, load_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam')
n = 2*(rnn_size - activation_rnn_size)

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

def gensamples(X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size,
               short=True, temperature=1., use_unk=True):
    if X is None or isinstance(X, int):
        if X is None:
            i = random.randint(0, len(X_test) - 1)
        else:
            i = X
        print 'HEAD %d:' % i, ' '.join(idx2word[w] for w in Y_test[i])
        print 'DESC:', ' '.join(idx2word[w] for w in X_test[i])
        sys.stdout.flush()
        x = X_test[i]
    else:
        x = [word2idx[w.rstrip('^')] for w in X.split()]

    if avoid:
        if isinstance(avoid, str) or isinstance(avoid[0], int):
            avoid = [avoid]
        avoid = [a.split() if isinstance(a, str) else a for a in avoid]
        avoid = [vocab_fold([w if isinstance(w, int) else word2idx[w] for w in a], vocab_size, glove_idx2idx)
                 for a in avoid]

    print 'HEADS:'
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend, len(x)), max(maxlend, len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start, vocab_size, glove_idx2idx)
        sample, score = beamsearch_p(start=fold_start, maxsample=maxlen, avoid=avoid, avoid_score=avoid_score,
                                     k=k, temperature=temperature, use_unk=use_unk, model=model, maxlen=maxlen, maxlend=maxlend,
                                     sequence=sequence, vocab_size=vocab_size, weights=weights)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s, start, scr) for s, scr in zip(sample, score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample, vocab_size-nb_unknown_words)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
            if distance > -0.6:
                print score, ' '.join(words)
        else:
            print score, ' '.join(words)
        codes.append(code)
    return samples

seed = 8
random.seed(seed)
np.random.seed(seed)

for i in xrange(len(X)):
    x = " ".join(map(lambda a: idx2word[a], X[i])[:25])
    y = " ".join(map(lambda a: idx2word[a], Y[i]))
    print y
    samples = gensamples(X=x, skips=2, batch_size=batch_size, k=10, temperature=1.)
    print ""
