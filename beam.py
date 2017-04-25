import numpy as np
from config import *
from util import output2probs
import keras.backend as K

# mod https://github.com/ryankiros/skip-thoughts/blob/master/decoding

def beamsearch_p(start, maxsample, vocab_size, model, maxlen, maxlend, sequence, weights=None, avoid=None, avoid_score=1,
               k=1, use_unk=True, oov=None, empty=empty, eos=eos, temperature=1.0):
    if not oov:
        oov = vocab_size - 1

    def sample(energy, n, temperature=temperature):
        """sample at most n different elements according to their energy"""
        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in xrange(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.
        return res

    dead_samples = []
    dead_scores = []
    live_samples = [list(start)]
    live_scores = [0]

    while live_samples:
        probs = keras_rnn_predict(live_samples, empty, model, maxlen, maxlend, sequence, weights)
        assert vocab_size == probs.shape[1]

        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_scores[:, empty] = 1e20
        if not use_unk and oov is not None:
            cand_scores[:, oov] = 1e20
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        cand_scores[i, a[n]] += avoid_score
        live_scores = list(cand_scores.flatten())

        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        dead_scores = [dead_scores[r] for r in ranks if r < n]
        dead_samples = [dead_samples[r] for r in ranks if r < n]

        live_scores = [live_scores[r - n] for r in ranks if r >= n]
        live_samples = [live_samples[(r - n) // vocab_size] + [(r - n) % vocab_size] for r in ranks if r >= n]

        def is_zombie(s):
            return s[-1] == eos or len(s) > maxsample

        dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
        dead_samples += [s for s in live_samples if is_zombie(s)]

        live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
        live_samples = [s for s in live_samples if not is_zombie(s)]

    return dead_samples, dead_scores

def beamsearch_t(start, vocab_size, maxsample, model, maxlen, maxlend, sequence, weights,
               k=1, use_unk=True, empty=empty, eos=eos, temperature=1.0):

    def sample(energy, n, temperature=temperature):
        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in xrange(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.
        return res

    dead_samples = []
    dead_scores = []
    live_k = 1
    live_samples = [list(start)]
    live_scores = [0]

    while live_k:
        probs = keras_rnn_predict(live_samples, empty, model, maxlen, maxlend, sequence, weights)

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_scores[:, empty] = 1e20
        if not use_unk:
            for i in range(nb_unknown_words):
                cand_scores[:, vocab_size - 1 - i] = 1e20
        live_scores = list(cand_scores.flatten())

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        ranks_dead = [r for r in ranks if r < n]
        ranks_live = [r - n for r in ranks if r >= n]

        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]

        live_scores = [live_scores[r] for r in ranks_live]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_live]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s, z in zip(live_samples, zombie) if z]
        dead_scores += [s for s, z in zip(live_scores, zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living
        live_samples = [s for s, z in zip(live_samples, zombie) if not z]
        live_scores = [s for s, z in zip(live_scores, zombie) if not z]
        live_k = len(live_samples)

    return dead_samples + live_samples, dead_scores + live_scores

def keras_rnn_predict(samples, empty, model, maxlen, maxlend, sequence, weights=None):
    sample_lengths = map(len, samples)
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    if not weights:
        return np.array([prob[sample_length - maxlend - 1] for prob, sample_length in zip(probs, sample_lengths)])

    return np.array([output2probs(prob[sample_length-maxlend-1], weights) for prob, sample_length in zip(probs, sample_lengths)])

def vocab_fold(xs, vocab_size, glove_idx2idx):
    xs = [x if x < vocab_size-nb_unknown_words else glove_idx2idx.get(x,x) for x in xs]
    outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def vocab_unfold(desc, xs, oov0):
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]
