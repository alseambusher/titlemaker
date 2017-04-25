import numpy as np

def prt(label, idx2word, x):
    print label+':',
    for w in x:
        print idx2word[w],
    print ""

# out very own softmax
def output2probs(output, weights):
    output = np.dot(output, weights[0]) + weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output

def output2probs1(output, weights, n):
    output0 = np.dot(output[:n//2], weights[0][:n//2,:])
    output1 = np.dot(output[n//2:], weights[0][n//2:,:])
    output = output0 + output1 # + output0 * output1
    output += weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output