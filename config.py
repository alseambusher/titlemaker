DEBUG = False

empty = 0
eos = 1
rnn_size = 512
rnn_layers = 3
batch_norm = False
activation_rnn_size = 40

# training parameters
seed = 42 # answer to everything
p_W, p_U, p_dense, weight_decay = 0, 0, 0, 0
optimizer = 'adam'
batch_size = 64
nflips = 10
nb_unknown_words = 10
LR = 1e-4

embedding_dim = 100 # selects glove file based on this

nb_train_samples = 300
nb_val_samples = 37


num_iterations = 500
num_epochs = 5

glove_dir = "glove.6B"
glove_name = 'glove.6B.%dd.txt' % embedding_dim
glove_n_symbols = 400000
