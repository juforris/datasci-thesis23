# Import modules
import pickle
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import backend as K
tf.random.set_seed(1337)  # sets seeds for base-python, numpy and tf

# Validation set
maxlen = 200
X_valid10bm = pd.read_pickle("../data10bm/X_valid.pkl")
X_valid = pad_sequences(X_valid10bm, padding="post", maxlen=maxlen)

# Hybrid model trained on four data subsets
models = []
models.append(load_model("Hybrid10bm.h5"))
models.append(load_model("Hybrid25bm.h5"))
models.append(load_model("Hybrid50bm.h5"))
models.append(load_model("Hybrid100bm.h5"))

# First conv layer specs
nfilters = 224
kernel = 9
dims = 4 

# Extract motifs from first convolution layer of each model
for model, name in zip(models, ["10bm", "25bm", "50bm", "100bm"]):
    # First convolution layer
    conv_layer = model.layers[0]
    conv_input = conv_layer.input
    conv_output = conv_layer.get_output_at(0)
    # Function to find max activations and corresponding position index
    f = K.function([conv_input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])
    # Create motifs
    motifs = np.zeros((nfilters, kernel, dims))
    nsites = np.zeros(nfilters)
    # Batches of 100 from X_valid
    for i in range(0, len(X_valid), 100):
        x = X_valid[i:i+100]
        # Pass batch through conv layer
        max_inds, max_acts = f([x])
        for m in range(nfilters):
            for n in range(len(x)):
                if max_acts[n, m] > 0:
                    nsites[m] += 1
                    motif = x[n, max_inds[n, m]:max_inds[n, m] + kernel, :]
                    motifs[m] += motif
                    
    motifs_file = open("HybridConvMotifs/motifs"+name+".txt", "w")
    motifs_file.write("MEME version 5.5.1\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")
    for m in range(nfilters):
        if nsites[m] == 0:
            continue
        motifs_file.write("MOTIF M%i O%i\n" % (m, m))
        motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i E= 1337.0e-6\n" % (kernel, nsites[m]))
        for j in range(kernel):
            motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:dims] / np.sum(motifs[m, j, 0:dims])))
        motifs_file.write("\n")
    motifs_file.close()
