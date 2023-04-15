# Import modules
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle

# Load validation sequences
X_valid = pd.read_pickle("../data/X_valid.pkl")

# Input length
maxlen = 200

# Model custom object
def Spearman(y_true, y_pred):
     return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32))

# Load model
model = load_model("CNN.h5", custom_objects={"Spearman": Spearman})

# Preprocessing
data = X_valid
one_hot = pad_sequences(data, padding="post", maxlen=maxlen)


### DeepExplainer functions
# Source: https://github.com/bernardo-de-almeida/DeepSTARR/blob/main/DeepSTARR/DeepSTARR_nucl_contr_scores.py
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    dinuc_shuffle_n=100
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(dinuc_shuffle_n)])
    return [to_return] #wrap in list for compatibility with multiple modes

def my_deepExplainer(model, one_hot):
    explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), data=dinuc_shuffle_several_times)
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    # running on all sequences
    shap_values_hypothetical = explainer.shap_values(one_hot, check_additivity=False)
    # normalising contribution scores
    # sum the deeplift importance scores across the ACGT axis (different nucleotides at the same position)
    # and “project” that summed importance onto whichever base is actually present at that position
    shap_values_contribution = shap_values_hypothetical[0]*one_hot
    return shap_values_hypothetical[0], shap_values_contribution

# Run DeepExplainer
scores = my_deepExplainer(model, one_hot)
shap_values_hypothetical, shap_values_contribution = scores

# Save scores
np.save("modisco/contribution_scores.npy", shap_values_contribution)
np.save("modisco/hypothetical_scores.npy", shap_values_hypothetical)
np.save("modisco/10bm_valid_one_hot.npy", one_hot)