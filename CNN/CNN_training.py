# Data wrangling and plotting
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.config import list_physical_devices

# Deep learning
import tensorflow as tf
tf.random.set_seed(1337)  # sets seeds for base-python, numpy and tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, InputLayer, Input
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint

# Additional metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde, spearmanr

# Model custom object
def Spearman(y_true, y_pred):
     return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32))

# Check GPU
physical_devices = list_physical_devices("GPU")
print("\nNum GPUs:", len(physical_devices))

# Load data
X_train = pd.read_pickle("../data/X_train.pkl")
y_train = pd.read_pickle("../data/y_train.pkl")

X_test = pd.read_pickle("../data/X_test.pkl")
y_test = pd.read_pickle("../data/y_test.pkl")

X_valid = pd.read_pickle("../data/X_valid.pkl")
y_valid = pd.read_pickle("../data/y_valid.pkl")

# Preprocessing
maxlen = 200
X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)
X_valid = pad_sequences(X_valid, padding="post", maxlen=maxlen)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_valid = y_valid.to_numpy()

# Model parameters
params = {"batch_size": 128, "epochs": 100, "early_stop": 10, "kernel_size1": 9, "kernel_size2": 7, "kernel_size3": 5, "kernel_size4": 7, "lr": 0.002, "num_filters": 224, "num_filters2": 60, "num_filters3": 60, "num_filters4": 80, "n_conv_layer": 4, "n_add_layer": 2, "dropout_prob": 0.4, "dense_neurons1": 256, "dense_neurons2": 256, "pad":"same"}

# CNN model build function
def build_cnn(params=params):
    lr = params["lr"]
    dropout_prob = params["dropout_prob"]
    n_conv_layer = params["n_conv_layer"]
    n_add_layer = params["n_add_layer"]
    inputs = Input(shape=(maxlen, 4))              
    x = Conv1D(params["num_filters"], kernel_size=params["kernel_size1"], padding="valid", name="Conv1D_1st")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    for i in range(1, n_conv_layer):
        x = Conv1D(params["num_filters"+str(i+1)], kernel_size=params["kernel_size"+str(i+1)], padding=params["pad"], name=str("Conv1D_"+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    # dense layers
    for i in range(0, n_add_layer):
        x = Dense(params["dense_neurons"+str(i+1)], name=str("Dense_"+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout_prob)(x)
    output = (Dense(1, activation="linear", name=str("Dense_out"))(x))
    model = Model([inputs], output)
    model.compile(Adam(learning_rate=lr), loss=["mse"], loss_weights=[1], metrics=[Spearman])
    return model, params

# Training function
def train(selected_model, X_train, y_train, X_test, y_test, params):
    my_history=selected_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=params["batch_size"], epochs=params["epochs"], callbacks=[EarlyStopping(patience=params["early_stop"], monitor="val_loss", restore_best_weights=True), History()])
    return selected_model, my_history

# Build CNN
main_model, main_params = build_cnn()
main_model.summary()

# Train CNN
main_model, history = train(main_model, X_train, y_train, X_test, y_test, main_params)

# Save model
main_model.save("CNN.h5")           

# Save history
with open('CNN_history.pkl', 'wb') as f:    
    pickle.dump(history.history, f)

# Plot history
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss)+1)

plt.figure(figsize=(5, 3))
plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("CNN_loss.pdf", bbox_inches="tight")    

# Predict
y_pred = main_model.predict(X_valid, batch_size=main_params["batch_size"])
print("    MAE = " + str("{0:0.4f}".format(mean_absolute_error(y_valid, y_pred))))
print("    MSE = " + str("{0:0.4f}".format(mean_squared_error(y_valid, y_pred))))

# Correlation
y_pred = y_pred.reshape(y_valid.shape)
correlation = np.corrcoef(y_pred, y_valid)
print(f"Correlation ypred ytest:\n{correlation}")
print(f"\nR-squared:    {r2_score(y_valid, y_pred)}")

# Calculate point density
xy = np.vstack([y_valid,y_pred])
z = gaussian_kde(xy)(xy)

# Sort points by density
idx = z.argsort()
y_valid, y_pred, z = y_valid[idx], y_pred[idx], z[idx]

# Plot correlation
plt.figure(figsize=(5,5))
plt.scatter(y_valid, y_pred, c=z, s=5)
plt.xlabel("Observed Log2 Fold Change", fontsize=15)
plt.ylabel("Predicted Log2 Fold Change", fontsize=15)
plt.title(f"Pearson Correlation Coefficient = {correlation[0][1]:.2f}")
plt.axis("equal")
plt.savefig("CNN_corr.pdf", bbox_inches="tight")

