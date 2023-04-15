# Data wrangling and plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.config import list_physical_devices

# Deep Learning
import tensorflow as tf
tf.random.set_seed(1337)  # sets seeds for base-python, numpy and tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import InputLayer, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Additional metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Denisty function
from scipy.stats import gaussian_kde

# Check GPU
physical_devices = list_physical_devices('GPU')
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

# Build hybrid model
model = Sequential()
# CNN part
model.add(InputLayer(input_shape=(maxlen, 4), name='Input_Layer'))
model.add(Conv1D(filters=224, kernel_size=9, padding="valid", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# Bidirectional LSTM part
forward_lstm = LSTM(25, return_sequences=True)
backward_lstm = LSTM(25, return_sequences=True, go_backwards=True)
brnn = Bidirectional(layer=forward_lstm, backward_layer=backward_lstm)
model.add(brnn)
model.add(Dropout(0.3))
# Fully connected part
model.add(Flatten())
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=["mae"])

# Print summary
model.summary()

# Train model
epochs=100
batch_size=128
stop_early = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[stop_early])

# Save model
model.save("Hybrid.h5")

# Save history
with open('Hybrid_history.pkl', 'wb') as f:    ### NAME
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
plt.savefig("Hybrid_loss.pdf", bbox_inches="tight")      ### NAME

# Predict
y_pred = model.predict(X_valid, batch_size=batch_size)
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
plt.savefig("Hybrid_corr.pdf", bbox_inches="tight")        ### NAME
