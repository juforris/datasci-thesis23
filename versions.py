import tensorflow as tf
import keras_tuner
import matplotlib
import numpy
import pandas
import sklearn
import scipy

# Check GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("\nModule versions")
print("Tensorflow", tf.__version__)
print("KerasTuner", keras_tuner.__version__)
print("Matplotlib", matplotlib.__version__)
print("NumPy", numpy.__version__)
print("Pandas", pandas.__version__)
print("Scikit-learn",  sklearn.__version__)
print("SciPy", scipy.__version__)

