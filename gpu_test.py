import tensorflow as tf
from tensorflow.python.client import device_lib

print("devices: ", [d.name for d in device_lib.list_local_devices()])
print("GPUs:    ", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TF v.:   ", tf.__version__)