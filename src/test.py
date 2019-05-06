import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np

import sys
sys.path.insert(0, './lib/')
from help_functions import *

# data, labels = load_tfrecord('./DRIVE_datasets/dataset__train*.tfrecord', (48, 48), 32, 216320)
data = np.random.random((1000, 1, 48, 48))
labels = np.random.random((1000, 1, 48, 48))

inputs = tf.keras.Input(shape=(1,48,48))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
x = layers.Conv2D(1, (1, 1), activation='relu',padding='same', data_format='channels_first')(x)

model = Model(inputs=inputs, outputs=x)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
