###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser

import tensorflow as tf
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model as plot
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from loader import load_trainset, load_images_labels
from unet import get_unet

session = K.get_session()

def weighted_cross_entropy(weight):
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight)
    return loss

#========= Load settings from Config file =====================================
config = configparser.RawConfigParser()
config.read('configuration.txt')

#patch to the datasets
path_data = config.get('data paths', 'path_local')
train_path = config.get('data paths', 'train_data_path')

#Experiment
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
experiment_path = path_data + '/' + name_experiment + '_' + arch

u_net = arch == 'unet'

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
N_subimgs = int(config.get('training settings', 'N_subimgs'))
patch_size = (
    int(config.get('data attributes', 'patch_height')),
    int(config.get('data attributes', 'patch_width'))
)


#========= Save a sample of what you're feeding to the neural network ==========
patches_imgs_samples, patches_gts_samples = load_images_labels(
    train_path,
    batch_size,
    N_subimgs
)
patches_imgs_samples = patches_imgs_samples[0:20] * 85. + 127.5
patches_gts_samples = tf.cast(patches_gts_samples[0:20] * 255., tf.float32)
patches_gts_samples = tf.reshape(
    patches_gts_samples,
    (20, 1, patch_size[0], patch_size[1])
)

imgs_samples = session.run(
    tf.concat([patches_imgs_samples, patches_gts_samples], 0)
)
visualize(group_images(imgs_samples, 5), experiment_path + '/' + "sample_input")

#============ Load the data and normalize =======================================
test_dataset, train_dataset = load_trainset(
    train_path,
    batch_size,
    N_subimgs
)
# _, labels = train_dataset.make_one_shot_iterator().get_next()

# gt_mean = tf.math.reduce_mean(labels)
# gt_mean = session.run(gt_mean)
# LOSS_WEIGHT = (1. - gt_mean) / gt_mean

#=========== Construct and save the model arcitecture ===========================
if u_net:
    model = get_unet(1, batch_size, patch_size[0], patch_size[1])  #the U-net model
else:
    model = get_resnet()

model.compile(
    optimizer = 'adam',
    # loss = weighted_cross_entropy(LOSS_WEIGHT),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print("Check: final output of the network:")
print(model.output_shape)

plot(model, to_file = experiment_path + '/' + name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open(experiment_path + '/' + name_experiment +'_architecture.json', 'w').write(json_string)

#============  Training ========================================================
checkpointer = ModelCheckpoint(
    filepath = experiment_path + '/' + name_experiment +'_best_weights.h5',
    verbose = 1,
    monitor = 'val_loss',
    mode = 'auto', 
    save_best_only = True) #save at each epoch if the validation decreased

tensorboard = TensorBoard(
    log_dir = experiment_path + '/logs/{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')),
    #write_images = True,
    batch_size = batch_size,
    histogram_freq = 5,
)

# train_x, train_y = train_dataset.make_one_shot_iterator().get_next()
# size = int(batch_size * 30)
# train_x = tf.tile(tf.reshape(train_x[0], [1, 1, 48, 48]), [size, 1, 1, 1])
# train_y = tf.tile(tf.reshape(train_y[0], [1, 1, 2304]), [size, 1, 1])

# train_x, train_y = session.run([train_x, train_y])

# model.fit(
#     train_x,
#     train_y,
#     epochs = N_epochs,
#     batch_size = batch_size,
#     verbose = 2,
#     validation_split= 0.1,
#     callbacks = [checkpointer, tensorboard])


model.fit(
    train_dataset,
    epochs = N_epochs,
    steps_per_epoch = int(N_subimgs / batch_size),
    validation_data = test_dataset,
    validation_steps = int(10),
    verbose = 2,
    callbacks = [checkpointer, tensorboard])

#========== Save and test the last model ==================================
model.save_weights(experiment_path + '/' + name_experiment +'_last_weights.h5', overwrite=True)
