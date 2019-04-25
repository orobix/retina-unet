
import configparser
import os

borderMasks_test = '_borderMasks_test.hdf5'
borderMasks_train = '_borderMasks_train.hdf5'
imgs_test = '_imgs_test.hdf5'
imgs_train = '_imgs_train.hdf5'
gt_test = '_groundTruth_test.hdf5'
gt_train = '_groundTruth_train.hdf5'

DRIVE_subimgs = 20
DRIVE_testimgs = 20

Synth_subimgs = 190000
Synth_testimgs = 1000

# first is train second test
settings = [['DRIVE', 'DRIVE'], ['DRIVE', 'Synth'], ['Synth', 'DRIVE'], ['Synth', 'Synth']]


for setup in settings:
  config = configparser.RawConfigParser()
  config.read('./configuration_template.txt')
  ### write config
  experiment = setup[0] + '_' + setup[1] + '_experiment'
  config.set('experiment name', 'name', experiment)

  #config.set('data paths', 'path_local', experiment)
  config.set('data paths', 'train_imgs_original', setup[0] + '_datasets/dataset' + imgs_train)
  config.set('data paths', 'train_groundTruth', setup[0] + '_datasets/dataset' + gt_train)
  config.set('data paths', 'train_border_masks', setup[0] + '_datasets/dataset' + borderMasks_train)
  config.set('data paths', 'test_imgs_original', setup[1] + '_datasets/dataset' + imgs_test)
  config.set('data paths', 'test_groundTruth', setup[1] + '_datasets/dataset' + gt_test)
  config.set('data paths', 'test_border_masks', setup[1] + '_datasets/dataset' + borderMasks_test)

  config.set('training settings', 'N_subimgs', eval(setup[0] + '_subimgs'))
  config.set('testing settings' , 'full_images_to_test', eval(setup[1] + '_testimgs'))

  with open('configuration.txt', "w") as f:
    config.write(f)

  ### run experiment
  os.system('python run_training.py')
  os.system('python run_testing.py')
  # break
