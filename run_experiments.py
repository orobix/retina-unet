
import configparser
import os

subimgs_per_img = 104 * 104

DRIVE_subimgs = subimgs_per_img * 20
DRIVE_testimgs = subimgs_per_img * 20

Synth_subimgs = subimgs_per_img * 9000
Synth_testimgs = subimgs_per_img * 1000

# first is train second test
settings = [['DRIVE', 'DRIVE'], ['DRIVE', 'Synth'], ['Synth', 'DRIVE'], ['Synth', 'Synth']]


for setup in settings:
  config = configparser.RawConfigParser()
  config.read('./configuration_template.txt')
  ### write config
  experiment = setup[0] + '_' + setup[1] + '_experiment'
  config.set('experiment name', 'name', experiment)

  #config.set('data paths', 'path_local', experiment)
  config.set('data paths', 'train_data_path', setup[0] + '_datasets/dataset__train*.tfrecord')
  config.set('data paths', 'test_data_path', setup[1] + '_datasets/dataset__test*.tfrecord')

  config.set('training settings', 'N_subimgs', eval(setup[0] + '_subimgs'))
  config.set('testing settings' , 'full_images_to_test', eval(setup[1] + '_testimgs'))

  with open('configuration.txt', "w") as f:
    config.write(f)

  ### run experiment
  os.system('python3 run_training.py')
  # os.system('python run_testing.py')
  break
