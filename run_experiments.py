
import configparser
import os

global_config = configparser.RawConfigParser()
global_config.read('./global_config.txt')


stats_parser = configparser.RawConfigParser()
stats_parser.read('./DRIVE_datasets/stats_train.txt')

subimgs_per_img = int(stats_parser.get('statistics', 'subimages_per_image'))

DRIVE_imgs_train = int(global_config.get('DRIVE', 'N_imgs_train'))
DRIVE_imgs_test = int(global_config.get('DRIVE', 'N_imgs_test'))

DRIVE_subimgs = subimgs_per_img * DRIVE_imgs_train
DRIVE_testimgs = subimgs_per_img * DRIVE_imgs_test

Synth_imgs_train = int(global_config.get('Synth', 'N_imgs_train'))
Synth_imgs_test = int(global_config.get('Synth', 'N_imgs_test'))

Synth_subimgs = subimgs_per_img * Synth_imgs_train
Synth_testimgs = subimgs_per_img * Synth_imgs_test

imgs_to_visualize = global_config.get('global', 'imgs_to_visualize')

# first is train second test
settings = ['Synth', 'DRIVE']
archs = ['unet'] #['unet','resnet']

for arch in archs:
  for trainset in settings:
    config = configparser.RawConfigParser()
    config.read('./configuration_template.txt')
    ### write config
    experiment = trainset + '_training'
    config.set('experiment', 'name', experiment)
    config.set('experiment', 'arch', arch)
    
    config.set('data paths', 'train_data_path', './' + trainset + '_datasets/dataset__train*.tfrecord')
    config.set('data paths', 'train_data_stats', './' + trainset + '_datasets/stats_train.txt')

    config.set('training settings', 'N_subimgs', eval(trainset + '_subimgs'))

    with open('configuration.txt', "w") as f:
      config.write(f)
    
    ### run training
    os.system('python run_training.py')
    
    for testset in settings:
      config.set('experiment', 'testset', experiment)
      config.set('data paths', 'test_data_path', './' + testset + '_datasets/dataset__test*.tfrecord')
      config.set('data paths', 'test_data_stats', './' + testset + '_datasets/stats_test.txt')
      config.set('testing settings', 'N_subimgs', eval(testset + '_testimgs'))
      config.set('testing settings', 'imgs_to_visualize', imgs_to_visualize)
      
      with open('configuration.txt', "w") as f:
        config.write(f)

      os.system('python run_testing.py')
      break
    break
