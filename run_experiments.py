
import configparser
import os

global_config = configparser.RawConfigParser()
global_config.read('./global_config.txt')

subimgs_per_dim = int(global_config.get('global', 'subimgs_per_dim'))
subimgs_per_img = subimgs_per_dim**2

DRIVE_imgs_train = int(global_config.get('DRIVE', 'N_imgs_train'))
DRIVE_imgs_test = int(global_config.get('DRIVE', 'N_imgs_test'))

DRIVE_subimgs = subimgs_per_img * DRIVE_imgs_train
DRIVE_testimgs = subimgs_per_img * DRIVE_imgs_test


Synth_imgs_train = int(global_config.get('Synth', 'N_imgs_train'))
Synth_imgs_test = int(global_config.get('Synth', 'N_imgs_test'))

Synth_subimgs = subimgs_per_img * Synth_imgs_train
Synth_testimgs = subimgs_per_img * Synth_imgs_test

subimgs_per_img = subimgs_per_dim**2
imgs_to_visualize = global_config.get('global', 'imgs_to_visualize')

# first is train second test
settings = ['DRIVE', 'Synth']
archs = ['resnet'] #['unet','resnet']

for arch in archs:
  for trainset in settings:
    config = configparser.RawConfigParser()
    config.read('./configuration_template.txt')
    ### write config
    experiment = trainset + '_DRIVE_experiment'
    config.set('experiment', 'name', experiment)
    config.set('experiment', 'arch', arch)
    
    config.set('data paths', 'train_data_path', './' + trainset + '_datasets/dataset__train*.tfrecord')
    config.set('data paths', 'train_data_stats', './' + trainset + '_datasets/stats_train.txt')

    config.set('training settings', 'N_subimgs', eval(trainset + '_subimgs'))

    ### run training
    os.system('python run_training.py')
    
    # for testset in settings:
    #   config.set('data paths', 'test_data_path', './' + testset + '_datasets/dataset__test*.tfrecord')
    #   config.set('data paths', 'test_data_stats', './' + testset + '_datasets/stats_test.txt')
    #   config.set('testing settings', 'N_subimgs', eval(testset + '_subimgs'))
    #   config.set('testing settings', 'imgs_to_visualize', imgs_to_visualize)

    #   with open('configuration.txt', "w") as f:
    #     config.write(f)

    #   os.system('python run_testing.py')
    #   break
    break
