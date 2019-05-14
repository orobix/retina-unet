###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import configparser
sys.path.insert(0, './lib/')
from help_functions import mkdir_p


#config file to read from
config = configparser.RawConfigParser()
config.read('./configuration.txt')
#===========================================
#name of the experiment
path_data = config.get('data paths', 'path_local')
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
experiment_path = path_data + '/' + name_experiment + '_' + arch

config_path = experiment_path + '/' + name_experiment + '_configuration.txt'

nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results
mkdir_p(experiment_path)

print("copy the configuration file in the results folder")
if sys.platform=='win32':
    os.system('copy configuration.txt ' + config_path.replace('/', '\\'))
else:
    os.system('cp configuration.txt ' + config_path)

# run the experiment
if nohup:
    print("\n2. Run the training on GPU with nohup")
    os.system(run_GPU +' nohup python -u ./src/retinaNN_training.py > ' + experiment_path + '/' + name_experiment + '_training.nohup')
else:
    print("\n2. Run the training on GPU (no nohup)")
    os.system(run_GPU +' python ./src/retinaNN_training.py')

#Prediction/testing is run with a different script
print("Done!")