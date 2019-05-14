###################################################
#
#   Script to execute the prediction
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


# finally run the prediction
if nohup:
    print("\n2. Run the prediction on GPU  with nohup")
    os.system(run_GPU +' nohup python -u ./src/retinaNN_predict.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
    print("\n2. Run the prediction on GPU (no nohup)")
    os.system(run_GPU +' python ./src/retinaNN_predict.py')

print("Done!")
