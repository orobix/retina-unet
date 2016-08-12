###################################################
#
#   Script to launch the training
#
##################################################

import os
import ConfigParser


#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

run_GPU = ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results
result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
os.system('mkdir -p ' +result_dir)

print "copy the configuration file in the results folder"
os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

# run the experiment
if nohup:
    print "\n2. Run the training on GPU with nohup"
    os.system(run_GPU +' nohup python -u ./src/retinaNN_training.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
else:
    print "\n2. Run the training on GPU (no nohup)"
    os.system(run_GPU +' python ./src/retinaNN_training.py')

#Prediction/testing is run with a different script
