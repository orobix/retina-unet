import os, sys
import configparser


#config file to read from
config = configparser.RawConfigParser()
config.readfp(open(r'./global_config.txt'))

DRIVE = 'DRIVE'
Synth = 'Synth'

experiments = [
    [DRIVE, DRIVE],
    [DRIVE, Synth],
    [Synth, DRIVE],
    [Synth, Synth],
]

for experiment in experiments:
    # write configuration.txt
    # run training
    # run testing