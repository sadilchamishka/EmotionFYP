#!/bin/python
# python2
# Extract acoustic features (only llds) (csv)

import os

# Audio folders
folder_audio_train = "/home/madhavi/Downloads/train_wav/"  # Folder containing the audio files of the training partition
folder_audio_test  = "/home/madhavi/Downloads/test_wav/"   # Folder containing the audio files of the test partition

#SMILExtract -C config/ComParE_2016.conf -I exampleaudio/opensmile.wav -O speechenergy.csv

# openSMILE
exe_opensmile = "/home/madhavi/Downloads/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract"  # MODIFY this path to the SMILExtract (version 2.3) executable
path_config   = "/home/madhavi/Downloads/opensmile-2.3.0/config/ComParE_2016.conf"                     # MODIFY this path to the config file of openSMILE 2.3 - under Windows (cygwin): no POSIX

# Output files
outfilename_train = "new_audio_llds_train.csv"
outfilename_test  = "new_audio_llds_test.csv"

# Clear LLD files
if os.path.exists(outfilename_train):
    os.remove(outfilename_train)
if os.path.exists(outfilename_test):
    os.remove(outfilename_test)

opensmile_options = "-configfile " + path_config + " -appendcsvlld 1 -timestampcsvlld 1 -headercsvlld 1"

# Extract features for train
for fn in os.listdir(folder_audio_train):
    infilename   = folder_audio_train + fn
    instancename = os.path.splitext(fn)[0]
    outfilename  = outfilename_train
    opensmile_call = exe_opensmile + " " + opensmile_options + " -inputfile " + infilename + " -lldcsvoutput " + outfilename + " -instname " + instancename
    os.system(opensmile_call)

# Extract features for test
for fn in os.listdir(folder_audio_test):
    infilename   = folder_audio_test + fn
    instancename = os.path.splitext(fn)[0]
    outfilename  = outfilename_test
    opensmile_call = exe_opensmile + " " + opensmile_options + " -inputfile " + infilename + " -lldcsvoutput " + outfilename + " -instname " + instancename
    os.system(opensmile_call)