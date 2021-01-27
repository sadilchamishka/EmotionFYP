IOTRAIN="-i new_audio_llds_train.csv -o new_xbow_train.arff -B codebook"

java -Xmx10000m -jar openXBOW.jar $IOTRAIN -standardizeInput -log -size 1000 -a 5 -attributes nt1[65]2[65]

IOTEST="-i new_audio_llds_test.csv -o new_xbow_test.arff -b codebook"

!java -Xmx10000m -jar openXBOW.jar $IOTEST -attributes nt1[65]2[65]